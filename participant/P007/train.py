"""参赛模板训练脚本（学习基线示例）：SB3 PPO，训练环境与评测同源。

在训练专用 venv 内运行（torch/sb3/supersuit 仅训练 venv 安装，评测镜像不含）：
    .venv-train/Scripts/python.exe train.py --total-steps 1000000 --curve-points 100
关键约束：训练栈不用 supersuit 的 flatten_observation_v0（其对 Discrete 字段做
one-hot 展开，与评测侧特征顺序不一致）；本文件 FlattenCoverageVecEnv 直接调用
coverage_bench.spaces.flatten_observation，训练与评测共用同一 flatten 实现。

实现说明（针对实际安装版本 stable-baselines3 2.9.0 / supersuit 3.11.0 的三处适配，
flatten 语义与 VecNormalize 设置与设计保持一致）：
1. AliveAgentsBridge：CoverageParallelEnv.possible_agents 按容量给出（agent_capacity），
   而每回合仅 config.num_agents 个 agent 在场；supersuit 的 pettingzoo_env_to_vec_env_v1
   要求 reset 后所有 possible_agents 均有观测，故在桥接处收拢 possible_agents。
2. FlattenCoverageVecEnv.seed/reset：supersuit 3.11 的 ConcatVecEnv 没有 seed() 方法，
   且 SB3 2.9 的 learn() 不再调用 env.seed()，种子只能经 reset(seed=...) 传入；
   因此 seed() 暂存种子，下一次 reset() 时经 ConcatVecEnv.reset(seed) 下发
   （ConcatVecEnv 会给第 i 个副本 seed+i，直达 CoverageParallelEnv.reset(seed=...)）。
3. ScenarioScheduleWrapper 使用独立、递增的环境种子轮换 uniform/crossing；
   RecordingVecMonitor 同时记录回报和两类实际完成回合数。
4. 每段完整 PPO rollout 更新后保存模型与归一化统计，并在独立的 40 回合固定
   模型选择集上评测，避免用训练回报或最终模型直接选 checkpoint。
"""
import argparse
from collections import Counter
import json
import platform
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import supersuit as ss
import yaml
from pettingzoo.utils.wrappers import BaseParallelWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import get_protocol_spec
from coverage_bench.spaces import flatten_observation, flattened_observation_size
from coverage_bench.suites import load_suite

_PUBLIC_SUITE = _REPO_ROOT / "configs" / "public-suite-v1.yaml"
_STRATEGY_PATH = Path(__file__).resolve().parent / "training-strategy-v1.yaml"
TRAINING_SEED_DEFAULT = 7101  # PPO 初始化种子，与环境场景种子分离
ENVIRONMENT_SEED_DEFAULT = 51001


class ScenarioScheduleWrapper(BaseParallelWrapper):
    """按确定性种子序列轮换训练布局，并把布局信息写入每步 info。"""

    def __init__(self, env, layouts: tuple[str, ...], seed_start: int, seed_stride: int):
        super().__init__(env)
        if not layouts:
            raise ValueError("训练布局列表不能为空")
        self._layouts = layouts
        self._configs = {
            layout: env.config.model_copy(
                update={
                    "scenario": env.config.scenario.model_copy(
                        update={"layout_kind": layout}
                    )
                }
            )
            for layout in layouts
        }
        self._seed_start = int(seed_start)
        self._seed_stride = int(seed_stride)
        self._next_seed: int | None = None
        self._current_layout = layouts[0]
        self._current_seed = self._seed_start

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._next_seed = int(seed)
        elif self._next_seed is None:
            self._next_seed = self._seed_start

        episode_seed = int(self._next_seed)
        self._next_seed += self._seed_stride
        layout = self._layouts[episode_seed % len(self._layouts)]
        self.env.config = self._configs[layout]
        self._current_layout = layout
        self._current_seed = episode_seed
        return self.env.reset(seed=episode_seed, options=options)

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        for info in infos.values():
            info["training_layout"] = self._current_layout
            info["training_episode_seed"] = self._current_seed
        return observations, rewards, terminations, truncations, infos


class AliveAgentsBridge(BaseParallelWrapper):
    """把 possible_agents 对齐到场景实际存活的 agent（num_agents 个）。

    CoverageParallelEnv.possible_agents 按容量 agent_capacity 给出，而每回合只有
    config.num_agents 个 agent 在场；supersuit 的 pettingzoo_env_to_vec_env_v1
    要求 reset 后所有 possible_agents 都有观测，故在训练栈内做本桥接包装。
    """

    def __init__(self, env):
        super().__init__(env)
        self.possible_agents = [f"agent_{i}" for i in range(env.config.num_agents)]

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.possible_agents = list(self.env.agents)
        return obs, infos


class PotentialRewardWrapper(BaseParallelWrapper):
    """加入目标接近势函数奖励，仅供训练使用。

    势函数取每个目标与最近机器人距离的截断接近度均值。距离超过感知半径时
    接近度为零，因此不会使用全局状态向策略泄露远处目标方向；全局状态只用于
    生成训练奖励，策略得到的观测仍是协议规定的局部观测。
    """

    def __init__(
        self,
        env,
        *,
        alpha: float,
        gamma: float,
        total_world_steps: int | None = None,
        decay_start_fraction: float | None = None,
        decay_end_fraction: float | None = None,
    ):
        super().__init__(env)
        if alpha <= 0.0:
            raise ValueError("势函数奖励系数 alpha 必须为正数")
        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self._total_world_steps = total_world_steps
        self._decay_start_fraction = decay_start_fraction
        self._decay_end_fraction = decay_end_fraction
        self._world_steps = 0
        self._spec = get_protocol_spec()
        self._previous_potential = 0.0
        self._official_return = 0.0
        self._shaping_return = 0.0
        self._alpha_sum = 0.0
        if total_world_steps is not None:
            if total_world_steps < 1:
                raise ValueError("退火调度的总世界步数必须为正整数")
            if not (
                0.0 <= float(decay_start_fraction) < float(decay_end_fraction) <= 1.0
            ):
                raise ValueError("退火起止比例必须满足 0 <= start < end <= 1")

    def _current_alpha(self) -> float:
        """根据累计世界步数返回当前塑形系数。"""
        if self._total_world_steps is None:
            return self._alpha
        progress = min(1.0, self._world_steps / self._total_world_steps)
        start = float(self._decay_start_fraction)
        end = float(self._decay_end_fraction)
        if progress <= start:
            return self._alpha
        if progress >= end:
            return 0.0
        return self._alpha * (end - progress) / (end - start)

    def _potential(self) -> float:
        """计算 [0, 1] 范围内的全队目标接近度。"""
        state = np.asarray(self.env.state(), dtype=np.float64)
        agent_capacity = self._spec.agent_capacity
        target_capacity = self._spec.target_capacity
        robot_table = state[: 5 * agent_capacity].reshape(agent_capacity, 5)
        robot_exists = state[5 * agent_capacity : 6 * agent_capacity] > 0.5
        target_offset = 6 * agent_capacity
        target_table = state[
            target_offset : target_offset + 5 * target_capacity
        ].reshape(target_capacity, 5)
        target_exists = state[
            target_offset + 5 * target_capacity : target_offset + 6 * target_capacity
        ] > 0.5

        robot_positions = robot_table[robot_exists, :2]
        target_positions = target_table[target_exists, :2]
        if len(robot_positions) == 0 or len(target_positions) == 0:
            return 0.0

        distances = np.linalg.norm(
            robot_positions[:, None, :] - target_positions[None, :, :], axis=2
        )
        nearest = np.min(distances, axis=0)
        sense_radius = float(self.env.config.public.sense_radius)
        closeness = 1.0 - np.clip(nearest / sense_radius, 0.0, 1.0)
        return float(np.mean(closeness))

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self._previous_potential = self._potential()
        self._official_return = 0.0
        self._shaping_return = 0.0
        self._alpha_sum = 0.0
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        current_potential = self._potential()
        current_alpha = self._current_alpha()
        shaping = current_alpha * (
            self._gamma * current_potential - self._previous_potential
        )
        self._previous_potential = current_potential
        self._world_steps += 1

        agent_ids = list(rewards)
        official = float(rewards[agent_ids[0]]) if agent_ids else 0.0
        self._official_return += official
        self._shaping_return += shaping
        self._alpha_sum += current_alpha
        shaped_rewards = {
            agent_id: float(reward + shaping)
            for agent_id, reward in rewards.items()
        }
        episode_done = bool(agent_ids) and all(
            bool(terminations[agent_id] or truncations[agent_id])
            for agent_id in agent_ids
        )
        for info in infos.values():
            terms = info.get("reward_terms")
            if terms is not None:
                terms["official_team_reward"] = official
                terms["potential_shaping"] = float(shaping)
                terms["potential_alpha"] = float(current_alpha)
                terms["training_team_reward"] = float(official + shaping)
            if episode_done:
                info["episode_official_return"] = float(self._official_return)
                info["episode_shaping_return"] = float(self._shaping_return)
                info["episode_mean_potential_alpha"] = float(
                    self._alpha_sum / max(1, self.env.config.horizon)
                )
        return observations, shaped_rewards, terminations, truncations, infos


class FlattenCoverageVecEnv(VecEnvWrapper):
    """把 agent 拆分后的 Dict 观测压平为与评测侧完全一致的 104 维向量。"""

    def __init__(self, venv):
        super().__init__(venv)
        self._spec = get_protocol_spec()
        size = flattened_observation_size(self._spec)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self._pending_seed = None

    @staticmethod
    def _split_batch(obs):
        """把 gymnasium concatenate 的 dict-of-(n,...) 批量观测拆回 n 个单环境 Dict。"""
        num_envs = len(next(iter(obs.values())))
        return [{key: value[i] for key, value in obs.items()} for i in range(num_envs)]

    def seed(self, seed=None):
        """暂存种子，供下一次 reset() 消费（见模块 docstring 适配 2）。"""
        self._pending_seed = seed
        return [seed]

    def reset(self):
        if self._pending_seed is not None:
            # 绕过 SB3VecEnvWrapper（其 reset(seed) 会调用不存在的 ConcatVecEnv.seed），
            # 直接经 ConcatVecEnv.reset(seed=...) 把种子传给各副本的 pettingzoo 环境。
            obs = self.venv.venv.reset(seed=self._pending_seed)[0]
            self._pending_seed = None
        else:
            obs = self.venv.reset()
        # SB3 VecEnv 约定 reset 返回 (num_envs, obs_dim) 的 ndarray（VecNormalize 有断言）
        return np.stack([flatten_observation(item, self._spec) for item in self._split_batch(obs)])

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        stacked = np.stack([flatten_observation(item, self._spec) for item in self._split_batch(obs)])
        return stacked, rewards, dones, infos


class RecordingVecMonitor(VecMonitor):
    """补记完成 episode 回报历史的 VecMonitor（SB3 2.9 的 VecMonitor 无此功能）。"""

    def __init__(self, venv):
        super().__init__(venv)
        self._completed_rewards = []
        self._completed_official_returns = []
        self._completed_shaping_returns = []
        self._completed_mean_alphas = []
        self._layout_episodes = Counter()

    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()
        for i, done in enumerate(dones):
            if done and isinstance(infos[i], dict) and "episode" in infos[i]:
                self._completed_rewards.append(float(infos[i]["episode"]["r"]))
                layout = infos[i].get("training_layout")
                if layout is not None:
                    self._layout_episodes[str(layout)] += 1
                if "episode_official_return" in infos[i]:
                    self._completed_official_returns.append(
                        float(infos[i]["episode_official_return"])
                    )
                if "episode_shaping_return" in infos[i]:
                    self._completed_shaping_returns.append(
                        float(infos[i]["episode_shaping_return"])
                    )
                if "episode_mean_potential_alpha" in infos[i]:
                    self._completed_mean_alphas.append(
                        float(infos[i]["episode_mean_potential_alpha"])
                    )
        return obs, rewards, dones, infos

    def get_episode_rewards(self):
        return self._completed_rewards

    def get_layout_episode_counts(self):
        # 每个团队回合会为三台机器人各产生一个 done，只记录团队回合数。
        return {
            layout: count // 3
            for layout, count in sorted(self._layout_episodes.items())
        }

    def get_reward_component_returns(self):
        """返回已经结束回合的官方与塑形回报，供 checkpoint 记录。"""
        return {
            "official": self._completed_official_returns,
            "shaping": self._completed_shaping_returns,
            "mean_alpha": self._completed_mean_alphas,
        }


def build_env_stack(
    num_vec_envs: int,
    seed: int,
    layouts: tuple[str, ...] = ("uniform",),
    reward_config: dict | None = None,
    gamma: float = 0.99,
    total_model_steps: int | None = None,
):
    """构造到 FlattenCoverageVecEnv 为止的训练栈（不含 VecMonitor/VecNormalize）。"""
    suite = load_suite(_PUBLIC_SUITE)
    case = suite.groups[0].cases[0]
    env = make_training_env(case.task_config)
    env = ScenarioScheduleWrapper(
        env,
        layouts=layouts,
        seed_start=seed,
        seed_stride=num_vec_envs,
    )
    reward_config = reward_config or {"kind": "official"}
    if reward_config["kind"] == "official_plus_potential":
        env = PotentialRewardWrapper(
            env,
            alpha=float(reward_config["potential_alpha"]),
            gamma=float(gamma),
        )
    elif reward_config["kind"] == "official_plus_potential_schedule":
        if total_model_steps is None:
            raise ValueError("退火奖励需要提供 total_model_steps")
        transitions_per_world_step = num_vec_envs * case.task_config.num_agents
        if total_model_steps % transitions_per_world_step != 0:
            raise ValueError(
                "总训练步数必须能被 num_vec_envs * num_agents 整除，才能精确执行退火"
            )
        env = PotentialRewardWrapper(
            env,
            alpha=float(reward_config["initial_alpha"]),
            gamma=float(gamma),
            total_world_steps=total_model_steps // transitions_per_world_step,
            decay_start_fraction=float(reward_config["decay_start_fraction"]),
            decay_end_fraction=float(reward_config["decay_end_fraction"]),
        )
    elif reward_config["kind"] != "official":
        raise ValueError(f"不支持的训练奖励类型: {reward_config['kind']}")
    env = AliveAgentsBridge(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=0, base_class="stable_baselines3")
    env = FlattenCoverageVecEnv(env)
    env.seed(seed)
    return env


def _load_training_strategy() -> dict:
    """读取 P007 训练实验协议。"""
    return yaml.safe_load(_STRATEGY_PATH.read_text(encoding="utf-8"))


def _save_normalization_stats(env: VecNormalize, path: Path) -> None:
    """以审计允许的 npz 格式保存观测归一化统计量。"""
    np.savez(
        path,
        obs_mean=np.asarray(env.obs_rms.mean, dtype=np.float64),
        obs_var=np.asarray(env.obs_rms.var, dtype=np.float64),
        obs_eps=np.array(env.epsilon, dtype=np.float64),
        obs_clip=np.array(env.clip_obs, dtype=np.float64),
    )


def _normalize_observations(raw: np.ndarray, env: VecNormalize) -> np.ndarray:
    return np.clip(
        (raw - np.asarray(env.obs_rms.mean, dtype=np.float64))
        / np.sqrt(np.asarray(env.obs_rms.var, dtype=np.float64) + float(env.epsilon)),
        -float(env.clip_obs),
        float(env.clip_obs),
    ).astype(np.float32)


def evaluate_model_selection(model: PPO, norm_env: VecNormalize, strategy: dict) -> dict:
    """在独立的 uniform/crossing 固定种子上评估当前 checkpoint。"""
    suite = load_suite(_PUBLIC_SUITE)
    base_config = suite.groups[0].cases[0].task_config
    spec = get_protocol_spec()
    selection = strategy["data"]["model_selection"]
    group_results = {}

    for layout in ("uniform", "crossing"):
        start, end = selection[f"{layout}_seeds"]
        config = base_config.model_copy(
            update={
                "scenario": base_config.scenario.model_copy(
                    update={"layout_kind": layout}
                )
            }
        )
        episode_rows = []
        for scenario_seed in range(int(start), int(end) + 1):
            env = make_training_env(config)
            observations, _ = env.reset(seed=scenario_seed)
            step_j = []
            step_coverage = []
            step_collision = []
            for _ in range(config.horizon):
                agent_ids = list(observations)
                raw = np.stack(
                    [flatten_observation(observations[agent_id], spec) for agent_id in agent_ids]
                )
                normalized = _normalize_observations(raw, norm_env)
                actions, _ = model.predict(normalized, deterministic=True)
                action_dict = {
                    agent_id: np.asarray(actions[index], dtype=np.float32)
                    for index, agent_id in enumerate(agent_ids)
                }
                observations, _, _, _, infos = env.step(action_dict)
                metrics = infos[agent_ids[0]]["metrics"]
                coverage = float(metrics.coverage_rate)
                collision = float(metrics.collision_rate)
                step_coverage.append(coverage)
                step_collision.append(collision)
                step_j.append(coverage - config.collision_weight * collision)
            env.close()
            episode_rows.append(
                {
                    "seed": scenario_seed,
                    "mean_j": float(np.mean(step_j)),
                    "mean_coverage": float(np.mean(step_coverage)),
                    "mean_collision": float(np.mean(step_collision)),
                }
            )

        group_results[layout] = {
            "mean_j": float(np.mean([row["mean_j"] for row in episode_rows])),
            "mean_coverage": float(np.mean([row["mean_coverage"] for row in episode_rows])),
            "mean_collision": float(np.mean([row["mean_collision"] for row in episode_rows])),
            "zero_coverage_episodes": int(
                sum(row["mean_coverage"] <= 1e-12 for row in episode_rows)
            ),
            "episodes": episode_rows,
        }

    score = 500.0 * (
        group_results["uniform"]["mean_j"]
        + group_results["crossing"]["mean_j"]
    )
    return {
        "performance_score": float(score),
        "groups": group_results,
    }


class TrainingCheckpointRecorder:
    """在每段 PPO 参数更新完成后保存 checkpoint 并运行独立评测。"""

    def __init__(
        self,
        model: PPO,
        out_dir: Path,
        norm_env: VecNormalize,
        strategy: dict,
    ):
        self._model = model
        self._out_dir = out_dir
        self._norm_env = norm_env
        self._strategy = strategy
        self._history = []

    def _monitor(self) -> RecordingVecMonitor:
        return self._norm_env.venv

    def _write_history(self) -> None:
        path = self._out_dir / "checkpoint_history.json"
        path.write_text(
            json.dumps(self._history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        columns = [
            "total_steps",
            "mean_episode_reward_last100",
            "mean_official_return_last100",
            "mean_shaping_return_last100",
            "mean_potential_alpha_last100",
            "selection_score",
            "uniform_mean_j",
            "crossing_mean_j",
            "uniform_collision",
            "crossing_collision",
            "uniform_zero_coverage",
            "crossing_zero_coverage",
            "train_uniform_episodes",
            "train_crossing_episodes",
        ]
        lines = [",".join(columns)]
        for row in self._history:
            groups = row["model_selection"]["groups"]
            counts = row["layout_episode_counts"]
            values = [
                row["total_steps"],
                row["mean_episode_reward_last100"],
                row["mean_official_return_last100"],
                row["mean_shaping_return_last100"],
                row["mean_potential_alpha_last100"],
                row["model_selection"]["performance_score"],
                groups["uniform"]["mean_j"],
                groups["crossing"]["mean_j"],
                groups["uniform"]["mean_collision"],
                groups["crossing"]["mean_collision"],
                groups["uniform"]["zero_coverage_episodes"],
                groups["crossing"]["zero_coverage_episodes"],
                counts.get("uniform", 0),
                counts.get("crossing", 0),
            ]
            lines.append(",".join("" if value is None else str(value) for value in values))
        (self._out_dir / "training_curve.csv").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        best = max(
            self._history,
            key=lambda row: (
                row["model_selection"]["performance_score"],
                min(
                    row["model_selection"]["groups"]["uniform"]["mean_j"],
                    row["model_selection"]["groups"]["crossing"]["mean_j"],
                ),
                -sum(
                    row["model_selection"]["groups"][layout]["mean_collision"]
                    for layout in ("uniform", "crossing")
                ),
            ),
        )
        best_summary = {
            "label": best["label"],
            "total_steps": best["total_steps"],
            "selection_score": best["model_selection"]["performance_score"],
            "groups": {
                layout: {
                    key: best["model_selection"]["groups"][layout][key]
                    for key in (
                        "mean_j",
                        "mean_coverage",
                        "mean_collision",
                        "zero_coverage_episodes",
                    )
                }
                for layout in ("uniform", "crossing")
            },
        }
        (self._out_dir / "best_checkpoint.json").write_text(
            json.dumps(best_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def save_snapshot(self, label: str) -> dict:
        checkpoint_dir = self._out_dir / "checkpoints" / label
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._model.save(str(checkpoint_dir / "model"))
        _save_normalization_stats(
            self._norm_env,
            checkpoint_dir / "vecnormalize-stats.npz",
        )

        rewards = self._monitor().get_episode_rewards()
        components = self._monitor().get_reward_component_returns()
        validation = evaluate_model_selection(self._model, self._norm_env, self._strategy)
        row = {
            "label": label,
            "total_steps": int(self._model.num_timesteps),
            "mean_episode_reward_last100": (
                float(np.mean(rewards[-100:])) if rewards else None
            ),
            "mean_official_return_last100": (
                float(np.mean(components["official"][-100:]))
                if components["official"] else None
            ),
            "mean_shaping_return_last100": (
                float(np.mean(components["shaping"][-100:]))
                if components["shaping"] else None
            ),
            "mean_potential_alpha_last100": (
                float(np.mean(components["mean_alpha"][-100:]))
                if components["mean_alpha"] else None
            ),
            "layout_episode_counts": self._monitor().get_layout_episode_counts(),
            "model_selection": validation,
        }
        self._history.append(row)
        self._write_history()
        print(
            "[checkpoint] "
            f"label={label} steps={self._model.num_timesteps} "
            f"selection_score={validation['performance_score']:.4f}"
        )
        return row


def train(args):
    strategy = _load_training_strategy()
    common = strategy["common"]
    experiment = strategy["experiments"][args.experiment]
    layouts = tuple(
        layout
        for layout, weight in experiment["layout_weights"].items()
        if float(weight) > 0.0
    )
    total_steps = int(
        args.total_steps
        if args.total_steps is not None
        else common["total_timesteps"]
    )
    checkpoint_interval = int(
        args.checkpoint_interval
        if args.checkpoint_interval is not None
        else common["checkpoint_interval"]
    )
    if total_steps < 1 or checkpoint_interval < 1:
        raise ValueError("训练步数和 checkpoint 间隔必须为正整数")
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"训练输出目录必须不存在或为空: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    flat_env = build_env_stack(
        args.num_vec_envs,
        args.env_seed_start,
        layouts,
        reward_config=experiment["reward"],
        gamma=float(common["ppo"]["gamma"]),
        total_model_steps=total_steps,
    )
    monitor = RecordingVecMonitor(flat_env)
    env = VecNormalize(monitor, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        n_steps=int(common["ppo"]["n_steps"]),
        batch_size=int(common["ppo"]["batch_size"]),
        learning_rate=float(common["ppo"]["learning_rate"]),
        gamma=float(common["ppo"]["gamma"]),
        gae_lambda=float(common["ppo"]["gae_lambda"]),
        verbose=1,
    )
    recorder = TrainingCheckpointRecorder(
        model=model,
        out_dir=out_dir,
        norm_env=env,
        strategy=strategy,
    )

    started = time.time()
    while model.num_timesteps < total_steps:
        remaining = total_steps - model.num_timesteps
        step_chunk = min(checkpoint_interval, remaining)
        model.learn(total_timesteps=step_chunk, reset_num_timesteps=False)
        recorder.save_snapshot(f"step-{model.num_timesteps:09d}")
    wall = time.time() - started

    model.save(str(out_dir / "model"))
    _save_normalization_stats(env, out_dir / "vecnormalize-stats.npz")
    meta = {
        "experiment": args.experiment,
        "model_seed": args.seed,
        "environment_seed_start": args.env_seed_start,
        "layouts": list(layouts),
        "reward": experiment["reward"],
        "requested_total_steps": total_steps,
        "actual_total_steps": model.num_timesteps,
        "checkpoint_interval": checkpoint_interval,
        "wall_seconds": round(wall, 1),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "num_vec_envs": args.num_vec_envs,
        "strategy_config": str(_STRATEGY_PATH.relative_to(_REPO_ROOT)),
        "layout_episode_counts": monitor.get_layout_episode_counts(),
    }
    (out_dir / "training_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    env.close()
    print(f"训练完成: {model.num_timesteps} 步, {wall:.1f}s → {out_dir}")


def check_export(args):
    """导出一致性检查：同一批环境观测上比较 SB3 原模型与 numpy 导出模型的确定性动作。

    覆盖观测预处理（flatten 同源函数）、归一化、网络前向、动作裁剪与 dtype 转换。
    """
    import torch

    model_dir = Path(args.model_dir)
    model = PPO.load(str(model_dir / "model.zip"), device="cpu")
    norm = np.load(model_dir / "vecnormalize-stats.npz")
    data = np.load(Path(args.npz))

    # 真实环境 rollout 收集未归一化的 104 维观测（与训练路径同栈、同 flatten）
    env = build_env_stack(args.num_vec_envs, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    raw = []
    obs = env.reset()
    while len(raw) < args.num_obs:
        raw.extend(np.asarray(o, dtype=np.float64) for o in obs)
        actions = [rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) for _ in range(len(obs))]
        obs, _, _, _ = env.step(actions)
    raw = np.stack(raw[: args.num_obs])

    # SB3 路径：用训练侧统计量归一化（先除后 clip，与 SB3 VecNormalize.normalize_obs 一致）
    x_sb3 = np.clip(
        (raw - np.asarray(norm["obs_mean"], dtype=np.float64)) / np.sqrt(np.asarray(norm["obs_var"], dtype=np.float64) + float(norm["obs_eps"])),
        -float(norm["obs_clip"]), float(norm["obs_clip"]),
    )
    # numpy 路径：完全用 npz 内导出统计量，验证统计量逐位一致
    x_np = np.clip(
        (raw - data["obs_mean"]) / np.sqrt(data["obs_var"] + data["obs_eps"]),
        -data["obs_clip"], data["obs_clip"],
    )
    assert np.allclose(x_np, x_sb3, rtol=0.0, atol=1e-12), "npz 归一化统计量与训练侧统计量不一致"

    with torch.no_grad():
        sb3_actions, _ = model.predict(x_sb3, deterministic=True)
    sb3_actions = np.asarray(sb3_actions, dtype=np.float64)

    h = np.tanh(x_np @ data["W1"] + data["b1"])
    h = np.tanh(h @ data["W2"] + data["b2"])
    np_actions = np.clip(h @ data["W3"] + data["b3"], -1.0, 1.0)

    max_diff = float(np.max(np.abs(sb3_actions - np_actions)))
    env.close()
    print(f"样本数: {len(x_sb3)}  max|Δa|: {max_diff:.3e}  容差: 1e-5")
    if max_diff > 1e-5:
        raise SystemExit(f"导出一致性检查失败: max|Δa|={max_diff:.3e} > 1e-5")
    print("导出一致性检查通过")


def main():
    parser = argparse.ArgumentParser(description="P007 可复现 PPO 训练与 checkpoint 评测")
    parser.add_argument(
        "--experiment", choices=("T000", "T001", "T002", "T003"), default="T000"
    )
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--num-vec-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED_DEFAULT)
    parser.add_argument("--env-seed-start", type=int, default=ENVIRONMENT_SEED_DEFAULT)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "outputs" / "P007" / "training",
    )
    parser.add_argument("--check-export", action="store_true", help="运行导出一致性检查后退出")
    parser.add_argument("--model-dir", type=Path, default=_REPO_ROOT / "outputs" / "learning-training")
    parser.add_argument("--npz", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "policy.npz")
    parser.add_argument("--num-obs", type=int, default=256)
    args = parser.parse_args()
    if args.check_export:
        check_export(args)
        return
    train(args)


if __name__ == "__main__":
    main()
