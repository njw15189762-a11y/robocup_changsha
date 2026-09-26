"""混合策略训练：规则负责追踪，PPO 只学习严格搜索状态的动作残差。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch as th
import torch.nn.functional as F
import yaml
from gymnasium import spaces
from pettingzoo.utils.wrappers import BaseParallelWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.protocol import EpisodeContext, PublicTaskParams, get_protocol_spec  # noqa: E402
from coverage_bench.spaces import flatten_observation, flattened_observation_size  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402
from entry import DeterministicAssignmentPolicy  # noqa: E402
from train import AliveAgentsBridge, ScenarioScheduleWrapper  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parent / "hybrid-training-v1.yaml"
PUBLIC_SUITE = REPO_ROOT / "configs" / "public-suite-v1.yaml"


def public_params(config) -> PublicTaskParams:
    """把环境公开参数转换为单机器人规则策略的回合上下文。"""
    task = config.public
    return PublicTaskParams(
        map_half_extent=task.map_half_extent,
        dt=task.dt,
        robot_radius=task.robot_radius,
        robot_mass=task.robot_mass,
        drive_force=task.drive_force,
        damping=task.damping,
        robot_max_speed=task.robot_max_speed,
        contact_force=task.contact_force,
        contact_margin=task.contact_margin,
        target_radius=task.target_radius,
        target_max_speed=task.target_max_speed,
        sense_radius=task.sense_radius,
        motion_kind=task.motion_kind,
        turn_interval_steps=tuple(task.turn_interval_steps),
        target_speed_fraction=tuple(task.target_speed_fraction),
        robot_boundary=task.robot_boundary,
        target_boundary=task.target_boundary,
    )


class HybridSearchWrapper(BaseParallelWrapper):
    """执行规则动作，仅在严格搜索状态叠加 PPO 残差并奖励首次发现。"""

    _MASK_KEY = "search_mask"

    def __init__(
        self,
        env,
        *,
        residual_scale: float,
        discovery_bonus: float,
        discovery_schedule: str = "constant",
    ):
        super().__init__(env)
        self._residual_scale = float(residual_scale)
        self._discovery_bonus = float(discovery_bonus)
        if discovery_schedule not in {"constant", "remaining_linear"}:
            raise ValueError(f"未知首次发现奖励日程: {discovery_schedule}")
        self._discovery_schedule = discovery_schedule
        self._policies: dict[str, DeterministicAssignmentPolicy] = {}
        self._observations: dict = {}
        self._rule_actions: dict[str, np.ndarray] = {}
        self._search_masks: dict[str, bool] = {}
        self._ever_seen_targets: set[int] = set()
        self._official_return = 0.0
        self._discovery_return = 0.0
        self._search_steps = 0
        self._team_steps = 0

    def observation_space(self, agent):
        base = self.env.observation_space(agent)
        augmented = dict(base.spaces)
        augmented[self._MASK_KEY] = gym.spaces.Box(
            low=0.0, high=1.0, shape=(), dtype=np.float32
        )
        return gym.spaces.Dict(augmented)

    def _reset_policies(self, seed: int | None) -> None:
        self._policies = {}
        for index, agent_id in enumerate(self.env.agents):
            policy = DeterministicAssignmentPolicy()
            policy.reset(
                EpisodeContext(
                    agent_index=index,
                    num_agents=self.env.config.num_agents,
                    num_targets=self.env.config.num_targets,
                    horizon=self.env.config.horizon,
                    task=public_params(self.env.config),
                    policy_seed=max(0, int(seed or 0) + index),
                )
            )
            self._policies[agent_id] = policy

    def _prepare(self, observations: dict) -> dict:
        """每个状态只推进一次规则内部记忆，并缓存对应规则动作与搜索门。"""
        self._observations = observations
        self._rule_actions = {}
        self._search_masks = {}
        augmented = {}
        for agent_id, observation in observations.items():
            policy = self._policies[agent_id]
            self._rule_actions[agent_id] = policy.act(observation)
            search = policy._last_control_mode == policy._MODE_LEARNED_SEARCH
            self._search_masks[agent_id] = search
            row = dict(observation)
            row[self._MASK_KEY] = np.float32(search)
            augmented[agent_id] = row
        return augmented

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self._reset_policies(seed)
        self._ever_seen_targets = {
            target_index
            for observation in observations.values()
            for target_index, visible in enumerate(observation["target_visible"])
            if bool(visible)
        }
        self._official_return = 0.0
        self._discovery_return = 0.0
        self._search_steps = 0
        self._team_steps = 0
        return self._prepare(observations), infos

    def _search_action(self, agent_id: str, residual: np.ndarray) -> np.ndarray:
        """在与正式策略相同的位置加入搜索残差，再执行规则避碰和裁剪。"""
        observation = self._observations[agent_id]
        policy = self._policies[agent_id]
        desired = policy._search_direction(observation)
        distance = float(np.linalg.norm(desired))
        direction = (
            desired / distance
            if distance > policy._EPS
            else np.zeros(2, dtype=np.float64)
        )
        velocity = np.asarray(observation["self_state"][2:4], dtype=np.float64)
        drive = min(1.0, 2.5 * distance) * direction - 0.35 * velocity
        drive += self._residual_scale * np.asarray(residual, dtype=np.float64)
        drive += 0.9 * policy._avoidance(observation)
        return np.clip(drive, -1.0, 1.0).astype(np.float32)

    def _discovery_weight(self) -> float:
        """让早期发现获得更高奖励，末步发现因无法继续追踪而记为零。"""
        if self._discovery_schedule == "constant":
            return 1.0
        horizon = int(self.env.config.horizon)
        return float(
            np.clip(
                (horizon - self._team_steps) / max(1, horizon - 1),
                0.0,
                1.0,
            )
        )

    def step(self, actions):
        applied_actions = {}
        previous_masks = dict(self._search_masks)
        for agent_id in self.env.agents:
            if previous_masks[agent_id]:
                applied_actions[agent_id] = self._search_action(agent_id, actions[agent_id])
                self._search_steps += 1
            else:
                applied_actions[agent_id] = self._rule_actions[agent_id]
        self._team_steps += 1

        observations, rewards, terminations, truncations, infos = self.env.step(
            applied_actions
        )
        official = float(next(iter(rewards.values()))) if rewards else 0.0
        self._official_return += official

        bonuses = {agent_id: 0.0 for agent_id in rewards}
        newly_seen = {
            target_index
            for observation in observations.values()
            for target_index, visible in enumerate(observation["target_visible"])
            if bool(visible) and target_index not in self._ever_seen_targets
        }
        for target_index in newly_seen:
            discoverers = [
                agent_id
                for agent_id, observation in observations.items()
                if previous_masks.get(agent_id, False)
                and bool(observation["target_visible"][target_index])
            ]
            if discoverers:
                weighted_bonus = self._discovery_bonus * self._discovery_weight()
                value = weighted_bonus / (
                    self.env.config.num_targets * len(discoverers)
                )
                for agent_id in discoverers:
                    bonuses[agent_id] += value
                self._discovery_return += weighted_bonus / self.env.config.num_targets
        self._ever_seen_targets.update(newly_seen)

        shaped_rewards = {
            agent_id: float(reward + bonuses[agent_id])
            for agent_id, reward in rewards.items()
        }
        episode_done = bool(rewards) and all(
            bool(terminations[agent_id] or truncations[agent_id])
            for agent_id in rewards
        )
        for agent_id, info in infos.items():
            # SuperSuit 在截断后自动 reset，reset info 会覆盖同名 metrics；另存供评测。
            info["hybrid_metrics"] = info.get("metrics")
            info["search_mask"] = float(previous_masks.get(agent_id, False))
            info["discovery_bonus"] = float(bonuses.get(agent_id, 0.0))
            if episode_done:
                info["episode_official_return"] = float(self._official_return)
                info["episode_discovery_return"] = float(self._discovery_return)
                info["episode_search_fraction"] = float(
                    self._search_steps
                    / max(1, self._team_steps * self.env.config.num_agents)
                )
        return (
            self._prepare(observations),
            shaped_rewards,
            terminations,
            truncations,
            infos,
        )


class HybridFlattenVecEnv(VecEnvWrapper):
    """压平合法104维观测，并把搜索门作为第105维训练特征。"""

    def __init__(self, venv):
        super().__init__(venv)
        self._spec = get_protocol_spec()
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            (flattened_observation_size(self._spec) + 1,),
            np.float32,
        )
        self._pending_seed = None

    @staticmethod
    def _split_batch(obs):
        count = len(next(iter(obs.values())))
        return [{key: value[index] for key, value in obs.items()} for index in range(count)]

    def _flatten(self, item: dict) -> np.ndarray:
        search_mask = np.float32(item.pop(HybridSearchWrapper._MASK_KEY))
        base = flatten_observation(item, self._spec)
        return np.concatenate((base, np.array([search_mask], dtype=np.float32)))

    def seed(self, seed=None):
        self._pending_seed = seed
        return [seed]

    def reset(self):
        if self._pending_seed is not None:
            obs = self.venv.venv.reset(seed=self._pending_seed)[0]
            self._pending_seed = None
        else:
            obs = self.venv.reset()
        return np.stack([self._flatten(item) for item in self._split_batch(obs)])

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for info in infos:
            terminal = info.get("terminal_observation")
            if isinstance(terminal, dict):
                info["terminal_observation"] = self._flatten(dict(terminal))
        stacked = np.stack([self._flatten(item) for item in self._split_batch(obs)])
        return stacked, rewards, dones, infos


class SearchMaskVecNormalize(VecNormalize):
    """标准化104维环境特征，但保持末位搜索门严格为0或1。"""

    def normalize_obs(self, obs):
        normalized = super().normalize_obs(obs)
        if isinstance(obs, np.ndarray):
            normalized[..., -1] = obs[..., -1]
        return normalized


class MaskedPPO(PPO):
    """价值函数使用全部样本，策略与熵损失只使用搜索门为1的样本。"""

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses = [], [], []
        clip_fractions, approx_kl_divs, mask_fractions = [], [], []
        last_loss = th.zeros((), device=self.device)
        for _epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                search_mask = rollout_data.observations[:, -1] > 0.5
                mask_fractions.append(float(search_mask.float().mean().item()))

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -float(clip_range_vf),
                        float(clip_range_vf),
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(float(value_loss.item()))

                if bool(th.any(search_mask)):
                    advantages = rollout_data.advantages[search_mask]
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )
                    masked_log_prob = log_prob[search_mask]
                    masked_old_log_prob = rollout_data.old_log_prob[search_mask]
                    ratio = th.exp(masked_log_prob - masked_old_log_prob)
                    policy_loss = -th.min(
                        advantages * ratio,
                        advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
                    ).mean()
                    if entropy is None:
                        entropy_loss = masked_log_prob.mean()
                    else:
                        entropy_loss = -entropy[search_mask].mean()
                    clip_fraction = th.mean(
                        (th.abs(ratio - 1) > clip_range).float()
                    )
                    with th.no_grad():
                        log_ratio = masked_log_prob - masked_old_log_prob
                        approx_kl = th.mean(
                            (th.exp(log_ratio) - 1) - log_ratio
                        ).item()
                else:
                    policy_loss = log_prob.sum() * 0.0
                    entropy_loss = log_prob.sum() * 0.0
                    clip_fraction = th.zeros((), device=self.device)
                    approx_kl = 0.0

                pg_losses.append(float(policy_loss.item()))
                entropy_losses.append(float(entropy_loss.item()))
                clip_fractions.append(float(clip_fraction.item()))
                approx_kl_divs.append(float(approx_kl))
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                last_loss = loss
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/search_sample_fraction", np.mean(mask_fractions))
        self.logger.record("train/loss", float(last_loss.item()))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)


def build_stack(
    *,
    num_vec_envs: int,
    seed: int,
    layouts: tuple[str, ...],
    residual_scale: float,
    discovery_bonus: float,
    discovery_schedule: str = "constant",
):
    suite = load_suite(PUBLIC_SUITE)
    env = make_training_env(suite.groups[0].cases[0].task_config)
    env = ScenarioScheduleWrapper(
        env,
        layouts=layouts,
        seed_start=seed,
        seed_stride=num_vec_envs,
    )
    env = HybridSearchWrapper(
        env,
        residual_scale=residual_scale,
        discovery_bonus=discovery_bonus,
        discovery_schedule=discovery_schedule,
    )
    env = AliveAgentsBridge(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=num_vec_envs,
        num_cpus=0,
        base_class="stable_baselines3",
    )
    env = HybridFlattenVecEnv(env)
    env.seed(seed)
    return env


def normalize_for_model(raw: np.ndarray, env: SearchMaskVecNormalize) -> np.ndarray:
    """使用训练统计归一化，并保持搜索门不变。"""
    return np.asarray(env.normalize_obs(raw), dtype=np.float32)


def evaluate_model(model: MaskedPPO, norm_env: SearchMaskVecNormalize, config: dict) -> dict:
    """用完整混合动作在固定40回合集上评测搜索残差。"""
    results = {}
    residual_scale = float(config["experiment"]["residual_scale"])
    discovery_bonus = float(config["experiment"]["discovery_bonus"])
    discovery_schedule = str(
        config["experiment"].get("discovery_schedule", "constant")
    )
    for layout in ("uniform", "crossing"):
        start, end = config["model_selection"][f"{layout}_seeds"]
        rows = []
        for scenario_seed in range(int(start), int(end) + 1):
            env = build_stack(
                num_vec_envs=1,
                seed=scenario_seed,
                layouts=(layout,),
                residual_scale=residual_scale,
                discovery_bonus=discovery_bonus,
                discovery_schedule=discovery_schedule,
            )
            raw = env.reset()
            step_j, step_coverage, step_collision = [], [], []
            for _ in range(10):
                actions, _ = model.predict(
                    normalize_for_model(raw, norm_env), deterministic=True
                )
                raw, _, _, infos = env.step(actions)
                metrics = infos[0]["hybrid_metrics"]
                coverage = float(metrics.coverage_rate)
                collision = float(metrics.collision_rate)
                step_coverage.append(coverage)
                step_collision.append(collision)
                step_j.append(coverage - 0.2 * collision)
            env.close()
            rows.append(
                {
                    "mean_j": float(np.mean(step_j)),
                    "mean_coverage": float(np.mean(step_coverage)),
                    "mean_collision": float(np.mean(step_collision)),
                }
            )
        results[layout] = {
            "mean_j": float(np.mean([row["mean_j"] for row in rows])),
            "mean_coverage": float(np.mean([row["mean_coverage"] for row in rows])),
            "mean_collision": float(np.mean([row["mean_collision"] for row in rows])),
            "zero_coverage_episodes": int(
                sum(row["mean_coverage"] <= 1e-12 for row in rows)
            ),
        }
    return {
        "performance_score": 500.0 * (
            results["uniform"]["mean_j"] + results["crossing"]["mean_j"]
        ),
        "groups": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P007 混合搜索训练")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    total_steps = int(args.total_steps or training["total_timesteps"])
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"输出目录必须为空或不存在: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    flat_env = build_stack(
        num_vec_envs=int(training["num_vec_envs"]),
        seed=int(training["environment_seed_start"]),
        layouts=tuple(config["experiment"]["layouts"]),
        residual_scale=float(config["experiment"]["residual_scale"]),
        discovery_bonus=float(config["experiment"]["discovery_bonus"]),
        discovery_schedule=str(
            config["experiment"].get("discovery_schedule", "constant")
        ),
    )
    monitor = VecMonitor(flat_env)
    env = SearchMaskVecNormalize(
        monitor, norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    model = MaskedPPO(
        "MlpPolicy",
        env,
        seed=int(training["model_seed"]),
        n_steps=int(training["n_steps"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]),
        policy_kwargs={"net_arch": list(training["net_arch"])},
        verbose=1,
    )
    # 确定性初始策略输出严格为零残差；随机策略仍由 log_std 提供探索。
    with th.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()

    started = time.time()
    model.learn(total_timesteps=total_steps)
    elapsed = time.time() - started
    evaluation = evaluate_model(model, env, config)
    model.save(str(args.out / "model"))
    np.savez(
        args.out / "vecnormalize-stats.npz",
        obs_mean=np.asarray(env.obs_rms.mean, dtype=np.float64),
        obs_var=np.asarray(env.obs_rms.var, dtype=np.float64),
        obs_eps=np.array(env.epsilon, dtype=np.float64),
        obs_clip=np.array(env.clip_obs, dtype=np.float64),
    )
    summary = {
        "experiment": config["experiment"],
        "total_steps": int(model.num_timesteps),
        "wall_seconds": round(elapsed, 1),
        "model_selection": evaluation,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    env.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
