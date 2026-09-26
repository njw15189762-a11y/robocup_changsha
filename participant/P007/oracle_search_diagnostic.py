"""H003 搜索 Oracle 可救性诊断；仅用于本地开发，不进入正式策略。"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import EpisodeContext
from coverage_bench.suites import load_suite
from entry import DeterministicAssignmentPolicy
from train_hybrid import PUBLIC_SUITE, public_params


CONFIG_PATH = Path(__file__).resolve().parent / "hybrid-training-h002.yaml"
CONTROLLERS = ("rule", "nearest", "assignment")
SCALES = (0.35, 0.7, 1.0)


def _visible_targets(observations: dict) -> set[int]:
    """汇总团队局部观测中当前可见的目标编号。"""
    return {
        target_index
        for observation in observations.values()
        for target_index, visible in enumerate(observation["target_visible"])
        if bool(visible)
    }


def _one_to_one_assignment(
    agent_indices: list[int],
    target_indices: list[int],
    robot_positions: np.ndarray,
    target_positions: np.ndarray,
) -> dict[int, int]:
    """穷举至多 3 对匹配，返回总距离最小的一对一分配。"""
    if not agent_indices or not target_indices:
        return {}

    best_cost = float("inf")
    best_pairs: tuple[tuple[int, int], ...] = ()
    if len(agent_indices) <= len(target_indices):
        candidates = (
            tuple(zip(agent_indices, targets))
            for targets in itertools.permutations(target_indices, len(agent_indices))
        )
    else:
        candidates = (
            tuple(zip(agents, target_indices))
            for agents in itertools.permutations(agent_indices, len(target_indices))
        )

    for pairs in candidates:
        cost = sum(
            float(np.linalg.norm(robot_positions[agent] - target_positions[target]))
            for agent, target in pairs
        )
        tie_key = tuple(pairs)
        if cost < best_cost - 1e-12 or (
            abs(cost - best_cost) <= 1e-12 and tie_key < best_pairs
        ):
            best_cost = cost
            best_pairs = pairs
    return dict(best_pairs)


def _oracle_targets(
    controller: str,
    strict_agents: list[int],
    unseen_targets: list[int],
    robot_positions: np.ndarray,
    target_positions: np.ndarray,
) -> dict[int, int]:
    """为严格搜索机器人选择真实但尚未被团队看到的目标。"""
    if controller == "assignment":
        return _one_to_one_assignment(
            strict_agents,
            unseen_targets,
            robot_positions,
            target_positions,
        )
    if controller == "nearest":
        return {
            agent: min(
                unseen_targets,
                key=lambda target: (
                    float(
                        np.linalg.norm(
                            robot_positions[agent] - target_positions[target]
                        )
                    ),
                    target,
                ),
            )
            for agent in strict_agents
        } if unseen_targets else {}
    return {}


def _oracle_action(
    policy: DeterministicAssignmentPolicy,
    observation: dict,
    target_position: np.ndarray,
    residual_scale: float,
) -> np.ndarray:
    """在当前残差幅度约束内，尽量把规则搜索驱动力改成目标方向。"""
    search_vector = policy._search_direction(observation)
    search_distance = float(np.linalg.norm(search_vector))
    search_direction = (
        search_vector / search_distance
        if search_distance > policy._EPS
        else np.zeros(2, dtype=np.float64)
    )
    velocity = np.asarray(observation["self_state"][2:4], dtype=np.float64)
    base_drive = (
        min(1.0, 2.5 * search_distance) * search_direction - 0.35 * velocity
    )

    self_position = np.asarray(observation["self_state"][:2], dtype=np.float64)
    target_vector = np.asarray(target_position, dtype=np.float64) - self_position
    # 与正式可见追踪相同地逐轴使用最大接近力，构造残差接口能实现的强 Oracle。
    target_drive = np.clip(5.0 * target_vector, -1.0, 1.0)

    # PPO 残差动作限制在 [-1,1]；尺度越小，能纠正规则航点的幅度越有限。
    residual = np.clip(
        (target_drive - base_drive) / residual_scale,
        -1.0,
        1.0,
    )
    drive = base_drive + residual_scale * residual
    drive += 0.9 * policy._avoidance(observation)
    return np.clip(drive, -1.0, 1.0).astype(np.float32)


def run_episode(*, layout: str, seed: int, controller: str, scale: float) -> dict:
    """运行一个固定场景，并统计发现时点和官方逐步指标。"""
    suite = load_suite(PUBLIC_SUITE)
    env = make_training_env(suite.groups[0].cases[0].task_config)
    env.config = env.config.model_copy(
        update={
            "scenario": env.config.scenario.model_copy(
                update={"layout_kind": layout}
            )
        }
    )
    observations, _ = env.reset(seed=seed)
    policies = {}
    for agent_index, agent_id in enumerate(env.agents):
        policy = DeterministicAssignmentPolicy()
        policy.reset(
            EpisodeContext(
                agent_index=agent_index,
                num_agents=env.config.num_agents,
                num_targets=env.config.num_targets,
                horizon=env.config.horizon,
                task=public_params(env.config),
                policy_seed=seed + agent_index,
            )
        )
        policies[agent_id] = policy

    ever_seen = _visible_targets(observations)
    first_seen_steps = {target: 0 for target in ever_seen}
    step_coverage = []
    step_collision = []
    oracle_steps = 0

    for step_index in range(1, env.config.horizon + 1):
        actions = {
            agent_id: policies[agent_id].act(observation)
            for agent_id, observation in observations.items()
        }
        strict_agent_ids = [
            agent_id
            for agent_id, policy in policies.items()
            if policy._last_control_mode == policy._MODE_LEARNED_SEARCH
        ]
        strict_indices = [int(agent_id.rsplit("_", 1)[1]) for agent_id in strict_agent_ids]
        unseen_targets = [
            target
            for target in range(env.config.num_targets)
            if target not in ever_seen
        ]
        snapshot = env._current_snapshot
        if snapshot is None:
            raise RuntimeError("环境全局状态尚未初始化")
        assignments = _oracle_targets(
            controller,
            strict_indices,
            unseen_targets,
            snapshot.robot_positions,
            snapshot.target_positions,
        )
        for agent_index, target_index in assignments.items():
            agent_id = f"agent_{agent_index}"
            actions[agent_id] = _oracle_action(
                policies[agent_id],
                observations[agent_id],
                snapshot.target_positions[target_index],
                scale,
            )
            oracle_steps += 1

        observations, _, _, _, infos = env.step(actions)
        currently_seen = _visible_targets(observations)
        for target in currently_seen - ever_seen:
            first_seen_steps[target] = step_index
        ever_seen.update(currently_seen)
        metrics = next(iter(infos.values()))["metrics"]
        step_coverage.append(float(metrics.coverage_rate))
        step_collision.append(float(metrics.collision_rate))

    env.close()
    mean_coverage = float(np.mean(step_coverage))
    mean_collision = float(np.mean(step_collision))
    return {
        "seed": seed,
        "mean_j": mean_coverage - 0.2 * mean_collision,
        "mean_coverage": mean_coverage,
        "mean_collision": mean_collision,
        "targets_seen": len(ever_seen),
        "first_seen_steps": first_seen_steps,
        "oracle_steps": oracle_steps,
    }


def summarize(rows: list[dict]) -> dict:
    """汇总同一布局的 20 个固定回合。"""
    discovery_steps = [
        int(step)
        for row in rows
        for step in row["first_seen_steps"].values()
        if int(step) > 0
    ]
    return {
        "mean_j": float(np.mean([row["mean_j"] for row in rows])),
        "mean_coverage": float(np.mean([row["mean_coverage"] for row in rows])),
        "mean_collision": float(np.mean([row["mean_collision"] for row in rows])),
        "zero_coverage_episodes": int(
            sum(row["mean_coverage"] <= 1e-12 for row in rows)
        ),
        "mean_targets_seen": float(np.mean([row["targets_seen"] for row in rows])),
        "new_discoveries": len(discovery_steps),
        "mean_first_seen_step": (
            float(np.mean(discovery_steps)) if discovery_steps else None
        ),
        "oracle_steps": int(sum(row["oracle_steps"] for row in rows)),
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P007 H003 搜索 Oracle 诊断")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    experiments = []
    settings = [("rule", 0.0)] + [
        (controller, scale)
        for controller in CONTROLLERS[1:]
        for scale in SCALES
    ]
    for controller, scale in settings:
        groups = {}
        for layout in ("uniform", "crossing"):
            start, end = config["model_selection"][f"{layout}_seeds"]
            rows = [
                run_episode(
                    layout=layout,
                    seed=seed,
                    controller=controller,
                    scale=scale,
                )
                for seed in range(int(start), int(end) + 1)
            ]
            groups[layout] = summarize(rows)
        score = 500.0 * (groups["uniform"]["mean_j"] + groups["crossing"]["mean_j"])
        experiments.append(
            {
                "controller": controller,
                "residual_scale": scale,
                "performance_score": score,
                "groups": groups,
            }
        )
        print(
            f"{controller:>10} scale={scale:.2f} score={score:.4f} "
            f"uniform={groups['uniform']['mean_j']:.6f} "
            f"crossing={groups['crossing']['mean_j']:.6f} "
            f"zeros={groups['uniform']['zero_coverage_episodes'] + groups['crossing']['zero_coverage_episodes']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"experiment": "H003-oracle", "results": experiments}, indent=2),
        encoding="utf-8",
    )
    print(f"详细结果：{args.output}")


if __name__ == "__main__":
    main()
