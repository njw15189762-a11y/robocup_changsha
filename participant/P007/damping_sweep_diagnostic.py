"""H004 可见目标覆盖线外末段阻尼单变量扫描。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.protocol import EpisodeContext  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402
from entry import DeterministicAssignmentPolicy  # noqa: E402
from train_hybrid import public_params  # noqa: E402


SUITE_PATH = Path(__file__).resolve().parent / "dev/dev-suite-v1.yaml"
DAMPING_VALUES = (1.2, 0.6, 0.0)


def run_case(case, damping: float, source: str = "predicted") -> dict:
    """运行单个开发集场景并保留逐目标最近距离。"""
    env = make_training_env(case.task_config)
    observations, _ = env.reset(seed=case.scenario_seed)
    policies = {}
    for agent_index, agent_id in enumerate(env.agents):
        policy = DeterministicAssignmentPolicy()
        policy._VISIBLE_APPROACH_DAMPING = float(damping)
        policy._VISIBLE_DAMPING_DISTANCE_SOURCE = source
        policy.reset(
            EpisodeContext(
                agent_index=agent_index,
                num_agents=env.config.num_agents,
                num_targets=env.config.num_targets,
                horizon=env.config.horizon,
                task=public_params(env.config),
                policy_seed=agent_index,
            )
        )
        policies[agent_id] = policy

    snapshot = env._current_snapshot
    if snapshot is None:
        raise RuntimeError("环境全局状态尚未初始化")
    minimum_distances = np.min(
        np.linalg.norm(
            snapshot.robot_positions[:, None, :]
            - snapshot.target_positions[None, :, :],
            axis=2,
        ),
        axis=0,
    )
    coverage_values = []
    collision_values = []
    for _ in range(env.config.horizon):
        actions = {
            agent_id: policies[agent_id].act(observation)
            for agent_id, observation in observations.items()
        }
        observations, _, _, _, infos = env.step(actions)
        snapshot = env._current_snapshot
        if snapshot is None:
            raise RuntimeError("环境推进后全局状态缺失")
        distances = np.min(
            np.linalg.norm(
                snapshot.robot_positions[:, None, :]
                - snapshot.target_positions[None, :, :],
                axis=2,
            ),
            axis=0,
        )
        minimum_distances = np.minimum(minimum_distances, distances)
        metrics = next(iter(infos.values()))["metrics"]
        coverage_values.append(float(metrics.coverage_rate))
        collision_values.append(float(metrics.collision_rate))
    env.close()

    mean_coverage = float(np.mean(coverage_values))
    mean_collision = float(np.mean(collision_values))
    return {
        "case_id": case.case_id,
        "scenario_seed": case.scenario_seed,
        "mean_j": mean_coverage - 0.2 * mean_collision,
        "mean_coverage": mean_coverage,
        "mean_collision": mean_collision,
        "minimum_target_distances": minimum_distances.tolist(),
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "mean_j": float(np.mean([row["mean_j"] for row in rows])),
        "mean_coverage": float(np.mean([row["mean_coverage"] for row in rows])),
        "mean_collision": float(np.mean([row["mean_collision"] for row in rows])),
        "zero_coverage_episodes": int(
            sum(row["mean_coverage"] <= 1e-12 for row in rows)
        ),
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P007 H004/H005 末段阻尼扫描")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=SUITE_PATH)
    parser.add_argument(
        "--source", choices=("predicted", "observed"), default="predicted"
    )
    parser.add_argument("--damping", nargs="+", type=float, default=DAMPING_VALUES)
    args = parser.parse_args()
    suite = load_suite(args.suite)

    results = []
    for damping in args.damping:
        groups = {}
        for group in suite.groups:
            rows = [run_case(case, damping, args.source) for case in group.cases]
            groups[group.group_id] = summarize(rows)
        score = 500.0 * sum(group["mean_j"] for group in groups.values())
        result = {
            "visible_approach_damping": damping,
            "distance_source": args.source,
            "performance_score": score,
            "groups": groups,
        }
        results.append(result)
        print(
            f"source={args.source} damping={damping:.1f} score={score:.4f} "
            f"groups={[(key, round(group['mean_j'], 6)) for key, group in groups.items()]} "
            f"zeros={sum(group['zero_coverage_episodes'] for group in groups.values())}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"experiment": "H004-H005-damping-sweep", "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"详细结果：{args.output}")


if __name__ == "__main__":
    main()
