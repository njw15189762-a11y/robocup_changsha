"""核对正式入口与 H016 实验控制器在相同种子上的闭环得分。"""

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
from assignment_control_ablation import _groups, VisibleGain12Policy  # noqa: E402
from local_intervention_diagnostic import _baseline, _mean_j  # noqa: E402
from train_hybrid import public_params  # noqa: E402


def _verify(kind: str, per_layout: int, path: Path) -> dict:
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        int(row["seed"]): float(row["candidate_mean_j"])
        for group in saved["groups"].values()
        for row in group["cases"]
    }
    errors = []
    action_errors = []
    actual_seeds = []
    for _, cases in _groups(kind, per_layout):
        for case in cases:
            env = make_training_env(case.task_config)
            try:
                env.reset(seed=case.scenario_seed)
                assert env._scenario_state is not None
                frames, metrics = _baseline(
                    env._scenario_state, env.spec, case.scenario_seed
                )
                actual = _mean_j(metrics, env.config.collision_weight)
                errors.append(abs(actual - expected[case.scenario_seed]))
                actual_seeds.append(case.scenario_seed)
                policies = {}
                for index in range(env.config.num_agents):
                    policy = VisibleGain12Policy()
                    policy.reset(EpisodeContext(
                        agent_index=index,
                        num_agents=env.config.num_agents,
                        num_targets=env.config.num_targets,
                        horizon=env.config.horizon,
                        task=public_params(env.config),
                        policy_seed=case.scenario_seed + index,
                    ))
                    policies[f"agent_{index}"] = policy
                for frame in frames:
                    for agent, observation in frame["observations"].items():
                        candidate_action = policies[agent].act(observation)
                        action_errors.append(float(np.max(np.abs(
                            candidate_action - frame["actions"][agent]
                        ))))
            finally:
                env.close()
    if set(actual_seeds) != set(expected):
        raise AssertionError("入口验证种子与实验记录不一致")
    return {
        "dataset": kind,
        "episode_count": len(errors),
        "max_score_error": max(errors, default=None),
        "max_action_error": max(action_errors, default=None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="H016 正式入口逐回合分数复现")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    args = parser.parse_args()
    result = [
        _verify("selection", 20, args.selection),
        _verify("fresh", 60, args.fresh),
    ]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if any(
        item["max_score_error"] > 1e-9 or item["max_action_error"] > 1e-7
        for item in result
    ):
        raise SystemExit("正式入口与 H016 实验记录不一致")


if __name__ == "__main__":
    main()
