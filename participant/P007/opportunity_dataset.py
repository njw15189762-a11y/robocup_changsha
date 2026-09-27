"""H010：仅在本地训练期生成规则相对反事实标签，特征只含合法局部信息。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.spaces import flatten_observation  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402
from local_intervention_diagnostic import (  # noqa: E402
    RESIDUALS,
    _baseline,
    _candidate,
    _mean_j,
)

PUBLIC_SUITE = REPO_ROOT / "configs/public-suite-v1.yaml"


def _local_features(frame: dict, agent_id: str, target: int, spec) -> np.ndarray:
    """104 维协议观测，加规则动作、规则目标编号和本机历史速度估计。"""
    observation = frame["observations"][agent_id]
    policy = frame["policies"][agent_id]
    target_onehot = np.zeros(policy._num_targets, dtype=np.float32)
    target_onehot[target] = 1.0
    return np.concatenate((
        flatten_observation(observation, spec),
        np.asarray(frame["actions"][agent_id], dtype=np.float32),
        target_onehot,
        np.asarray(policy._target_velocities[target], dtype=np.float32),
    )).astype(np.float32)


def collect_case(case, *, episode_id: int, scale: float,
                 intervention_steps: int, max_step_index: int
                 ) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    env = make_training_env(case.task_config)
    try:
        env.reset(seed=case.scenario_seed)
        assert env._scenario_state is not None
        frames, baseline_metrics = _baseline(
            env._scenario_state, env.spec, case.scenario_seed
        )
        baseline_j = _mean_j(baseline_metrics, env.config.collision_weight)
        features, deltas, metadata = [], [], []
        for frame in frames:
            if int(frame["state"].step_index) > max_step_index:
                continue
            for agent_id, policy in frame["policies"].items():
                if (
                    policy._last_control_mode != policy._MODE_VISIBLE_TRACK
                    or policy._assigned_target is None
                ):
                    continue
                target = int(policy._assigned_target)
                feature = _local_features(frame, agent_id, target, env.spec)
                outcomes = np.asarray([
                    _candidate(
                        frame, agent_id, target, residual, scale,
                        intervention_steps,
                    )[0] - baseline_j
                    for residual in RESIDUALS
                ], dtype=np.float32)
                features.append(feature)
                deltas.append(outcomes)
                metadata.append({
                    "episode_id": episode_id,
                    "seed": case.scenario_seed,
                    "layout": case.task_config.scenario.layout_kind,
                    "step_index": int(frame["state"].step_index),
                    "agent_index": int(policy._agent_index),
                    "target_index": target,
                    "rule_mean_j": baseline_j,
                })
        return features, deltas, metadata
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="H010 独立场景反事实数据集生成")
    parser.add_argument("--uniform-start", type=int, required=True)
    parser.add_argument("--crossing-start", type=int, required=True)
    parser.add_argument("--per-layout", type=int, required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--intervention-steps", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--max-step-index", type=int, default=4,
        help="只学习回合早期；H009 已知最佳介入在第 0～4 步",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.per_layout < 1:
        raise ValueError("每种布局至少需要一个回合")
    if args.out.exists():
        raise FileExistsError(f"输出文件已存在，避免覆盖旧实验：{args.out}")
    suite = load_suite(PUBLIC_SUITE)
    base = suite.groups[0].cases[0].task_config
    rows = []
    episode_id = 0
    for layout, start in (
        ("uniform", args.uniform_start),
        ("crossing", args.crossing_start),
    ):
        task = base.model_copy(update={
            "scenario": base.scenario.model_copy(update={"layout_kind": layout})
        })
        for seed in range(start, start + args.per_layout):
            case = SimpleNamespace(scenario_seed=seed, task_config=task)
            rows.append(collect_case(
                case, episode_id=episode_id, scale=args.scale,
                intervention_steps=args.intervention_steps,
                max_step_index=args.max_step_index,
            ))
            episode_id += 1
            if episode_id % 25 == 0:
                states = sum(len(item[0]) for item in rows)
                positives = sum(
                    int(np.sum(np.max(item[1], axis=1) > 1e-12))
                    for item in rows if item[1]
                )
                print(
                    f"回合 {episode_id}/{2 * args.per_layout}: "
                    f"追踪状态 {states}，有益状态 {positives}",
                    flush=True,
                )
    features = np.concatenate([
        np.stack(item[0]) for item in rows if item[0]
    ]).astype(np.float32)
    deltas = np.concatenate([
        np.stack(item[1]) for item in rows if item[1]
    ]).astype(np.float32)
    metadata = [row for item in rows for row in item[2]]
    episode_ids = np.asarray([row["episode_id"] for row in metadata], dtype=np.int32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=features,
        deltas=deltas,
        episode_ids=episode_ids,
        seeds=np.asarray([row["seed"] for row in metadata], dtype=np.int32),
        steps=np.asarray([row["step_index"] for row in metadata], dtype=np.int8),
        agents=np.asarray([row["agent_index"] for row in metadata], dtype=np.int8),
        targets=np.asarray([row["target_index"] for row in metadata], dtype=np.int8),
        residuals=np.stack(RESIDUALS),
    )
    summary = {
        "episode_count": episode_id,
        "state_count": int(len(features)),
        "feature_dim": int(features.shape[1]),
        "helpful_states": int(np.sum(np.max(deltas, axis=1) > 1e-12)),
        "helpful_actions": int(np.sum(deltas > 1e-12)),
        "harmful_actions": int(np.sum(deltas < -1e-12)),
        "scale": args.scale,
        "intervention_steps": args.intervention_steps,
        "max_step_index": args.max_step_index,
        "seed_ranges": {
            "uniform": [args.uniform_start, args.uniform_start + args.per_layout - 1],
            "crossing": [args.crossing_start, args.crossing_start + args.per_layout - 1],
        },
        "warning": "训练标签使用本地反事实真值；推理输入仅由合法局部观测构成。",
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
