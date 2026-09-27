"""H011：诊断 H010 机会标签的几何分布与参数稳定性。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.suites import load_suite
from local_intervention_diagnostic import RESIDUALS, _baseline, _candidate, _mean_j


SUITE = REPO_ROOT / "configs/public-suite-v1.yaml"


def _read(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.array(data[key]) for key in data.files}


def _features(data: dict[str, np.ndarray], agent_capacity: int) -> dict[str, np.ndarray]:
    x = data["features"]
    rows = np.arange(len(x))
    target_start = 5 + 7 * agent_capacity
    rel = x[rows[:, None], target_start + 3 * data["targets"][:, None] + np.arange(2)]
    distance = np.linalg.norm(rel, axis=1)
    unit = rel / np.maximum(distance[:, None], 1e-8)
    relative_velocity = x[:, -2:] - x[:, 2:4]
    action = x[:, 104:106]
    self_pos = x[:, :2]
    return {
        "distance": distance,
        "radial_speed": np.sum(relative_velocity * unit, axis=1),
        "lateral_speed": np.abs(
            relative_velocity[:, 0] * unit[:, 1]
            - relative_velocity[:, 1] * unit[:, 0]
        ),
        "action_saturation": np.max(np.abs(action), axis=1),
        "target_boundary_margin": 0.85 - np.max(np.abs(self_pos + rel), axis=1),
    }


def _slice_summary(values: np.ndarray, mask: np.ndarray) -> dict:
    chosen = values[mask]
    return {
        "count": int(len(chosen)),
        "median": float(np.median(chosen)) if len(chosen) else None,
        "q25": float(np.quantile(chosen, 0.25)) if len(chosen) else None,
        "q75": float(np.quantile(chosen, 0.75)) if len(chosen) else None,
    }


def _distribution(data: dict[str, np.ndarray], agent_capacity: int) -> dict:
    positives = np.max(data["deltas"], axis=1) > 1e-12
    values = _features(data, agent_capacity)
    layouts = np.where(data["seeds"] < 70000, "uniform", "crossing")
    result = {
        "states": int(len(positives)),
        "positive_states": int(np.sum(positives)),
        "positive_episodes": int(len(np.unique(data["episode_ids"][positives]))),
        "by_layout": {},
        "by_step": {},
        "geometry": {},
        "distance_bins": {},
    }
    for layout in ("uniform", "crossing"):
        mask = layouts == layout
        result["by_layout"][layout] = {
            "states": int(np.sum(mask)),
            "positives": int(np.sum(positives & mask)),
        }
    for step in range(5):
        mask = data["steps"] == step
        result["by_step"][str(step)] = {
            "states": int(np.sum(mask)),
            "positives": int(np.sum(positives & mask)),
        }
    for name, array in values.items():
        result["geometry"][name] = {
            "positive": _slice_summary(array, positives),
            "negative": _slice_summary(array, ~positives),
        }
    distance = values["distance"]
    for lo, hi in ((0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 2.0)):
        mask = (distance >= lo) & (distance < hi)
        result["distance_bins"][f"{lo:.1f}-{hi:.1f}"] = {
            "states": int(np.sum(mask)),
            "positives": int(np.sum(positives & mask)),
        }
    return result


def _stability(data: dict[str, np.ndarray], base_config) -> dict:
    positive_rows = np.flatnonzero(np.max(data["deltas"], axis=1) > 1e-12)
    by_seed: dict[int, list[int]] = {}
    for row in positive_rows:
        by_seed.setdefault(int(data["seeds"][row]), []).append(int(row))
    scenarios = {
        "scale_035_two_steps": (0.35, 2),
        "scale_070_two_steps": (0.70, 2),
        "scale_050_one_step": (0.50, 1),
    }
    results = {name: [] for name in scenarios}
    reproduction_errors = []
    for seed, rows in by_seed.items():
        layout = "uniform" if seed < 70000 else "crossing"
        task = base_config.model_copy(update={
            "scenario": base_config.scenario.model_copy(update={"layout_kind": layout})
        })
        env = make_training_env(task)
        try:
            env.reset(seed=seed)
            assert env._scenario_state is not None
            frames, metrics = _baseline(env._scenario_state, env.spec, seed)
            baseline_j = _mean_j(metrics, env.config.collision_weight)
            for row in rows:
                step = int(data["steps"][row])
                agent = f"agent_{int(data['agents'][row])}"
                target = int(data["targets"][row])
                frame = frames[step]
                policy = frame["policies"][agent]
                if policy._assigned_target != target:
                    raise AssertionError(f"规则分配无法复现：{seed=} {step=} {agent=}")
                original = np.asarray([
                    _candidate(frame, agent, target, residual, 0.5, 2)[0] - baseline_j
                    for residual in RESIDUALS
                ])
                reproduction_errors.append(float(np.max(np.abs(original - data["deltas"][row]))))
                for name, (scale, steps) in scenarios.items():
                    revised = np.asarray([
                        _candidate(frame, agent, target, residual, scale, steps)[0] - baseline_j
                        for residual in RESIDUALS
                    ])
                    original_best = int(np.argmax(original))
                    results[name].append({
                        "any_helpful": bool(np.any(revised > 1e-12)),
                        "original_best_still_helpful": bool(revised[original_best] > 1e-12),
                        "same_helpful_action_count": int(np.sum(
                            (original > 1e-12) & (revised > 1e-12)
                        )),
                        "revised_best_delta_j": float(np.max(revised)),
                    })
        finally:
            env.close()
    return {
        "positive_states_checked": int(len(positive_rows)),
        "episodes_replayed": int(len(by_seed)),
        "max_reproduction_error": max(reproduction_errors, default=None),
        "perturbations": {
            name: {
                "any_helpful_states": sum(x["any_helpful"] for x in rows),
                "original_best_still_helpful": sum(
                    x["original_best_still_helpful"] for x in rows
                ),
                "mean_helpful_action_overlap": float(np.mean([
                    x["same_helpful_action_count"] for x in rows
                ])),
                "mean_revised_best_delta_j": float(np.mean([
                    x["revised_best_delta_j"] for x in rows
                ])),
            }
            for name, rows in results.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="H011 机会状态几何与标签稳定性诊断")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"结果已存在，避免覆盖：{args.out}")
    suite = load_suite(SUITE)
    base = suite.groups[0].cases[0].task_config
    env = make_training_env(base)
    try:
        agent_capacity = env.spec.agent_capacity
    finally:
        env.close()
    train = _read(args.train)
    validation = _read(args.validation)
    result = {
        "experiment": "H011-opportunity-diagnostic",
        "train": _distribution(train, agent_capacity),
        "validation": _distribution(validation, agent_capacity),
        "stability_on_validation_positives": _stability(validation, base),
        "warning": "反事实稳定性使用离线完整状态，不是可部署策略分数。",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
