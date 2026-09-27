"""H007 保存模型在开发集与公开集上的官方指标配对复核。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coverage_bench.suites import load_suite
from train_hybrid import MaskedPPO, build_stack

PARTICIPANT = Path(__file__).resolve().parent
REPO_ROOT = PARTICIPANT.parents[1]
DEFAULT_SUITES = (
    PARTICIPANT / "dev/dev-suite-v1.yaml",
    REPO_ROOT / "configs/public-suite-v1.yaml",
)


def _load_run(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    stats = np.load(directory / "vecnormalize-stats.npz")
    return {
        "model": MaskedPPO.load(str(directory / "model"), device="cpu"),
        "config": summary["experiment"],
        "mean": np.asarray(stats["obs_mean"], dtype=np.float64),
        "var": np.asarray(stats["obs_var"], dtype=np.float64),
        "eps": float(stats["obs_eps"]),
        "clip": float(stats["obs_clip"]),
    }


def _normalize(raw: np.ndarray, run: dict) -> np.ndarray:
    normalized = np.clip(
        (raw - run["mean"]) / np.sqrt(run["var"] + run["eps"]),
        -run["clip"],
        run["clip"],
    ).astype(np.float32)
    normalized[:, -1] = raw[:, -1]
    return normalized


def _evaluate_case(case, run: dict, tracking_gate: str | None = None) -> dict:
    config = run["config"]
    env = build_stack(
        num_vec_envs=1,
        seed=case.scenario_seed,
        layouts=(case.task_config.scenario.layout_kind,),
        residual_scale=float(config["residual_scale"]),
        discovery_bonus=float(config["discovery_bonus"]),
        discovery_schedule=str(config.get("discovery_schedule", "constant")),
        tracking_residual_scale=float(config["tracking_residual_scale"]),
        tracking_progress_alpha=float(config["tracking_progress_alpha"]),
        tracking_residual_gate=tracking_gate or str(
            config.get("tracking_residual_gate", "all_visible")
        ),
        tracking_residual_penalty=float(config.get("tracking_residual_penalty", 0.0)),
        task_config=case.task_config,
    )
    try:
        raw = env.reset()
        coverage, collision, progress = [], [], []
        for _ in range(case.task_config.horizon):
            actions, _ = run["model"].predict(_normalize(raw, run), deterministic=True)
            raw, _, _, infos = env.step(actions)
            metrics = infos[0]["hybrid_metrics"]
            coverage.append(float(metrics.coverage_rate))
            collision.append(float(metrics.collision_rate))
            progress.append(sum(
                float(info.get("tracking_progress_bonus", 0.0)) for info in infos
            ))
        return {
            "mean_j": float(np.mean(coverage) - 0.2 * np.mean(collision)),
            "mean_coverage": float(np.mean(coverage)),
            "mean_collision": float(np.mean(collision)),
            "tracking_progress_bonus_sum": float(sum(progress)),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="H007 官方分数配对复核")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suites", nargs="+", type=Path, default=DEFAULT_SUITES)
    parser.add_argument(
        "--tracking-gate",
        choices=("all_visible", "outside_coverage"),
        default=None,
        help="仅用于冻结模型的推理门对照；默认采用训练配置",
    )
    args = parser.parse_args()
    runs = {"control": _load_run(args.control), "progress": _load_run(args.progress)}
    results = {}
    for suite_path in args.suites:
        suite = load_suite(suite_path)
        groups = {}
        for group in suite.groups:
            rows = []
            for case in group.cases:
                control = _evaluate_case(case, runs["control"], args.tracking_gate)
                progress = _evaluate_case(case, runs["progress"], args.tracking_gate)
                rows.append({
                    "case_id": case.case_id,
                    "seed": case.scenario_seed,
                    "control": control,
                    "progress": progress,
                    "delta_j": progress["mean_j"] - control["mean_j"],
                })
            groups[group.group_id] = {
                "control_mean_j": float(np.mean([r["control"]["mean_j"] for r in rows])),
                "progress_mean_j": float(np.mean([r["progress"]["mean_j"] for r in rows])),
                "improved": sum(r["delta_j"] > 1e-12 for r in rows),
                "regressed": sum(r["delta_j"] < -1e-12 for r in rows),
                "cases": rows,
            }
        result = {
            "groups": groups,
            "control_score": 500.0 * sum(g["control_mean_j"] for g in groups.values()),
            "progress_score": 500.0 * sum(g["progress_mean_j"] for g in groups.values()),
        }
        results[suite.suite_id] = result
        print(
            f"{suite.suite_id}: control={result['control_score']:.4f} "
            f"progress={result['progress_score']:.4f}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"详细结果：{args.output}")


if __name__ == "__main__":
    main()
