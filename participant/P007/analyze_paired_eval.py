"""汇总三批独立种子的成对闭环结果。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FILES = [
    HERE / "dev/paired-batch-1.json",
    HERE / "dev/paired-batch-2.json",
    HERE / "dev/paired-batch-3.json",
]


def summarize(rows):
    control = np.array([r["control"]["j"] for r in rows])
    submission = np.array([r["submission"]["j"] for r in rows])
    delta = submission - control
    collision_delta = np.array([
        r["submission"]["collision_steps"] - r["control"]["collision_steps"]
        for r in rows
    ])
    target_delta = np.array([
        r["submission"]["target_steps"] - r["control"]["target_steps"]
        for r in rows
    ])
    return {
        "n": len(rows),
        "control_mean_j": float(control.mean()),
        "submission_mean_j": float(submission.mean()),
        "wins": int(np.sum(delta > 1e-9)),
        "losses": int(np.sum(delta < -1e-9)),
        "ties": int(np.sum(np.abs(delta) <= 1e-9)),
        "control_target_steps": sum(r["control"]["target_steps"] for r in rows),
        "submission_target_steps": sum(r["submission"]["target_steps"] for r in rows),
        "control_collision_steps": sum(r["control"]["collision_steps"] for r in rows),
        "submission_collision_steps": sum(r["submission"]["collision_steps"] for r in rows),
        "control_zero_coverage": sum(r["control"]["target_steps"] == 0 for r in rows),
        "submission_zero_coverage": sum(r["submission"]["target_steps"] == 0 for r in rows),
        "episodes_extra_collision": int(np.sum(collision_delta > 0)),
        "episodes_reduced_collision": int(np.sum(collision_delta < 0)),
        "pure_collision_losses": int(np.sum((collision_delta > 0) & (target_delta == 0))),
        "worst_episodes": [
            {"seed": rows[i]["seed"], "delta_j": float(delta[i]),
             "target_delta": int(target_delta[i]), "collision_delta": int(collision_delta[i])}
            for i in np.argsort(delta)[:5] if delta[i] < -1e-9
        ],
    }


def main():
    batches = [json.loads(path.read_text(encoding="utf-8")) for path in FILES]
    case_ids = list(batches[0]["cases"])
    result = {"sources": [str(p.relative_to(HERE)) for p in FILES], "batches": {}, "cases": {}}
    for path, batch in zip(FILES, batches):
        rows = [row for case in batch["cases"].values() for row in case["rows"]]
        result["batches"][path.stem] = summarize(rows)
        result["batches"][path.stem]["official_weighted_control_score"] = batch["control_score"]
        result["batches"][path.stem]["official_weighted_submission_score"] = batch["submission_score"]
        result["batches"][path.stem]["delta_ci95"] = batch["delta_ci95"]
    for case_id in case_ids:
        rows = [row for batch in batches for row in batch["cases"][case_id]["rows"]]
        seeds = [r["seed"] for r in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"重复种子: {case_id}")
        result["cases"][case_id] = summarize(rows)
    all_rows = [row for case_id in case_ids for batch in batches
                for row in batch["cases"][case_id]["rows"]]
    result["pooled"] = summarize(all_rows)
    result["pooled"]["official_weighted_control_score"] = float(
        250 * sum(result["cases"][case_id]["control_mean_j"] for case_id in case_ids)
    )
    result["pooled"]["official_weighted_submission_score"] = float(
        250 * sum(result["cases"][case_id]["submission_mean_j"] for case_id in case_ids)
    )
    # 四案分别有放回重采样，保留官方四案等权的折算口径。
    rng = np.random.default_rng(35001)
    samples = np.zeros(10000)
    for case_id in case_ids:
        rows = [row for batch in batches for row in batch["cases"][case_id]["rows"]]
        deltas = np.array([r["submission"]["j"] - r["control"]["j"] for r in rows])
        for i in range(len(samples)):
            samples[i] += 250 * rng.choice(deltas, size=len(deltas), replace=True).mean()
    result["pooled"]["delta_ci95"] = [float(x) for x in np.quantile(samples, [0.025, 0.975])]
    output = HERE / "dev/paired-summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
