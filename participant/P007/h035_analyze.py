"""H035：汇总三批独立种子的成对闭环结果。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FILES = [
    HERE / "dev/h034-fresh4000.json",
    HERE / "dev/h035-confirm4000-a.json",
    HERE / "dev/h035-confirm4000-b.json",
]


def summarize(rows):
    h016 = np.array([r["h016"]["j"] for r in rows])
    h031 = np.array([r["candidate"]["j"] for r in rows])
    delta = h031 - h016
    collision_delta = np.array([
        r["candidate"]["collision_steps"] - r["h016"]["collision_steps"]
        for r in rows
    ])
    target_delta = np.array([
        r["candidate"]["target_steps"] - r["h016"]["target_steps"]
        for r in rows
    ])
    return {
        "n": len(rows),
        "h016_mean_j": float(h016.mean()),
        "h031_mean_j": float(h031.mean()),
        "wins": int(np.sum(delta > 1e-9)),
        "losses": int(np.sum(delta < -1e-9)),
        "ties": int(np.sum(np.abs(delta) <= 1e-9)),
        "h016_target_steps": sum(r["h016"]["target_steps"] for r in rows),
        "h031_target_steps": sum(r["candidate"]["target_steps"] for r in rows),
        "h016_collision_steps": sum(r["h016"]["collision_steps"] for r in rows),
        "h031_collision_steps": sum(r["candidate"]["collision_steps"] for r in rows),
        "h016_zero_coverage": sum(r["h016"]["target_steps"] == 0 for r in rows),
        "h031_zero_coverage": sum(r["candidate"]["target_steps"] == 0 for r in rows),
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
        result["batches"][path.stem]["official_weighted_h016_score"] = batch["h016_score"]
        result["batches"][path.stem]["official_weighted_h031_score"] = batch["h031_score"]
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
    result["pooled"]["official_weighted_h016_score"] = float(
        250 * sum(result["cases"][case_id]["h016_mean_j"] for case_id in case_ids)
    )
    result["pooled"]["official_weighted_h031_score"] = float(
        250 * sum(result["cases"][case_id]["h031_mean_j"] for case_id in case_ids)
    )
    # 四案分别有放回重采样，保留官方四案等权的折算口径。
    rng = np.random.default_rng(35001)
    samples = np.zeros(10000)
    for case_id in case_ids:
        rows = [row for batch in batches for row in batch["cases"][case_id]["rows"]]
        deltas = np.array([r["candidate"]["j"] - r["h016"]["j"] for r in rows])
        for i in range(len(samples)):
            samples[i] += 250 * rng.choice(deltas, size=len(deltas), replace=True).mean()
    result["pooled"]["delta_ci95"] = [float(x) for x in np.quantile(samples, [0.025, 0.975])]
    output = HERE / "dev/h035-summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
