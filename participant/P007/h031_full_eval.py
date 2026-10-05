"""H031：按公开套件四个用例分层，在全新种子上成对复核 H016。"""

from __future__ import annotations

import argparse
import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import rule_final_eval
from candidate_policy import CoverageAssignmentPolicy
from coverage_bench.suites import load_suite


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-case", type=int, default=1000)
    parser.add_argument("--starts", type=int, nargs=4,
                        default=(3210001, 3220001, 3230001, 3240001))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    suite = load_suite(ROOT / "configs/public-suite-v1.yaml")
    cases = [case for group in suite.groups for case in group.cases]
    if len(cases) != 4:
        raise ValueError("预期公开套件包含四个用例")
    rule_final_eval.POLICY_CLASS = CoverageAssignmentPolicy
    start_time = time.monotonic()
    result = {"experiment": "H031 full fresh paired evaluation",
              "baseline": "P007 acd0d13 H016",
              "candidate": "P007-rule-final H031",
              "per_case": args.per_case, "starts": args.starts,
              "cases": {}}

    for case, start in zip(cases, args.starts):
        # 只改变场景种子；每个回合的 H016/H031 使用相同环境和公开任务配置。
        with redirect_stdout(io.StringIO()):
            rows = rule_final_eval.evaluate(case.task_config, start, args.per_case)
        result["cases"][case.case_id] = {
            "group_id": case.group_id,
            "layout": case.task_config.scenario.layout_kind,
            "seed_start": start,
            "rows": rows,
        }
        print("completed", case.case_id, len(rows), flush=True)

    case_deltas = []
    summary = {}
    for case in cases:
        rows = result["cases"][case.case_id]["rows"]
        h016 = np.array([r["h016"]["j"] for r in rows])
        h031 = np.array([r["candidate"]["j"] for r in rows])
        delta = h031 - h016
        case_deltas.append(delta)
        summary[case.case_id] = {
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
            "h031_interventions": sum(r["candidate"]["interventions"] for r in rows),
            "changed_episodes": [r["seed"] for r in rows
                                 if abs(r["candidate"]["j"] - r["h016"]["j"]) > 1e-9],
        }
    result["summary"] = summary
    result["h016_score"] = float(250 * sum(v["h016_mean_j"] for v in summary.values()))
    result["h031_score"] = float(250 * sum(v["h031_mean_j"] for v in summary.values()))
    rng = np.random.default_rng(34001)
    sampled = np.zeros(10000)
    for delta in case_deltas:
        sampled += 250 * rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    result["delta_ci95"] = [float(x) for x in np.quantile(sampled, [0.025, 0.975])]
    result["elapsed_seconds"] = time.monotonic() - start_time
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("score", result["h016_score"], result["h031_score"],
          "delta_ci95", result["delta_ci95"], flush=True)


if __name__ == "__main__":
    main()
