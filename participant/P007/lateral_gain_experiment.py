"""H018：在 H016 上单独检验可见追踪的横向相对速度补偿。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assignment_control_ablation import VisibleGain12Policy, _run_case, _summary
from timed_gain_experiment import _cases


class LateralCompensationPolicy(VisibleGain12Policy):
    """只改变 H016 已预留的横向速度补偿系数。"""

    _VISIBLE_LATERAL_GAIN = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="H018 横向速度补偿闭环实验")
    parser.add_argument("--set", choices=("diagnostic", "dev", "selection", "fresh"), required=True)
    parser.add_argument("--per-layout", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"结果已存在，避免覆盖：{args.out}")
    groups = {}
    for name, cases in _cases(args.set, args.per_layout):
        rows = []
        for case in cases:
            rows.append(_run_case(case, LateralCompensationPolicy))
        groups[name] = {"summary": _summary(rows), "cases": rows}
        print(f"{name}: {len(rows)}", flush=True)
    result = {
        "experiment": "H018-visible-lateral-compensation-0.5",
        "set": args.set,
        "baseline": "H016 visible tracking gain 12; lateral compensation 0",
        "candidate": "same policy; lateral compensation 0.5",
        "rule_score": 500 * sum(group["summary"]["rule_mean_j"] for group in groups.values()),
        "candidate_score": 500 * sum(group["summary"]["candidate_mean_j"] for group in groups.values()),
        "groups": groups,
        "warning": "本地克隆闭环；正式入口仍是 H016。",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "rule_score": result["rule_score"],
        "candidate_score": result["candidate_score"],
        "groups": {name: group["summary"] for name, group in groups.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
