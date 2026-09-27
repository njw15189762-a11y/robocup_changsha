"""H017：只在回合末三步把可见追踪增益从 12 退回 5。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coverage_bench.suites import load_suite  # noqa: E402
from assignment_control_ablation import (  # noqa: E402
    PUBLIC_SUITE,
    VisibleGain12Policy,
    _groups,
    _run_case,
    _summary,
)


FIRST_LOW_GAIN_STEP = 7


class TimedGainPolicy(VisibleGain12Policy):
    """早期保留强接近，末三步仅改变可见追踪比例增益。"""

    def reset(self, context) -> None:
        super().reset(context)
        self._current_step = 0

    @property
    def _VISIBLE_GAIN(self) -> float:
        return 5.0 if self._current_step >= FIRST_LOW_GAIN_STEP else 12.0

    def act(self, observation):
        self._current_step = int(observation["step_index"])
        return super().act(observation)


def _cases(kind: str, per_layout: int):
    if kind in ("dev", "selection", "public"):
        return _groups(kind, per_layout)
    starts = {
        "diagnostic": (62001, 72001),
        "fresh": (65001, 75001),
    }[kind]
    suite = load_suite(PUBLIC_SUITE)
    base = suite.groups[0].cases[0].task_config
    groups = []
    for layout, start in zip(("uniform", "crossing"), starts):
        task = base.model_copy(update={
            "scenario": base.scenario.model_copy(update={"layout_kind": layout})
        })
        groups.append((layout, [SimpleNamespace(
            case_id=f"{kind}-{layout}-{seed}",
            scenario_seed=seed,
            task_config=task,
        ) for seed in range(start, start + per_layout)]))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="H017 末段增益安全门实验")
    parser.add_argument("--set", choices=(
        "diagnostic", "dev", "selection", "fresh", "public"
    ), required=True)
    parser.add_argument("--per-layout", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"结果已存在，避免覆盖：{args.out}")
    groups = {}
    for name, cases in _cases(args.set, args.per_layout):
        rows = []
        for case in cases:
            rows.append(_run_case(case, TimedGainPolicy))
            if len(rows) % 20 == 0:
                print(f"{name}: {len(rows)}/{len(cases)}", flush=True)
        groups[name] = {"summary": _summary(rows), "cases": rows}
    result = {
        "experiment": "H017-final-three-step-low-gain",
        "set": args.set,
        "first_low_gain_step": FIRST_LOW_GAIN_STEP,
        "baseline": "H016 visible gain 12 on all steps",
        "candidate": "visible gain 12 on steps 0-6; gain 5 on steps 7-9",
        "rule_score": 500 * sum(
            item["summary"]["rule_mean_j"] for item in groups.values()
        ),
        "candidate_score": 500 * sum(
            item["summary"]["candidate_mean_j"] for item in groups.values()
        ),
        "groups": groups,
        "warning": "本地克隆闭环；正式入口仍是 H016。",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "rule_score": result["rule_score"],
        "candidate_score": result["candidate_score"],
        "groups": {key: value["summary"] for key, value in groups.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
