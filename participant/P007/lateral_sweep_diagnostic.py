"""H006 横向相对速度补偿的开发集与独立选择集扫描。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coverage_bench.suites import load_suite
from damping_sweep_diagnostic import SUITE_PATH, run_case, summarize
from entry import DeterministicAssignmentPolicy
from oracle_search_diagnostic import CONFIG_PATH, run_episode, summarize as summarize_selection


GAINS = (0.0, 0.5, 1.0, 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="P007 H006 横向相对速度补偿扫描")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = load_suite(SUITE_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    results = []
    for gain in GAINS:
        DeterministicAssignmentPolicy._VISIBLE_LATERAL_GAIN = gain
        development = {
            group.group_id: summarize([run_case(case, 0.0) for case in group.cases])
            for group in suite.groups
        }
        selection = {}
        for layout in ("uniform", "crossing"):
            first, last = config["model_selection"][f"{layout}_seeds"]
            selection[layout] = summarize_selection(
                [
                    run_episode(
                        layout=layout,
                        seed=seed,
                        controller="rule",
                        scale=0.0,
                        scope="strict",
                    )
                    for seed in range(int(first), int(last) + 1)
                ]
            )

        development_score = 500.0 * sum(
            group["mean_j"] for group in development.values()
        )
        selection_score = 500.0 * sum(
            group["mean_j"] for group in selection.values()
        )
        results.append(
            {
                "gain": gain,
                "development_score": development_score,
                "selection_score": selection_score,
                "development_groups": development,
                "selection_groups": selection,
            }
        )
        print(
            f"gain={gain:.1f} dev={development_score:.4f} "
            f"selection={selection_score:.4f} "
            f"dev_groups={[(key, round(value['mean_j'], 6)) for key, value in development.items()]} "
            f"selection_groups={[(key, round(value['mean_j'], 6)) for key, value in selection.items()]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"experiment": "H006-lateral-velocity", "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"详细结果：{args.output}")


if __name__ == "__main__":
    main()
