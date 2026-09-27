"""H012：在新种子上审计规则与可达估计之间的剩余空间。"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402
from full_state_planning_diagnostic import (  # noqa: E402
    DAMPINGS,
    GAINS,
    LEADS,
    _oracle_trajectory,
    _reachability_bound,
    _rule_case,
)


SUITE = REPO_ROOT / "configs/public-suite-v1.yaml"


def _load_interventions(path: Path) -> dict[int, float]:
    """把 H010 中每个回合的最佳事后两步干预整理为诊断见证。"""
    with np.load(path) as data:
        seeds = data["seeds"]
        best_by_state = np.max(data["deltas"], axis=1)
    best_by_seed: dict[int, float] = {}
    for seed, improvement in zip(seeds, best_by_state):
        seed = int(seed)
        best_by_seed[seed] = max(best_by_seed.get(seed, 0.0), float(improvement))
    return best_by_seed


def _planner_witness(initial_state) -> dict:
    """只在可达估计显示缺口的回合枚举旧 D001 有限控制器族。"""
    best = None
    best_params = None
    for assignment in itertools.permutations(
        range(initial_state.config.num_targets), initial_state.config.num_agents
    ):
        for gain, lead, damping in itertools.product(GAINS, LEADS, DAMPINGS):
            outcome = _oracle_trajectory(
                initial_state, assignment, gain, lead, damping
            )
            if best is None or outcome["mean_j"] > best["mean_j"] + 1e-12:
                best = outcome
                best_params = {
                    "assignment": assignment,
                    "gain": gain,
                    "lead_seconds": lead,
                    "damping": damping,
                }
    assert best is not None
    return {**best, "parameters": best_params}


def _summarize(rows: list[dict]) -> dict:
    def count(predicate) -> int:
        return sum(bool(predicate(row)) for row in rows)

    rule_j = np.mean([row["rule"]["mean_j"] for row in rows])
    free_j = np.mean([
        row["reachability"]["free_motion_mean_j_estimate"] for row in rows
    ])
    strict_j = np.mean([
        row["reachability"]["mean_j_upper"] for row in rows
    ])
    witness_j = np.mean([
        row["rule"]["mean_j"] + row["intervention_best_delta_j"] for row in rows
    ])
    planner_witness_j = np.mean([
        max(
            row["rule"]["mean_j"],
            row["planner"]["mean_j"] if row["planner"] else row["rule"]["mean_j"],
        )
        for row in rows
    ])
    return {
        "episodes": len(rows),
        "rule_mean_j": float(rule_j),
        "free_motion_mean_j_estimate": float(free_j),
        "speed_only_mean_j_upper": float(strict_j),
        "two_step_hindsight_witness_mean_j": float(witness_j),
        "planner_hindsight_witness_mean_j": float(planner_witness_j),
        "rule_zero_coverage": count(lambda row: row["rule"]["coverage_target_steps"] == 0),
        "zero_and_free_zero": count(lambda row:
            row["rule"]["coverage_target_steps"] == 0
            and row["reachability"]["free_motion_target_steps_estimate"] == 0
        ),
        "zero_but_free_positive": count(lambda row:
            row["rule"]["coverage_target_steps"] == 0
            and row["reachability"]["free_motion_target_steps_estimate"] > 0
        ),
        "free_exceeds_rule": count(lambda row:
            row["reachability"]["free_motion_target_steps_estimate"]
            > row["rule"]["coverage_target_steps"]
        ),
        "free_equals_rule": count(lambda row:
            row["reachability"]["free_motion_target_steps_estimate"]
            == row["rule"]["coverage_target_steps"]
        ),
        "free_below_rule": count(lambda row:
            row["reachability"]["free_motion_target_steps_estimate"]
            < row["rule"]["coverage_target_steps"]
        ),
        "two_step_witness_improves": count(
            lambda row: row["intervention_best_delta_j"] > 1e-12
        ),
        "planner_evaluated": count(lambda row: row["planner"] is not None),
        "planner_beats_rule": count(lambda row:
            row["planner"] is not None
            and row["planner"]["mean_j"] > row["rule"]["mean_j"] + 1e-12
        ),
        "planner_only_improves": count(lambda row:
            row["planner"] is not None
            and row["planner"]["mean_j"] > row["rule"]["mean_j"] + 1e-12
            and row["intervention_best_delta_j"] <= 1e-12
        ),
        "planner_rescues_zero": count(lambda row:
            row["planner"] is not None
            and row["rule"]["coverage_target_steps"] == 0
            and row["planner"]["coverage_target_steps"] > 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="H012 全新验证种子剩余收益空间审计")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--uniform-start", type=int, required=True)
    parser.add_argument("--crossing-start", type=int, required=True)
    parser.add_argument("--per-layout", type=int, required=True)
    parser.add_argument("--planner-gap-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"结果已存在，避免覆盖：{args.out}")
    if args.per_layout < 1:
        raise ValueError("每种布局至少需要一个回合")
    suite = load_suite(SUITE)
    base = suite.groups[0].cases[0].task_config
    interventions = _load_interventions(args.dataset)
    groups = {}
    for layout, start in (
        ("uniform", args.uniform_start),
        ("crossing", args.crossing_start),
    ):
        task = base.model_copy(update={
            "scenario": base.scenario.model_copy(update={"layout_kind": layout})
        })
        rows = []
        for index, seed in enumerate(range(start, start + args.per_layout), 1):
            env = make_training_env(task)
            try:
                observations, _ = env.reset(seed=seed)
                assert env._scenario_state is not None
                initial_state = copy.deepcopy(env._scenario_state)
                reachability = _reachability_bound(initial_state)
                rule = _rule_case(env, observations, seed, initial_state)
                gap = (
                    reachability["free_motion_target_steps_estimate"]
                    > rule["coverage_target_steps"]
                )
                planner = (
                    _planner_witness(initial_state)
                    if args.planner_gap_only and gap else None
                )
                rows.append({
                    "seed": seed,
                    "layout": layout,
                    "rule": rule,
                    "reachability": reachability,
                    "intervention_best_delta_j": interventions.get(seed, 0.0),
                    "planner": planner,
                })
            finally:
                env.close()
            if index % 10 == 0:
                print(f"{layout}: {index}/{args.per_layout} 回合完成", flush=True)
        groups[layout] = {"summary": _summarize(rows), "cases": rows}
    result = {
        "experiment": "H012-fresh-seed-headroom",
        "dataset": str(args.dataset),
        "seed_ranges": {
            "uniform": [args.uniform_start, args.uniform_start + args.per_layout - 1],
            "crossing": [args.crossing_start, args.crossing_start + args.per_layout - 1],
        },
        "groups": groups,
        "score_comparison": {
            name: 500 * sum(groups[group]["summary"][name] for group in groups)
            for name in (
                "rule_mean_j", "free_motion_mean_j_estimate",
                "speed_only_mean_j_upper", "two_step_hindsight_witness_mean_j",
                "planner_hindsight_witness_mean_j",
            )
        },
        "warning": (
            "无接触估计忽略碰撞和轨迹一致性，既非严格上界也非可实现分数；"
            "速度上界严格但宽松；事后两步干预和全状态规划只作诊断。"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "groups": {key: group["summary"] for key, group in groups.items()},
        "score_comparison": result["score_comparison"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
