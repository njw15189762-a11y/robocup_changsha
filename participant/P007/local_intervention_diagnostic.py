"""H009 诊断：规则轨迹上只短暂改一台机器人，寻找可救追踪时刻。"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.envs.motion import advance_targets  # noqa: E402
from coverage_bench.envs.physics import advance_robots  # noqa: E402
from coverage_bench.envs.scenario import snapshot  # noqa: E402
from coverage_bench.metrics import compute_step_metrics  # noqa: E402
from coverage_bench.observations import observe_agent  # noqa: E402
from coverage_bench.protocol import EpisodeContext  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402
from entry import DeterministicAssignmentPolicy  # noqa: E402
from train_hybrid import public_params  # noqa: E402

PARTICIPANT = Path(__file__).resolve().parent
DEFAULT_SUITE = PARTICIPANT / "dev/dev-suite-v1.yaml"
SELECTION_CONFIG = PARTICIPANT / "hybrid-training-h002.yaml"
RESIDUALS = tuple(
    np.array(pair, dtype=np.float32)
    for pair in itertools.product((-1.0, 0.0, 1.0), repeat=2)
    if pair != (0.0, 0.0)
)


def _observations(world, config, spec) -> dict:
    return {
        f"agent_{i}": observe_agent(world, i, config, spec)
        for i in range(config.num_agents)
    }


def _advance(state, actions, spec) -> tuple[dict, object]:
    advance_robots(state, actions)
    advance_targets(state)
    state.step_index += 1
    world = snapshot(state)
    return _observations(world, state.config, spec), compute_step_metrics(world)


def _mean_j(metrics, collision_weight: float) -> float:
    return float(np.mean([
        row.coverage_rate - collision_weight * row.collision_rate for row in metrics
    ]))


def _baseline(initial_state, spec, seed: int) -> tuple[list, list]:
    """保留规则每步动作前的状态、观测和策略记忆，供反事实分支克隆。"""
    state = copy.deepcopy(initial_state)
    policies = {}
    for index in range(state.config.num_agents):
        policy = DeterministicAssignmentPolicy()
        policy.reset(EpisodeContext(
            agent_index=index,
            num_agents=state.config.num_agents,
            num_targets=state.config.num_targets,
            horizon=state.config.horizon,
            task=public_params(state.config),
            policy_seed=seed + index,
        ))
        policies[f"agent_{index}"] = policy
    observations = _observations(snapshot(state), state.config, spec)
    frames = []
    metrics = []
    for _ in range(state.config.horizon):
        actions = {
            agent: policies[agent].act(observation)
            for agent, observation in observations.items()
        }
        frames.append({
            "state": copy.deepcopy(state),
            "policies": copy.deepcopy(policies),
            "observations": observations,
            "actions": actions,
            "prefix_metrics": list(metrics),
            "spec": spec,
        })
        observations, step_metrics = _advance(state, actions, spec)
        metrics.append(step_metrics)
    return frames, metrics


def _candidate(frame, agent_id: str, target: int, residual: np.ndarray,
               scale: float, intervention_steps: int) -> tuple[float, int, int]:
    """克隆规则状态，有限步叠加同一残差，此后完全交回规则。"""
    state = copy.deepcopy(frame["state"])
    policies = copy.deepcopy(frame["policies"])
    actions = dict(frame["actions"])
    metrics = list(frame["prefix_metrics"])
    step_offset = 0
    while state.step_index < state.config.horizon:
        if step_offset > 0:
            actions = {
                agent: policies[agent].act(observation)
                for agent, observation in observations.items()
            }
        if step_offset < intervention_steps:
            policy = policies[agent_id]
            if (
                policy._last_control_mode == policy._MODE_VISIBLE_TRACK
                and policy._assigned_target == target
            ):
                actions[agent_id] = np.clip(
                    actions[agent_id] + scale * residual, -1.0, 1.0
                ).astype(np.float32)
        observations, step_metrics = _advance(state, actions, frame["spec"])
        metrics.append(step_metrics)
        step_offset += 1
    return (
        _mean_j(metrics, state.config.collision_weight),
        int(sum(row.matched_targets for row in metrics)),
        int(sum(row.collision_agents for row in metrics)),
    )


def run_case(case, *, scale: float, intervention_steps: int) -> dict:
    env = make_training_env(case.task_config)
    try:
        env.reset(seed=case.scenario_seed)
        assert env._scenario_state is not None
        frames, baseline_metrics = _baseline(
            env._scenario_state, env.spec, case.scenario_seed
        )
        baseline_j = _mean_j(baseline_metrics, env.config.collision_weight)
        baseline_matches = int(sum(row.matched_targets for row in baseline_metrics))
        baseline_collisions = int(sum(row.collision_agents for row in baseline_metrics))
        best_j = baseline_j
        best_event = None
        active_states = 0
        helpful_events = 0
        helpful_candidates = 0
        for frame in frames:
            for agent_id, policy in frame["policies"].items():
                if (
                    policy._last_control_mode != policy._MODE_VISIBLE_TRACK
                    or policy._assigned_target is None
                ):
                    continue
                active_states += 1
                target = int(policy._assigned_target)
                observation = frame["observations"][agent_id]
                distance = float(np.linalg.norm(observation["targets"][target, :2]))
                useful_here = False
                for residual in RESIDUALS:
                    candidate_j, matched, collisions = _candidate(
                        frame, agent_id, target, residual, scale, intervention_steps
                    )
                    if candidate_j > baseline_j + 1e-12:
                        helpful_candidates += 1
                        useful_here = True
                    if candidate_j > best_j + 1e-12:
                        best_j = candidate_j
                        best_event = {
                            "step_index": int(frame["state"].step_index),
                            "agent_id": agent_id,
                            "target_index": target,
                            "target_distance": distance,
                            "residual": residual.tolist(),
                            "rule_action": np.asarray(
                                frame["actions"][agent_id], dtype=float
                            ).tolist(),
                            "candidate_target_steps": matched,
                            "candidate_collision_agent_steps": collisions,
                        }
                helpful_events += int(useful_here)
        return {
            "case_id": case.case_id,
            "seed": case.scenario_seed,
            "rule_mean_j": baseline_j,
            "best_mean_j": best_j,
            "delta_j": best_j - baseline_j,
            "rule_target_steps": baseline_matches,
            "rule_collision_agent_steps": baseline_collisions,
            "active_tracking_states": active_states,
            "helpful_states": helpful_events,
            "helpful_candidates": helpful_candidates,
            "best_event": best_event,
        }
    finally:
        env.close()


def _selection_groups(suite):
    config = yaml.safe_load(SELECTION_CONFIG.read_text(encoding="utf-8"))
    base = suite.groups[0].cases[0].task_config
    groups = []
    for layout in ("uniform", "crossing"):
        start, end = config["model_selection"][f"{layout}_seeds"]
        task = base.model_copy(update={
            "scenario": base.scenario.model_copy(update={"layout_kind": layout})
        })
        groups.append(SimpleNamespace(
            group_id=layout,
            cases=[SimpleNamespace(
                case_id=f"selection-{layout}-{seed}",
                scenario_seed=seed,
                task_config=task,
            ) for seed in range(int(start), int(end) + 1)],
        ))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="P007 单机器人短段残差反事实诊断")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--selection-seeds", action="store_true")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--intervention-steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--limit-per-group", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = load_suite(args.suite)
    groups = _selection_groups(suite) if args.selection_seeds else suite.groups
    results = {}
    for group in groups:
        cases = group.cases[:args.limit_per_group] if args.limit_per_group else group.cases
        rows = []
        for case in cases:
            row = run_case(
                case, scale=args.scale, intervention_steps=args.intervention_steps
            )
            rows.append(row)
            print(
                f"{case.case_id}: rule={row['rule_mean_j']:.4f} "
                f"best={row['best_mean_j']:.4f} "
                f"helpful={row['helpful_states']}/{row['active_tracking_states']}",
                flush=True,
            )
        results[group.group_id] = {
            "rule_mean_j": float(np.mean([r["rule_mean_j"] for r in rows])),
            "best_mean_j": float(np.mean([r["best_mean_j"] for r in rows])),
            "improved_episodes": sum(r["delta_j"] > 1e-12 for r in rows),
            "helpful_states": sum(r["helpful_states"] for r in rows),
            "active_tracking_states": sum(r["active_tracking_states"] for r in rows),
            "cases": rows,
        }
    comparison = {
        "rule_score": 500 * sum(g["rule_mean_j"] for g in results.values()),
        "one_intervention_oracle_score": 500 * sum(
            g["best_mean_j"] for g in results.values()
        ),
    } if len(results) == 2 else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "experiment": "H009-short-intervention",
        "warning": "逐回合事后择优只用于诊断，不是可提交策略或数学上界。",
        "scale": args.scale,
        "intervention_steps": args.intervention_steps,
        "comparison": comparison,
        "groups": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"汇总：{comparison}\n详细结果：{args.output}")


if __name__ == "__main__":
    main()
