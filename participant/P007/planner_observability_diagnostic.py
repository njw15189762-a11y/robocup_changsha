"""H013：检查 H012 五个规划收益场景是否依赖不可见目标。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.envs.scenario import snapshot  # noqa: E402
from coverage_bench.protocol import EpisodeContext  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402
from entry import DeterministicAssignmentPolicy  # noqa: E402
from local_intervention_diagnostic import _advance, _baseline, _mean_j, _observations  # noqa: E402
from train_hybrid import public_params  # noqa: E402


SUITE = REPO_ROOT / "configs/public-suite-v1.yaml"


def _policies(config, seed: int) -> dict[str, DeterministicAssignmentPolicy]:
    policies = {}
    for index in range(config.num_agents):
        policy = DeterministicAssignmentPolicy()
        policy.reset(EpisodeContext(
            agent_index=index,
            num_agents=config.num_agents,
            num_targets=config.num_targets,
            horizon=config.horizon,
            task=public_params(config),
            policy_seed=seed + index,
        ))
        policies[f"agent_{index}"] = policy
    return policies


def _planner_actions(state, assignment, gain: float, lead: float,
                     damping: float) -> dict[str, np.ndarray]:
    bound = state.config.public.map_half_extent - state.config.public.target_radius
    actions = {}
    for index, target in enumerate(assignment):
        predicted = np.clip(
            state.target_positions[target] + lead * state.target_velocities[target],
            -bound, bound,
        )
        drive = (
            gain * (predicted - state.robot_positions[index])
            - damping * state.robot_velocities[index]
        )
        actions[f"agent_{index}"] = np.clip(drive, -1, 1).astype(np.float32)
    return actions


def _simulate(initial_state, spec, seed: int, params: dict,
              rule_frames: list[dict]) -> dict:
    assignment = tuple(int(value) for value in params["assignment"])
    gain = float(params["gain"])
    lead = float(params["lead_seconds"])
    damping = float(params["damping"])
    planner_state = copy.deepcopy(initial_state)
    visible_state = copy.deepcopy(initial_state)
    visible_policies = _policies(initial_state.config, seed)
    planner_observations = _observations(
        snapshot(planner_state), planner_state.config, spec,
    )
    visible_observations = _observations(
        snapshot(visible_state), visible_state.config, spec,
    )
    planner_metrics = []
    visible_metrics = []
    trace = []
    for step in range(initial_state.config.horizon):
        planner_actions = _planner_actions(
            planner_state, assignment, gain, lead, damping
        )
        rule_frame = rule_frames[step]
        visible_actions = {}
        step_trace = {"step": step, "agents": []}
        for index, target in enumerate(assignment):
            agent_id = f"agent_{index}"
            observation = visible_observations[agent_id]
            policy = visible_policies[agent_id]
            rule_action = policy.act(observation)
            visible = bool(observation["target_visible"][target])
            action = rule_action
            if visible:
                scale = float(spec.position_scale)
                velocity_scale = float(spec.velocity_scale)
                self_pos = observation["self_state"][:2] * scale
                target_pos = self_pos + observation["targets"][target, :2] * scale
                bound = (
                    visible_state.config.public.map_half_extent
                    - visible_state.config.public.target_radius
                )
                predicted = np.clip(
                    target_pos + lead * policy._target_velocities[target],
                    -bound, bound,
                )
                self_velocity = observation["self_state"][2:4] * velocity_scale
                action = np.clip(
                    gain * (predicted - self_pos) - damping * self_velocity,
                    -1, 1,
                ).astype(np.float32)
            visible_actions[agent_id] = action
            rule_observation = rule_frame["observations"][agent_id]
            step_trace["agents"].append({
                "agent": index,
                "planner_target": target,
                "rule_assigned_target": rule_frame["policies"][agent_id]._assigned_target,
                "rule_mode": rule_frame["policies"][agent_id]._last_control_mode,
                "planner_target_visible_on_rule_path": bool(
                    rule_observation["target_visible"][target]
                ),
                "planner_target_visible_on_planner_path": bool(
                    planner_observations[agent_id]["target_visible"][target]
                ),
                "planner_target_visible_on_local_path": visible,
                "rule_action_on_rule_path": rule_frame["actions"][agent_id].tolist(),
                "planner_action_on_planner_path": planner_actions[agent_id].tolist(),
                "local_action": action.tolist(),
            })
        trace.append(step_trace)
        planner_observations, planner_step = _advance(
            planner_state, planner_actions, spec
        )
        visible_observations, visible_step = _advance(
            visible_state, visible_actions, spec
        )
        planner_metrics.append(planner_step)
        visible_metrics.append(visible_step)
    return {
        "planner_mean_j_replayed": _mean_j(
            planner_metrics, initial_state.config.collision_weight
        ),
        "local_visible_assignment_mean_j": _mean_j(
            visible_metrics, initial_state.config.collision_weight
        ),
        "planner_target_steps_replayed": int(sum(m.matched_targets for m in planner_metrics)),
        "local_visible_assignment_target_steps": int(
            sum(m.matched_targets for m in visible_metrics)
        ),
        "trace": trace,
    }


def _single_agent_ablations(initial_state, spec, seed: int,
                            params: dict) -> list[dict]:
    """每次只让一台机器人改动作，其余机器人保持规则。"""
    assignment = tuple(int(value) for value in params["assignment"])
    gain = float(params["gain"])
    lead = float(params["lead_seconds"])
    damping = float(params["damping"])
    outcomes = []
    for selected in range(initial_state.config.num_agents):
        for visible_only in (False, True):
            state = copy.deepcopy(initial_state)
            policies = _policies(state.config, seed)
            observations = _observations(snapshot(state), state.config, spec)
            metrics = []
            for _ in range(state.config.horizon):
                rule_actions = {
                    agent: policies[agent].act(obs)
                    for agent, obs in observations.items()
                }
                actions = dict(rule_actions)
                agent_id = f"agent_{selected}"
                target = assignment[selected]
                observation = observations[agent_id]
                if not visible_only:
                    actions[agent_id] = _planner_actions(
                        state, assignment, gain, lead, damping
                    )[agent_id]
                elif bool(observation["target_visible"][target]):
                    scale = float(spec.position_scale)
                    velocity_scale = float(spec.velocity_scale)
                    self_pos = observation["self_state"][:2] * scale
                    target_pos = (
                        self_pos + observation["targets"][target, :2] * scale
                    )
                    bound = (
                        state.config.public.map_half_extent
                        - state.config.public.target_radius
                    )
                    predicted = np.clip(
                        target_pos + lead * policies[agent_id]._target_velocities[target],
                        -bound, bound,
                    )
                    velocity = observation["self_state"][2:4] * velocity_scale
                    actions[agent_id] = np.clip(
                        gain * (predicted - self_pos) - damping * velocity,
                        -1, 1,
                    ).astype(np.float32)
                observations, step_metrics = _advance(state, actions, spec)
                metrics.append(step_metrics)
            outcomes.append({
                "agent": selected,
                "planner_target": target,
                "visible_only": visible_only,
                "mean_j": _mean_j(metrics, state.config.collision_weight),
                "target_steps": int(sum(m.matched_targets for m in metrics)),
            })
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description="H013 规划收益的局部可观测性检查")
    parser.add_argument("--h012", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"结果已存在，避免覆盖：{args.out}")
    prior = json.loads(args.h012.read_text(encoding="utf-8"))
    candidates = [
        row for group in prior["groups"].values() for row in group["cases"]
        if row["planner"] is not None
        and row["planner"]["mean_j"] > row["rule"]["mean_j"] + 1e-12
        and row["intervention_best_delta_j"] <= 1e-12
    ]
    suite = load_suite(SUITE)
    base = suite.groups[0].cases[0].task_config
    results = []
    for row in candidates:
        layout = row["layout"]
        seed = int(row["seed"])
        task = base.model_copy(update={
            "scenario": base.scenario.model_copy(update={"layout_kind": layout})
        })
        env = make_training_env(task)
        try:
            env.reset(seed=seed)
            assert env._scenario_state is not None
            frames, rule_metrics = _baseline(env._scenario_state, env.spec, seed)
            replay = _simulate(
                env._scenario_state, env.spec, seed,
                row["planner"]["parameters"], frames,
            )
            rule_j = _mean_j(rule_metrics, env.config.collision_weight)
            if abs(rule_j - row["rule"]["mean_j"]) > 1e-9:
                raise AssertionError(f"规则分数无法复现：{seed}")
            if abs(replay["planner_mean_j_replayed"] - row["planner"]["mean_j"]) > 1e-9:
                raise AssertionError(f"规划分数无法复现：{seed}")
            results.append({
                "seed": seed,
                "layout": layout,
                "rule_mean_j": rule_j,
                "rule_target_steps": row["rule"]["coverage_target_steps"],
                "planner_parameters": row["planner"]["parameters"],
                "single_agent_ablations": _single_agent_ablations(
                    env._scenario_state, env.spec, seed,
                    row["planner"]["parameters"],
                ),
                **replay,
            })
        finally:
            env.close()
    result = {
        "experiment": "H013-planner-only-local-observability",
        "cases": results,
        "warning": (
            "local_visible_assignment 使用事后选出的全局目标匹配及参数，"
            "仅限制动作执行时不读取不可见目标；仍不是合规可部署策略。"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "cases": [{key: value for key, value in row.items() if key != "trace"}
                  for row in results]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
