"""H014：只改可见且赢得竞价目标之间的选择准则。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


DEV_SUITE = Path(__file__).resolve().parent / "dev/dev-suite-v1.yaml"
PUBLIC_SUITE = REPO_ROOT / "configs/public-suite-v1.yaml"


class InterceptSelectionPolicy(DeterministicAssignmentPolicy):
    """只改目标排序；竞价资格、追踪控制和其他状态逻辑仍继承规则。"""

    def reset(self, context) -> None:
        super().reset(context)
        self._damping = float(context.task.damping)
        self._drive_force = float(context.task.drive_force)
        self._robot_mass = float(context.task.robot_mass)
        self._robot_max_speed = float(context.task.robot_max_speed)
        self._robot_bound = float(
            context.task.map_half_extent - context.task.robot_radius
        )
        self.selection_changes = 0
        self.multiple_candidates = 0

    def _estimate_contact(self, observation, target_index: int) -> tuple[int, int]:
        """用本机可观测位置/速度和公开动力学估计首次覆盖步与覆盖步数。"""
        robot = np.asarray(observation["self_state"][:2], dtype=np.float64).copy()
        velocity = np.asarray(observation["self_state"][2:4], dtype=np.float64).copy()
        target = robot + np.asarray(
            observation["targets"][target_index, :2], dtype=np.float64
        )
        target_velocity = self._target_velocities[target_index].copy()
        remaining = self._num_steps - int(observation["step_index"])
        first = remaining + 1
        covered_steps = 0
        for index in range(1, remaining + 1):
            predicted = self._reflect_prediction(
                target + self._VISIBLE_LEAD_SECONDS * target_velocity
            )
            pursuit = predicted - robot
            distance = float(np.linalg.norm(pursuit))
            damping = (
                self._TRACKING_DAMPING
                if distance < self._DAMPING_DISTANCE else 0.0
            )
            if self._target_radius <= distance < self._DAMPING_DISTANCE:
                damping = self._VISIBLE_APPROACH_DAMPING
            action = np.clip(
                np.clip(5.0 * pursuit, -1.0, 1.0) - damping * velocity,
                -1.0, 1.0,
            )
            robot = np.clip(robot + self._dt * velocity,
                            -self._robot_bound, self._robot_bound)
            velocity = (
                (1.0 - self._damping) * velocity
                + self._dt * self._drive_force / self._robot_mass * action
            )
            speed = float(np.linalg.norm(velocity))
            if speed > self._robot_max_speed:
                velocity *= self._robot_max_speed / speed
            target = self._reflect_prediction(target + self._dt * target_velocity)
            if np.linalg.norm(robot - target) <= self._target_radius:
                first = min(first, index)
                covered_steps += 1
        return first, covered_steps

    def _select_target(self, observation):
        visible = observation["target_visible"]
        targets = observation["targets"]
        candidates = []
        for target_index in range(min(self._num_targets, len(visible))):
            if not bool(visible[target_index]):
                continue
            rel = np.asarray(targets[target_index, :2], dtype=np.float64)
            if self._wins_target(rel, observation):
                candidates.append((
                    target_index,
                    float(np.linalg.norm(rel)),
                    self._visible_pursuit_rel(target_index, rel, observation),
                ))
        if not candidates:
            return super()._select_target(observation)
        nearest = min(candidates, key=lambda item: (
            item[1], (item[0] - self._agent_index) % max(1, self._num_targets)
        ))
        if len(candidates) > 1:
            self.multiple_candidates += 1
        ranked = []
        for candidate in candidates:
            first, covered = self._estimate_contact(observation, candidate[0])
            ranked.append((first, -covered, candidate[1],
                           (candidate[0] - self._agent_index) % max(1, self._num_targets),
                           candidate))
        selected = min(ranked, key=lambda item: item[:4])[-1]
        self.selection_changes += int(selected[0] != nearest[0])
        self._assigned_target = selected[0]
        return selected[2], self._MODE_VISIBLE_TRACK


def _run_case(case) -> dict:
    env = make_training_env(case.task_config)
    try:
        env.reset(seed=case.scenario_seed)
        assert env._scenario_state is not None
        initial_state = copy.deepcopy(env._scenario_state)
        _, rule_metrics = _baseline(initial_state, env.spec, case.scenario_seed)
        rule_j = _mean_j(rule_metrics, env.config.collision_weight)
        state = copy.deepcopy(initial_state)
        policies = {}
        for agent_index in range(env.config.num_agents):
            policy = InterceptSelectionPolicy()
            policy.reset(EpisodeContext(
                agent_index=agent_index,
                num_agents=env.config.num_agents,
                num_targets=env.config.num_targets,
                horizon=env.config.horizon,
                task=public_params(env.config),
                policy_seed=case.scenario_seed + agent_index,
            ))
            policy._num_steps = env.config.horizon
            policies[f"agent_{agent_index}"] = policy
        observations = _observations(snapshot(state), state.config, env.spec)
        metrics = []
        for _ in range(env.config.horizon):
            actions = {
                agent: policies[agent].act(observation)
                for agent, observation in observations.items()
            }
            observations, step_metric = _advance(state, actions, env.spec)
            metrics.append(step_metric)
        candidate_j = _mean_j(metrics, env.config.collision_weight)
        return {
            "seed": case.scenario_seed,
            "case_id": case.case_id,
            "rule_mean_j": rule_j,
            "candidate_mean_j": candidate_j,
            "delta_j": candidate_j - rule_j,
            "rule_target_steps": int(sum(m.matched_targets for m in rule_metrics)),
            "candidate_target_steps": int(sum(m.matched_targets for m in metrics)),
            "rule_collision_agent_steps": int(sum(m.collision_agents for m in rule_metrics)),
            "candidate_collision_agent_steps": int(sum(m.collision_agents for m in metrics)),
            "selection_changes": int(sum(p.selection_changes for p in policies.values())),
            "multiple_candidates": int(sum(p.multiple_candidates for p in policies.values())),
        }
    finally:
        env.close()


def _summary(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "rule_mean_j": float(np.mean([row["rule_mean_j"] for row in rows])),
        "candidate_mean_j": float(np.mean([row["candidate_mean_j"] for row in rows])),
        "improved_episodes": sum(row["delta_j"] > 1e-12 for row in rows),
        "regressed_episodes": sum(row["delta_j"] < -1e-12 for row in rows),
        "changed_assignments": sum(row["selection_changes"] for row in rows),
        "multiple_candidate_decisions": sum(row["multiple_candidates"] for row in rows),
        "candidate_collision_agent_steps": sum(
            row["candidate_collision_agent_steps"] for row in rows
        ),
    }


def _groups(kind: str, per_layout: int):
    if kind in ("dev", "public"):
        suite = load_suite(DEV_SUITE if kind == "dev" else PUBLIC_SUITE)
        return [(group.group_id, group.cases) for group in suite.groups]
    suite = load_suite(PUBLIC_SUITE)
    base = suite.groups[0].cases[0].task_config
    starts = {
        "selection": (41001, 42001),
        "fresh": (63001, 73001),
    }[kind]
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
    parser = argparse.ArgumentParser(description="H014 可见目标接触时间排序对照")
    parser.add_argument("--set", choices=("dev", "selection", "fresh", "public"),
                        required=True)
    parser.add_argument("--per-layout", type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"结果已存在，避免覆盖：{args.out}")
    groups = {}
    for name, cases in _groups(args.set, args.per_layout):
        rows = []
        for case in cases:
            rows.append(_run_case(case))
            if len(rows) % 20 == 0:
                print(f"{name}: {len(rows)}/{len(cases)}", flush=True)
        groups[name] = {"summary": _summary(rows), "cases": rows}
    score = 500 * sum(
        item["summary"]["candidate_mean_j"] for item in groups.values()
    )
    baseline = 500 * sum(
        item["summary"]["rule_mean_j"] for item in groups.values()
    )
    result = {
        "experiment": "H014-intercept-selection",
        "set": args.set,
        "rule_score": baseline,
        "candidate_score": score,
        "groups": groups,
        "warning": "仅本地成对闭环回放；正式入口未改。",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "rule_score": baseline,
        "candidate_score": score,
        "groups": {name: group["summary"] for name, group in groups.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
