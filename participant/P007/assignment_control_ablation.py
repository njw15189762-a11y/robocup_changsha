"""H015/H016：从同一规则基线分别检验竞价资格与可见追踪增益。"""

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


class EligibilityFallbackPolicy(DeterministicAssignmentPolicy):
    """只有当前无赢得目标时，允许追最近的可见但竞价失败的目标。"""

    def reset(self, context) -> None:
        super().reset(context)
        self.interventions = 0

    def _select_target(self, observation):
        visible = observation["target_visible"]
        targets = observation["targets"]
        visible_targets = []
        has_won_target = False
        for target_index in range(min(self._num_targets, len(visible))):
            if not bool(visible[target_index]):
                continue
            rel = np.asarray(targets[target_index, :2], dtype=np.float64)
            visible_targets.append((target_index, float(np.linalg.norm(rel)), rel))
            has_won_target |= self._wins_target(rel, observation)
        if visible_targets and not has_won_target:
            selected = min(visible_targets, key=lambda item: (
                item[1],
                (item[0] - self._agent_index) % max(1, self._num_targets),
            ))
            self._assigned_target = selected[0]
            self.interventions += 1
            return self._visible_pursuit_rel(
                selected[0], selected[2], observation
            ), self._MODE_VISIBLE_TRACK
        return super()._select_target(observation)


class VisibleGain12Policy(DeterministicAssignmentPolicy):
    """只将可见已分配目标的比例增益 5 改为 12，其余控制保持规则。"""

    _VISIBLE_GAIN = 12.0

    def reset(self, context) -> None:
        super().reset(context)
        self.interventions = 0

    def act(self, observation):
        # 与 entry.py 的 act 对齐，仅在标注处改变可见追踪比例增益。
        self._update_target_tracks(observation)
        target_rel, tracking_mode = self._select_target(observation)
        any_target_visible = bool(np.any(observation["target_visible"]))
        if tracking_mode is not None:
            self._last_control_mode = tracking_mode
        elif any_target_visible:
            self._last_control_mode = self._MODE_RULE_SEARCH
        else:
            self._last_control_mode = self._MODE_LEARNED_SEARCH
        desired = self._search_direction(observation) if target_rel is None else target_rel

        distance = float(np.linalg.norm(desired))
        direction = (
            desired / distance if distance > self._EPS
            else np.zeros(2, dtype=np.float64)
        )
        velocity = np.asarray(observation["self_state"][2:4], dtype=np.float64)

        if target_rel is None:
            drive = min(1.0, 2.5 * distance) * direction - 0.35 * velocity
            if self._last_control_mode == self._MODE_LEARNED_SEARCH:
                residual = self._learned_search_residual(observation)
                if np.any(residual):
                    drive += 0.35 * residual
            baseline_drive = drive.copy()
        else:
            damping_distance = distance
            if (
                tracking_mode == self._MODE_VISIBLE_TRACK
                and self._VISIBLE_DAMPING_DISTANCE_SOURCE == "observed"
                and self._assigned_target is not None
            ):
                damping_distance = float(np.linalg.norm(
                    observation["targets"][self._assigned_target, :2]
                ))
            damping = (
                self._TRACKING_DAMPING
                if damping_distance < self._DAMPING_DISTANCE else 0.0
            )
            if (
                tracking_mode == self._MODE_VISIBLE_TRACK
                and self._target_radius <= damping_distance < self._DAMPING_DISTANCE
            ):
                damping = self._VISIBLE_APPROACH_DAMPING
            baseline_drive = (
                np.clip(5.0 * target_rel, -1.0, 1.0) - damping * velocity
            )
            gain = self._VISIBLE_GAIN if tracking_mode == self._MODE_VISIBLE_TRACK else 5.0
            drive = np.clip(gain * target_rel, -1.0, 1.0) - damping * velocity
            if (
                tracking_mode == self._MODE_VISIBLE_TRACK
                and self._VISIBLE_LATERAL_GAIN > 0.0
                and self._assigned_target is not None
            ):
                current_rel = np.asarray(
                    observation["targets"][self._assigned_target, :2],
                    dtype=np.float64,
                )
                current_distance = float(np.linalg.norm(current_rel))
                if current_distance > self._EPS:
                    unit = current_rel / current_distance
                    relative_velocity = velocity - self._target_velocities[
                        self._assigned_target
                    ]
                    lateral_velocity = relative_velocity - np.dot(
                        relative_velocity, unit
                    ) * unit
                    correction = self._VISIBLE_LATERAL_GAIN * lateral_velocity
                    drive -= correction
                    baseline_drive -= correction

        avoidance = 0.9 * self._avoidance(observation)
        actual = np.clip(drive + avoidance, -1.0, 1.0).astype(np.float32)
        baseline = np.clip(
            baseline_drive + avoidance, -1.0, 1.0
        ).astype(np.float32)
        self.interventions += int(np.any(np.abs(actual - baseline) > 1e-7))
        return actual


POLICIES = {
    "eligibility": EligibilityFallbackPolicy,
    "gain12": VisibleGain12Policy,
}


def _run_case(case, policy_class) -> dict:
    env = make_training_env(case.task_config)
    try:
        env.reset(seed=case.scenario_seed)
        assert env._scenario_state is not None
        initial_state = copy.deepcopy(env._scenario_state)
        frames, rule_metrics = _baseline(initial_state, env.spec, case.scenario_seed)
        rule_j = _mean_j(rule_metrics, env.config.collision_weight)
        state = copy.deepcopy(initial_state)
        policies = {}
        for index in range(env.config.num_agents):
            policy = policy_class()
            policy.reset(EpisodeContext(
                agent_index=index,
                num_agents=env.config.num_agents,
                num_targets=env.config.num_targets,
                horizon=env.config.horizon,
                task=public_params(env.config),
                policy_seed=case.scenario_seed + index,
            ))
            policies[f"agent_{index}"] = policy
        observations = _observations(snapshot(state), state.config, env.spec)
        metrics = []
        first_action_difference = 0.0
        for step in range(env.config.horizon):
            actions = {
                agent: policies[agent].act(observation)
                for agent, observation in observations.items()
            }
            if step == 0:
                first_action_difference = max(
                    float(np.max(np.abs(actions[agent] - rule_action)))
                    for agent, rule_action in frames[0]["actions"].items()
                )
            observations, step_metrics = _advance(state, actions, env.spec)
            metrics.append(step_metrics)
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
            "interventions": int(sum(p.interventions for p in policies.values())),
            "first_action_difference": first_action_difference,
        }
    finally:
        env.close()


def _groups(kind: str, per_layout: int):
    if kind in ("dev", "public"):
        suite = load_suite(DEV_SUITE if kind == "dev" else PUBLIC_SUITE)
        return [(group.group_id, group.cases) for group in suite.groups]
    suite = load_suite(PUBLIC_SUITE)
    base = suite.groups[0].cases[0].task_config
    starts = {
        "selection": (41001, 42001),
        "fresh": (64001, 74001),
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


def _summary(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "rule_mean_j": float(np.mean([row["rule_mean_j"] for row in rows])),
        "candidate_mean_j": float(np.mean([row["candidate_mean_j"] for row in rows])),
        "improved_episodes": sum(row["delta_j"] > 1e-12 for row in rows),
        "regressed_episodes": sum(row["delta_j"] < -1e-12 for row in rows),
        "interventions": sum(row["interventions"] for row in rows),
        "candidate_collision_agent_steps": sum(
            row["candidate_collision_agent_steps"] for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="H015/H016 独立单变量闭环对照")
    parser.add_argument("--variant", choices=tuple(POLICIES), required=True)
    parser.add_argument("--set", choices=("dev", "selection", "fresh", "public"),
                        required=True)
    parser.add_argument("--per-layout", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"结果已存在，避免覆盖：{args.out}")
    groups = {}
    for name, cases in _groups(args.set, args.per_layout):
        rows = []
        for case in cases:
            rows.append(_run_case(case, POLICIES[args.variant]))
            if len(rows) % 20 == 0:
                print(f"{args.variant} {name}: {len(rows)}/{len(cases)}", flush=True)
        groups[name] = {"summary": _summary(rows), "cases": rows}
    result = {
        "experiment": "H015-eligibility" if args.variant == "eligibility" else "H016-visible-gain12",
        "variant": args.variant,
        "set": args.set,
        "per_layout": args.per_layout,
        "rule_score": 500 * sum(
            item["summary"]["rule_mean_j"] for item in groups.values()
        ),
        "candidate_score": 500 * sum(
            item["summary"]["candidate_mean_j"] for item in groups.values()
        ),
        "groups": groups,
        "warning": "本地闭环对照；两个候选互不叠加，正式 entry.py 未改。",
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
