"""仅供本地诊断：用全状态搜索十步轨迹，并估算宽松的运动学上界。"""

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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.envs.motion import advance_targets  # noqa: E402
from coverage_bench.envs.physics import advance_robots  # noqa: E402
from coverage_bench.envs.scenario import snapshot  # noqa: E402
from coverage_bench.metrics import compute_step_metrics  # noqa: E402
from coverage_bench.protocol import EpisodeContext  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402
from entry import DeterministicAssignmentPolicy  # noqa: E402
from train_hybrid import public_params  # noqa: E402

DEFAULT_SUITE = Path(__file__).resolve().parent / "dev/dev-suite-v1.yaml"
SELECTION_CONFIG = Path(__file__).resolve().parent / "hybrid-training-h002.yaml"
GAINS = (3.0, 6.0, 12.0)
LEADS = (0.0, 0.35, 0.7)
DAMPINGS = (0.0, 0.8, 1.6)


def _score(metrics: list, collision_weight: float) -> float:
    """使用官方单步覆盖与碰撞定义计算回合均值。"""
    return float(np.mean([
        m.coverage_rate - collision_weight * m.collision_rate for m in metrics
    ]))


def _rule_case(env, observations: dict, seed: int, initial_state) -> dict:
    """在同一个 reset 场景中运行当前正式规则策略。"""
    policies = {}
    for agent_index, agent_id in enumerate(env.agents):
        policy = DeterministicAssignmentPolicy()
        policy.reset(EpisodeContext(
            agent_index=agent_index,
            num_agents=env.config.num_agents,
            num_targets=env.config.num_targets,
            horizon=env.config.horizon,
            task=public_params(env.config),
            policy_seed=seed + agent_index,
        ))
        policies[agent_id] = policy
    metrics = []
    replica = copy.deepcopy(initial_state)
    minimum_distances = np.full(env.config.num_targets, np.inf)
    for _ in range(env.config.horizon):
        actions = {agent: policies[agent].act(obs) for agent, obs in observations.items()}
        observations, _, _, _, infos = env.step(actions)
        advance_robots(replica, actions)
        advance_targets(replica)
        replica.step_index += 1
        metrics.append(next(iter(infos.values()))["metrics"])
        world = env._current_snapshot
        assert world is not None
        # 验证诊断中的克隆模拟与正式环境在每一步完全相同。
        if not (
            np.allclose(replica.robot_positions, world.robot_positions, atol=1e-12)
            and np.allclose(replica.robot_velocities, world.robot_velocities, atol=1e-12)
            and np.allclose(replica.target_positions, world.target_positions, atol=1e-12)
        ):
            raise RuntimeError("离线仿真与正式环境状态不一致")
        distances = np.linalg.norm(
            world.robot_positions[:, None, :] - world.target_positions[None, :, :],
            axis=2,
        )
        minimum_distances = np.minimum(minimum_distances, distances.min(axis=0))
    return {
        "mean_j": _score(metrics, env.config.collision_weight),
        "coverage_target_steps": int(sum(m.matched_targets for m in metrics)),
        "collision_agent_steps": int(sum(m.collision_agents for m in metrics)),
        "minimum_target_distances": minimum_distances.tolist(),
    }


def _oracle_trajectory(initial_state, assignment: tuple[int, ...], gain: float,
                       lead: float, damping: float) -> dict:
    """以真实当前状态控制，并用完整十步轨迹选择参数；不用于正式推理。"""
    state = copy.deepcopy(initial_state)
    config = state.config
    target_bound = config.public.map_half_extent - config.public.target_radius
    metrics = []
    minimum_distances = np.full(config.num_targets, np.inf)
    for _ in range(config.horizon):
        actions = {}
        for agent, target in enumerate(assignment):
            predicted_target = np.clip(
                state.target_positions[target] + lead * state.target_velocities[target],
                -target_bound,
                target_bound,
            )
            drive = (
                gain * (predicted_target - state.robot_positions[agent])
                - damping * state.robot_velocities[agent]
            )
            actions[f"agent_{agent}"] = np.clip(drive, -1.0, 1.0).astype(np.float32)
        advance_robots(state, actions)
        advance_targets(state)
        state.step_index += 1
        world = snapshot(state)
        metrics.append(compute_step_metrics(world))
        distances = np.linalg.norm(
            world.robot_positions[:, None, :] - world.target_positions[None, :, :],
            axis=2,
        )
        minimum_distances = np.minimum(minimum_distances, distances.min(axis=0))
    return {
        "mean_j": _score(metrics, config.collision_weight),
        "coverage_target_steps": int(sum(m.matched_targets for m in metrics)),
        "collision_agent_steps": int(sum(m.collision_agents for m in metrics)),
        "minimum_target_distances": minimum_distances.tolist(),
    }


def _reachability_bound(initial_state) -> dict:
    """忽略控制、接触和轨迹一致性，只限制每步最大可移动距离。"""
    state = copy.deepcopy(initial_state)
    config = state.config
    start_positions = state.robot_positions.copy()
    initial_speeds = np.linalg.norm(state.robot_velocities, axis=1)
    matches = []
    free_motion_matches = []
    free_speed = float(np.max(initial_speeds))
    free_displacement = 0.0
    for step in range(1, config.horizon + 1):
        advance_targets(state)
        state.step_index += 1
        # 官方先用旧速度移动位置，再更新速度；此半径允许最快合法运动。
        max_travel = config.public.dt * (
            initial_speeds + (step - 1) * config.public.robot_max_speed
        )
        distances = np.linalg.norm(
            start_positions[:, None, :] - state.target_positions[None, :, :],
            axis=2,
        )
        reachable = distances <= max_travel[:, None] + state.target_radii[None, :]
        matching = maximum_bipartite_matching(csr_matrix(reachable), perm_type="column")
        matches.append(int(np.sum(matching >= 0)))
        # 无接触时各轴满力的最大位移；接触力可使实际位移超过此估计，故不是严格上界。
        free_displacement += config.public.dt * free_speed
        free_speed = min(
            config.public.robot_max_speed,
            (1.0 - config.public.damping) * free_speed
            + config.public.drive_force / config.public.robot_mass * config.public.dt,
        )
        reach_xy = np.maximum(
            np.abs(start_positions[:, None, :] - state.target_positions[None, :, :])
            - free_displacement,
            0.0,
        )
        free_reachable = np.linalg.norm(reach_xy, axis=2) <= state.target_radii[None, :]
        free_matching = maximum_bipartite_matching(
            csr_matrix(free_reachable), perm_type="column"
        )
        free_motion_matches.append(int(np.sum(free_matching >= 0)))
    return {
        "coverage_target_steps_upper": int(sum(matches)),
        "mean_j_upper": float(sum(matches) / (config.horizon * config.num_targets)),
        "free_motion_target_steps_estimate": int(sum(free_motion_matches)),
        "free_motion_mean_j_estimate": float(
            sum(free_motion_matches) / (config.horizon * config.num_targets)
        ),
    }


def run_case(case) -> dict:
    env = make_training_env(case.task_config)
    try:
        observations, _ = env.reset(seed=case.scenario_seed)
        assert env._scenario_state is not None
        initial_state = copy.deepcopy(env._scenario_state)
        bound = _reachability_bound(initial_state)
        best = None
        best_params = None
        # 这只是一个有限控制器族的离线轨迹搜索，不是全动作空间的最优解。
        for assignment in itertools.permutations(range(env.config.num_targets), env.config.num_agents):
            for gain, lead, damping in itertools.product(GAINS, LEADS, DAMPINGS):
                result = _oracle_trajectory(initial_state, assignment, gain, lead, damping)
                if best is None or (
                    result["mean_j"], result["coverage_target_steps"]
                ) > (best["mean_j"], best["coverage_target_steps"]):
                    best = result
                    best_params = {
                        "assignment": assignment,
                        "gain": gain,
                        "lead_seconds": lead,
                        "damping": damping,
                    }
        rule = _rule_case(env, observations, case.scenario_seed, initial_state)
        assert best is not None
        return {
            "case_id": case.case_id,
            "seed": case.scenario_seed,
            "rule": rule,
            "full_state_planner": {**best, "parameters": best_params},
            "reachability_bound": bound,
        }
    finally:
        env.close()


def _group_summary(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "rule_mean_j": float(np.mean([r["rule"]["mean_j"] for r in rows])),
        "planner_mean_j": float(np.mean([r["full_state_planner"]["mean_j"] for r in rows])),
        "reachability_mean_j_upper": float(np.mean([
            r["reachability_bound"]["mean_j_upper"] for r in rows
        ])),
        "free_motion_mean_j_estimate": float(np.mean([
            r["reachability_bound"]["free_motion_mean_j_estimate"] for r in rows
        ])),
        "planner_beats_rule": sum(
            r["full_state_planner"]["mean_j"] > r["rule"]["mean_j"] + 1e-12
            for r in rows
        ),
        "planner_worse_than_rule": sum(
            r["full_state_planner"]["mean_j"] < r["rule"]["mean_j"] - 1e-12
            for r in rows
        ),
        "rule_zero_coverage": sum(r["rule"]["coverage_target_steps"] == 0 for r in rows),
        "planner_zero_coverage": sum(
            r["full_state_planner"]["coverage_target_steps"] == 0 for r in rows
        ),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P007 全状态十步轨迹与可达性诊断")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-group", type=int, default=0)
    parser.add_argument(
        "--selection-seeds", action="store_true",
        help="使用 41001～42020 独立选择种子，并由公开任务配置生成场景",
    )
    args = parser.parse_args()
    suite = load_suite(args.suite)
    if args.selection_seeds:
        seed_config = yaml.safe_load(SELECTION_CONFIG.read_text(encoding="utf-8"))
        base_config = suite.groups[0].cases[0].task_config
        selected_groups = []
        for layout in ("uniform", "crossing"):
            start, end = seed_config["model_selection"][f"{layout}_seeds"]
            task_config = base_config.model_copy(update={
                "scenario": base_config.scenario.model_copy(
                    update={"layout_kind": layout}
                )
            })
            cases = [SimpleNamespace(
                case_id=f"selection-{layout}-{seed}",
                scenario_seed=seed,
                task_config=task_config,
            ) for seed in range(int(start), int(end) + 1)]
            selected_groups.append(SimpleNamespace(group_id=layout, cases=cases))
    else:
        selected_groups = suite.groups
    groups = {}
    for group in selected_groups:
        cases = group.cases[:args.limit_per_group] if args.limit_per_group else group.cases
        rows = []
        for case in cases:
            row = run_case(case)
            rows.append(row)
            print(
                f"{case.case_id}: rule={row['rule']['mean_j']:.4f} "
                f"planner={row['full_state_planner']['mean_j']:.4f} "
                f"bound={row['reachability_bound']['mean_j_upper']:.4f}",
                flush=True,
            )
        groups[group.group_id] = _group_summary(rows)
    if len(groups) == 2:
        values = list(groups.values())
        comparison = {
            "rule_score": 500 * sum(g["rule_mean_j"] for g in values),
            "planner_score": 500 * sum(g["planner_mean_j"] for g in values),
            "reachability_score_upper": 500 * sum(
                g["reachability_mean_j_upper"] for g in values
            ),
            "free_motion_score_estimate": 500 * sum(
                g["free_motion_mean_j_estimate"] for g in values
            ),
        }
    else:
        comparison = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "experiment": "P007-full-state-trajectory-diagnostic",
        "warning": "全状态和回合轨迹仅用于离线诊断；规划分数不是最优上界。",
        "controller_grid": {"gains": GAINS, "leads": LEADS, "dampings": DAMPINGS},
        "suite": str(args.suite),
        "selection_seeds": args.selection_seeds,
        "comparison": comparison,
        "groups": groups,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"汇总：{comparison}\n详细结果：{args.output}")


if __name__ == "__main__":
    main()
