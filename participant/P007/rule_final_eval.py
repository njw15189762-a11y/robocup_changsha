"""H030：在新种子上冻结核验 H022＋H021 对 H016 的纯规则收益。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import EpisodeContext, PublicTaskParams
from coverage_bench.suites import load_suite

from candidate_policy import CandidatePolicy, CoverageAssignmentPolicy, RobustLocalMPCPolicy
from entry import DeterministicAssignmentPolicy


ROOT = Path(__file__).resolve().parents[2]
POLICY_CLASS = CandidatePolicy


def policies(config, seed, candidate):
    params = PublicTaskParams(**config.public.model_dump())
    result = {}
    for i in range(config.num_agents):
        policy = POLICY_CLASS() if candidate else DeterministicAssignmentPolicy(12.)
        policy.reset(EpisodeContext(i, config.num_agents, config.num_targets,
                                    config.horizon, params, seed + i))
        result[f'agent_{i}'] = policy
    return result


def evaluate(config, start, count):
    env = make_training_env(config)
    rows = []
    try:
        for seed in range(start, start+count):
            row = {'seed': seed}
            for mode in ('h016', 'candidate'):
                obs, _ = env.reset(seed=seed)
                actors = policies(env.config, seed, mode == 'candidate')
                coverage = collisions = 0
                for _ in range(env.config.horizon):
                    actions = {name: policy.act(obs[name]) for name, policy in actors.items()}
                    obs, _, _, _, infos = env.step(actions)
                    metric = infos['agent_0']['metrics']
                    coverage += int(metric.matched_targets)
                    collisions += int(metric.collision_agents)
                row[mode] = {'j': (coverage / env.config.num_targets -
                                   env.config.collision_weight*collisions/env.config.num_agents) /
                                  env.config.horizon,
                             'target_steps': coverage, 'collision_steps': collisions,
                             'interventions': sum(int(p.interventions) for p in actors.values())
                             if mode == 'candidate' else 0}
            rows.append(row)
            if (seed-start+1) % 20 == 0:
                print('evaluated', start, seed-start+1, flush=True)
    finally:
        env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--starts', nargs=2, type=int, default=(2910001, 2920001))
    parser.add_argument('--per-layout', type=int, default=120)
    parser.add_argument('--policy', choices=('combined', 'assignment', 'mpc'), default='combined')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    global POLICY_CLASS
    POLICY_CLASS = {'combined': CandidatePolicy, 'assignment': CoverageAssignmentPolicy,
                    'mpc': RobustLocalMPCPolicy}[args.policy]
    began = time.monotonic()
    task = load_suite(ROOT/'configs/public-suite-v1.yaml').groups[0].cases[0].task_config
    result = {'experiment': 'H030-rule-ablation', 'policy': args.policy,
              'starts': args.starts, 'per_layout': args.per_layout,
              'groups': {}}
    for layout, start in zip(('uniform', 'crossing'), args.starts):
        config = task.model_copy(update={
            'scenario': task.scenario.model_copy(update={'layout_kind': layout})})
        result['groups'][layout] = {'rows': evaluate(config, start, args.per_layout)}
    for mode in ('h016', 'candidate'):
        result[mode+'_score'] = 500*sum(np.mean([r[mode]['j'] for r in result['groups'][layout]['rows']])
                                       for layout in ('uniform', 'crossing'))
    result['elapsed_seconds'] = time.monotonic()-began
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('complete', {key: result[key] for key in ('h016_score','candidate_score')}, flush=True)


if __name__ == '__main__':
    main()
