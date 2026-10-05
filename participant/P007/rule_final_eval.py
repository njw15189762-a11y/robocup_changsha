"""在相同场景种子上复核最终局部分配策略。"""
from __future__ import annotations

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import EpisodeContext, PublicTaskParams

from candidate_policy import CoverageAssignmentPolicy
from entry import DeterministicAssignmentPolicy


POLICY_CLASS = CoverageAssignmentPolicy


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
            for mode in ('control', 'submission'):
                obs, _ = env.reset(seed=seed)
                actors = policies(env.config, seed, mode == 'submission')
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
                             if mode == 'submission' else 0}
            rows.append(row)
            if (seed-start+1) % 20 == 0:
                print('evaluated', start, seed-start+1, flush=True)
    finally:
        env.close()
    return rows
