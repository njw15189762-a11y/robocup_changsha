"""H031：基于局部观测的覆盖价值分配。"""

from __future__ import annotations

import numpy as np
from functools import lru_cache

from entry import DeterministicAssignmentPolicy


def _local_assignment(utility):
    """枚举小规模局部匹配；公开任务只有三台机器人，无需额外推理依赖。"""
    row_count, col_count = utility.shape

    @lru_cache(maxsize=None)
    def solve(row, used):
        if row == row_count:
            return 0.0, ()
        best_value, best_cols = -float("inf"), ()
        for col in range(col_count):
            if used & (1 << col):
                continue
            suffix_value, suffix_cols = solve(row + 1, used | (1 << col))
            value = float(utility[row, col]) + suffix_value
            if value > best_value + 1e-12:
                best_value, best_cols = value, (col,) + suffix_cols
        return best_value, best_cols

    return tuple(range(row_count)), solve(0, 0)[1]


class CoverageAssignmentPolicy(DeterministicAssignmentPolicy):
    """用预计剩余覆盖步数代替逐目标最近距离竞价。"""

    def reset(self, context) -> None:
        super().reset(context)
        self._horizon = int(context.horizon)
        self._damping = float(context.task.damping)
        self._acceleration = float(context.task.drive_force / context.task.robot_mass)
        self._max_speed = float(context.task.robot_max_speed)
        self._robot_bound = float(context.task.map_half_extent - context.task.robot_radius)
        self._sense_radius = float(context.task.sense_radius)
        self.interventions = 0

    def _coverage_value(self, robot_pos, robot_vel, target_pos, target_vel, steps):
        """仅估算无遮挡、无碰撞的接触收益；真实回合分数由官方环境计算。"""
        pos = robot_pos.copy()
        velocity = robot_vel.copy()
        target = target_pos.copy()
        total = 0.0
        for _ in range(steps):
            pursuit = self._reflect_prediction(target + 0.5 * target_vel) - pos
            distance = float(np.linalg.norm(pursuit))
            damping = 1.2 if distance < 0.15 else 0.0
            action = np.clip(12.0 * pursuit - damping * velocity, -1.0, 1.0)
            pos = np.clip(pos + self._dt * velocity, -self._robot_bound, self._robot_bound)
            velocity = (1.0 - self._damping) * velocity + self._dt * self._acceleration * action
            speed = float(np.linalg.norm(velocity))
            if speed > self._max_speed:
                velocity *= self._max_speed / speed
            target = self._reflect_prediction(target + self._dt * target_vel)
            gap = float(np.linalg.norm(pos - target))
            total += 1.0 if gap <= self._target_radius else 0.0
            total += 0.05 * max(0.0, 0.35 - gap)
        return total

    def _select_target(self, observation):
        baseline, baseline_mode = super()._select_target(observation)
        visible_targets = [j for j in range(self._num_targets)
                           if observation["target_visible"][j]]
        visible_peers = [i for i in range(self._num_agents)
                         if observation["peer_visible"][i]]
        if len(visible_targets) < 2 or not visible_peers:
            return baseline, baseline_mode
        own_pos = np.asarray(observation["self_state"][:2], dtype=np.float64)
        own_vel = np.asarray(observation["self_state"][2:4], dtype=np.float64)
        robots = [(self._agent_index, own_pos, own_vel)]
        for index in visible_peers:
            row = observation["peers"][index]
            robots.append((index, own_pos + np.asarray(row[:2], dtype=np.float64),
                           own_vel + np.asarray(row[2:4], dtype=np.float64)))
        steps = min(self._horizon - int(observation["step_index"]), 8)
        utility = np.zeros((len(robots), len(visible_targets) + len(robots)))
        for i, (_, robot_pos, robot_vel) in enumerate(robots):
            for k, target_index in enumerate(visible_targets):
                target_pos = own_pos + np.asarray(observation["targets"][target_index, :2],
                                                  dtype=np.float64)
                # 不假定队友能看到超出其感知半径的目标。
                if np.linalg.norm(robot_pos - target_pos) > self._sense_radius:
                    utility[i, k] = -1.0
                    continue
                utility[i, k] = self._coverage_value(
                    robot_pos, robot_vel, target_pos,
                    self._target_velocities[target_index], steps)
        rows, cols = _local_assignment(utility)
        selected = next((visible_targets[col] for row, col in zip(rows, cols)
                         if row == 0 and col < len(visible_targets)), None)
        if selected is None or utility[0, visible_targets.index(selected)] <= 0.0:
            selected = None
        baseline_selected = self._assigned_target if baseline_mode == self._MODE_VISIBLE_TRACK else None
        if selected == baseline_selected:
            return baseline, baseline_mode
        self.interventions += 1
        self._assigned_target = selected
        if selected is None:
            return None, None
        rel = np.asarray(observation["targets"][selected, :2], dtype=np.float64)
        return self._visible_pursuit_rel(selected, rel, observation), self._MODE_VISIBLE_TRACK
