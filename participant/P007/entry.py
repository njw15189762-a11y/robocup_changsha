"""包含目标分配、历史追踪和避碰的确定性规则策略。"""

from __future__ import annotations

import math

import numpy as np


class DeterministicAssignmentPolicy:
    """在局部观测内分配目标，并短暂预测刚离开视野的目标。"""

    _EPS = 1e-6
    _MEMORY_STEPS = 4
    _VISIBLE_LEAD_SECONDS = 0.5

    def __init__(self) -> None:
        self._agent_index = 0
        self._num_agents = 1
        self._num_targets = 1
        self._dt = 0.1
        self._target_max_speed = 0.2
        self._target_bound = 0.85
        self._last_target_positions = np.empty((0, 2), dtype=np.float64)
        self._target_velocities = np.empty((0, 2), dtype=np.float64)
        self._last_seen_steps = np.empty(0, dtype=np.int64)
        self._assigned_target: int | None = None

    def reset(self, context) -> None:
        self._agent_index = int(context.agent_index)
        self._num_agents = int(context.num_agents)
        self._num_targets = int(context.num_targets)
        self._dt = float(context.task.dt)
        self._target_max_speed = float(context.task.target_max_speed)
        self._target_bound = float(context.task.map_half_extent - context.task.target_radius)
        self._last_target_positions = np.full((self._num_targets, 2), np.nan, dtype=np.float64)
        self._target_velocities = np.zeros((self._num_targets, 2), dtype=np.float64)
        self._last_seen_steps = np.full(self._num_targets, -1, dtype=np.int64)
        self._assigned_target = None

    def _update_target_tracks(self, observation) -> None:
        """更新目标的绝对位置和经过平滑的速度估计。"""
        step = int(observation["step_index"])
        self_pos = np.asarray(observation["self_state"][:2], dtype=np.float64)
        visible = observation["target_visible"]
        targets = observation["targets"]

        for target_index in range(min(self._num_targets, len(visible))):
            if not bool(visible[target_index]):
                continue
            absolute_position = self_pos + np.asarray(targets[target_index, :2], dtype=np.float64)
            previous_step = int(self._last_seen_steps[target_index])
            if previous_step >= 0 and step > previous_step:
                elapsed = (step - previous_step) * self._dt
                measured_velocity = (
                    absolute_position - self._last_target_positions[target_index]
                ) / elapsed
                speed = float(np.linalg.norm(measured_velocity))
                if speed > self._target_max_speed and speed > self._EPS:
                    measured_velocity *= self._target_max_speed / speed
                # 过滤单步位置差分中的数值波动，同时保证在十步短回合内及时响应。
                self._target_velocities[target_index] = (
                    0.4 * self._target_velocities[target_index]
                    + 0.6 * measured_velocity
                )
            self._last_target_positions[target_index] = absolute_position
            self._last_seen_steps[target_index] = step

    def _reflect_prediction(self, position: np.ndarray) -> np.ndarray:
        """按照目标的反射边界，将短期预测位置折回合法场地。"""
        predicted = np.array(position, dtype=np.float64, copy=True)
        bound = self._target_bound
        for axis in (0, 1):
            while predicted[axis] > bound or predicted[axis] < -bound:
                if predicted[axis] > bound:
                    predicted[axis] = 2.0 * bound - predicted[axis]
                elif predicted[axis] < -bound:
                    predicted[axis] = -2.0 * bound - predicted[axis]
        return predicted

    def _predicted_target_rel(self, target_index: int, observation) -> np.ndarray | None:
        """返回目标的短期相对位置预测；记忆过期时返回 None。"""
        step = int(observation["step_index"])
        last_step = int(self._last_seen_steps[target_index])
        age = step - last_step
        if last_step < 0 or age < 1 or age > self._MEMORY_STEPS:
            return None
        predicted = (
            self._last_target_positions[target_index]
            + self._target_velocities[target_index] * (age * self._dt)
        )
        predicted = self._reflect_prediction(predicted)
        self_pos = np.asarray(observation["self_state"][:2], dtype=np.float64)
        return predicted - self_pos

    def _wins_target(self, target_rel: np.ndarray, observation) -> bool:
        """判断本机器人是否赢得某个可见目标的局部距离竞价。"""
        self_distance = float(np.linalg.norm(target_rel))
        peers = observation["peers"]
        visible = observation["peer_visible"]

        for peer_index in range(min(self._num_agents, len(visible))):
            if not bool(visible[peer_index]):
                continue
            # 队友到目标的向量 = 本机到目标的向量 - 本机到队友的向量。
            peer_distance = float(np.linalg.norm(target_rel - peers[peer_index, :2]))
            if peer_distance < self_distance - self._EPS:
                return False
            if abs(peer_distance - self_distance) <= self._EPS and peer_index < self._agent_index:
                return False
        return True

    def _visible_pursuit_rel(
        self,
        target_index: int,
        target_rel: np.ndarray,
        observation,
    ) -> np.ndarray:
        """对可见目标加入短提前量，但不改变目标竞价使用的当前距离。"""
        self_pos = np.asarray(observation["self_state"][:2], dtype=np.float64)
        predicted = (
            self_pos
            + target_rel
            + self._target_velocities[target_index] * self._VISIBLE_LEAD_SECONDS
        )
        predicted = self._reflect_prediction(predicted)
        return predicted - self_pos

    def _select_target(self, observation) -> np.ndarray | None:
        visible = observation["target_visible"]
        targets = observation["targets"]
        candidates: list[tuple[int, float, np.ndarray]] = []

        for target_index in range(min(self._num_targets, len(visible))):
            if not bool(visible[target_index]):
                continue
            rel = np.asarray(targets[target_index, :2], dtype=np.float64)
            if self._wins_target(rel, observation):
                pursuit_rel = self._visible_pursuit_rel(target_index, rel, observation)
                candidates.append((target_index, float(np.linalg.norm(rel)), pursuit_rel))

        if candidates:
            candidates.sort(
                key=lambda item: (
                    item[1],
                    (item[0] - self._agent_index) % max(1, self._num_targets),
                )
            )
            self._assigned_target = candidates[0][0]
            return candidates[0][2]

        # 目标短暂离开视野时，继续执行上一次的目标分配。
        if self._assigned_target is not None:
            predicted = self._predicted_target_rel(self._assigned_target, observation)
            if predicted is not None:
                return predicted
            self._assigned_target = None

        # 按目标编号为历史目标指定唯一机器人，避免多台机器人追逐同一段记忆。
        remembered: list[tuple[int, float, np.ndarray]] = []
        for target_index in range(self._num_targets):
            if target_index % max(1, self._num_agents) != self._agent_index:
                continue
            predicted = self._predicted_target_rel(target_index, observation)
            if predicted is not None:
                remembered.append((target_index, float(np.linalg.norm(predicted)), predicted))
        if remembered:
            remembered.sort(key=lambda item: (item[1], item[0]))
            self._assigned_target = remembered[0][0]
            return remembered[0][2]
        return None

    def _search_direction(self, observation) -> np.ndarray:
        """没有目标可追踪时，前往按机器人编号分配的内圈搜索航点。"""
        self_pos = np.asarray(observation["self_state"][:2], dtype=np.float64)
        phase = 2.0 * math.pi * self._agent_index / max(1, self._num_agents)
        waypoint = 0.35 * np.array([math.cos(phase), math.sin(phase)], dtype=np.float64)
        return waypoint - self_pos

    def _avoidance(self, observation) -> np.ndarray:
        """对附近的可见队友施加随距离增强的斥力。"""
        repulsion = np.zeros(2, dtype=np.float64)
        peers = observation["peers"]
        visible = observation["peer_visible"]

        for peer_index in range(min(self._num_agents, len(visible))):
            if not bool(visible[peer_index]):
                continue
            rel = np.asarray(peers[peer_index, :2], dtype=np.float64)
            distance = float(np.linalg.norm(rel))
            if self._EPS < distance < 0.25:
                strength = (0.25 - distance) / 0.25
                repulsion -= (rel / distance) * strength
        return repulsion

    def act(self, observation):
        self._update_target_tracks(observation)
        target_rel = self._select_target(observation)
        desired = self._search_direction(observation) if target_rel is None else target_rel

        distance = float(np.linalg.norm(desired))
        direction = desired / distance if distance > self._EPS else np.zeros(2, dtype=np.float64)
        velocity = np.asarray(observation["self_state"][2:4], dtype=np.float64)

        if target_rel is None:
            drive = min(1.0, 2.5 * distance) * direction - 0.35 * velocity
        else:
            damping = 1.2 if distance < 0.18 else 0.0
            # 两个动作分量分别饱和，允许对角方向同时使用两个轴的最大驱动力。
            drive = np.clip(5.0 * target_rel, -1.0, 1.0) - damping * velocity

        drive += 0.9 * self._avoidance(observation)
        return np.clip(drive, -1.0, 1.0).astype(np.float32)

    def close(self) -> None:
        pass


def build_policy(context):
    del context
    return DeterministicAssignmentPolicy()
