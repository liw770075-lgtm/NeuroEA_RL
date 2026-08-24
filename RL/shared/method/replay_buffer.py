"""Replay buffer 实现。

这里保留两条路径：
- `ReplayBuffer`：给一维 observation 的 simpleNN-SAC 使用
- `StructuredReplayBuffer`：给结构化 observation 的 EvoFormer-SAC 使用
"""

from __future__ import annotations

import numpy as np

from RL.shared.method.observation import clone_observation


class ReplayBuffer:
    """简单的 numpy replay buffer，专门给当前 SAC 用。"""

    def __init__(self, observation_dim, action_dim, capacity=100_000):
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)

        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.position = 0
        self.size = 0

    def add(self, observation, action, reward, next_observation, done):
        idx = self.position
        self.observations[idx] = observation
        self.actions[idx] = action
        self.rewards[idx] = float(reward)
        self.next_observations[idx] = next_observation
        self.dones[idx] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def __len__(self):
        return self.size


class StructuredReplayBuffer:
    """
    给结构化 observation 用的 replay buffer。

    和 `ReplayBuffer` 的区别：
    - 不要求 observation 先 flatten 成固定长度向量
    - 直接保存 dict observation
    - sample 时再调用 `collate_fn` 做 batch padding / tensor 化

    这条路径主要给三层 Transformer 编码器使用。
    """

    def __init__(self, action_dim, capacity=100_000, collate_fn=None):
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.collate_fn = collate_fn

        self.observations = [None] * self.capacity
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_observations = [None] * self.capacity
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.position = 0
        self.size = 0

    def add(self, observation, action, reward, next_observation, done):
        idx = self.position
        self.observations[idx] = clone_observation(observation)
        self.actions[idx] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[idx] = float(reward)
        self.next_observations[idx] = clone_observation(next_observation)
        self.dones[idx] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=int(batch_size))
        observations = [self.observations[index] for index in indices]
        next_observations = [self.next_observations[index] for index in indices]

        if self.collate_fn is not None:
            observations = self.collate_fn(observations)
            next_observations = self.collate_fn(next_observations)

        return {
            "observations": observations,
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": next_observations,
            "dones": self.dones[indices],
        }

    def __len__(self):
        return self.size
