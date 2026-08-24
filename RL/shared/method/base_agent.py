"""所有 RL agent 共享的最小接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """最小 agent 接口。

    这里刻意保持得很薄：
    - `act()` 是必须实现的
    - `reset()/observe()/update()` 都是可选钩子

    这样同一套 trainer 可以同时兼容：
    - simpleNN + SAC
    - EvoFormer + SAC
    - 未来新的 RL 方法
    """

    def reset(self):
        """每个 episode 开始时调用，可用于清空 agent 的内部状态。"""

    @abstractmethod
    def act(self, observation):
        """根据 observation 产生动作。"""
        pass

    def observe(self, observation, action, reward, next_observation, terminated, truncated, info):
        """可选的 transition 钩子，当前 SAC 路径默认不依赖它。"""

    def update(self):
        """可选的参数更新入口。"""
        return {}
