"""
Gymnasium-compatible spaces with a lightweight fallback implementation.

这里特意做了两套路径：

1. 如果安装了 `gymnasium`
   - `StepwiseEAEnv` 会继承标准的 `gymnasium.Env`
   - `action_space` / `observation_space` 也是标准 `gymnasium.spaces`
   - 这时更容易和外部 RL 生态对接，例如 wrappers、vector env、日志工具等

2. 如果没有安装 `gymnasium`
   - 当前仓库里的环境、SAC、简单 trainer 仍然可以运行
   - 因为这里提供了一个极简 fallback，只实现了本项目当前要用到的 `Box` / `Dict`
   - 但它不是完整的 Gymnasium 实现，因此和依赖 Gymnasium 标准接口的第三方库兼容性会差很多

结论：
- 如果你只是运行当前仓库里的自带 RL 代码，不安装 `gymnasium` 也可以
- 如果你后面要接入标准 RL 工具链，建议安装 `gymnasium`
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    EnvBase = gym.Env
except ImportError:
    # 这里不是为了完全替代 Gymnasium，只是为了让当前项目在“未安装 gymnasium”
    # 的情况下仍然能跑起来。
    class _EnvBase:
        metadata = {}

        def reset(self, *, seed=None, options=None):
            return None

        def step(self, action):
            raise NotImplementedError

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            low_array = np.asarray(low, dtype=dtype)
            high_array = np.asarray(high, dtype=dtype)
            if shape is None:
                shape = low_array.shape if low_array.shape else high_array.shape
            if not shape:
                raise ValueError("Box space requires a non-empty shape.")

            self.shape = tuple(shape)
            self.dtype = np.dtype(dtype)

            if low_array.shape == ():
                low_array = np.full(self.shape, low_array, dtype=self.dtype)
            if high_array.shape == ():
                high_array = np.full(self.shape, high_array, dtype=self.dtype)

            self.low = low_array.astype(self.dtype, copy=False)
            self.high = high_array.astype(self.dtype, copy=False)

        def sample(self):
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class _Dict:
        def __init__(self, spaces_dict):
            self.spaces = dict(spaces_dict)

        def sample(self):
            return {key: space.sample() for key, space in self.spaces.items()}

    spaces = SimpleNamespace(Box=_Box, Dict=_Dict)
    EnvBase = _EnvBase


__all__ = ["EnvBase", "spaces"]
