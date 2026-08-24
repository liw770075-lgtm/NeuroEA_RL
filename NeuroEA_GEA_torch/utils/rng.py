import numpy as np
import torch


class NumpyTorchRNG:
    """NumPy-backed RNG that returns tensors on the requested device."""

    def __init__(self, seed=None, device="cpu", dtype=torch.float64):
        self.state = np.random.RandomState(seed)
        self.device = torch.device(device)
        self.dtype = dtype

    def _to_tensor(self, array, dtype=None):
        target_dtype = self.dtype if dtype is None else dtype
        return torch.as_tensor(array, device=self.device, dtype=target_dtype)

    def rand_numpy(self, shape):
        return self.state.rand(*shape)

    def standard_normal_numpy(self, shape):
        return self.state.standard_normal(shape)

    def randint_numpy(self, low, high=None, size=None):
        return self.state.randint(low, high=high, size=size)

    def rand(self, *shape):
        return self._to_tensor(self.rand_numpy(shape))

    def standard_normal(self, shape):
        return self._to_tensor(self.standard_normal_numpy(shape))

    def randint(self, low, high=None, size=None):
        return self._to_tensor(self.randint_numpy(low, high=high, size=size), dtype=torch.long)

    def uniform(self, low, high, size=None):
        low_np = np.asarray(low, dtype=np.float64)
        high_np = np.asarray(high, dtype=np.float64)
        values = self.state.uniform(low=low_np, high=high_np, size=size)
        return self._to_tensor(values)
