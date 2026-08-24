import numpy as np
import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem


class BBOB_F5(Problem):
    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30

        state = np.random.RandomState(1)
        self.x_opt = torch.as_tensor(5 * np.sign(state.rand(self.D) - 0.5), device=self.device, dtype=self.dtype)
        self.lower = torch.full((self.D,), -5.0, dtype=self.dtype, device=self.device)
        self.upper = torch.full((self.D,), 5.0, dtype=self.dtype, device=self.device)

    def cal_obj(self, x):
        n, d = x.shape
        z = x
        mask = (self.x_opt.reshape(1, -1) * z) >= 25
        z = torch.where(mask, self.x_opt.reshape(1, -1).expand(n, -1), z)
        exponent = torch.arange(d, device=self.device, dtype=self.dtype) / (d - 1)
        s = torch.sign(self.x_opt) * (10 ** exponent)
        return torch.sum(5 * torch.abs(s).reshape(1, -1) - s.reshape(1, -1) * z, dim=1, keepdim=True)
