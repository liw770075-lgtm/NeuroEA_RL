import numpy as np
import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem
from NeuroEA_GEA_torch.Problems.problem_ops import fpen, tosz


class BBOB_F4(Problem):
    def __init__(self, **kwargs):
        self.xopt_val = kwargs.get("xopt", 0)
        super().__init__(**kwargs)

    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30

        if np.isscalar(self.xopt_val):
            self.x_opt = torch.full((self.D,), float(self.xopt_val), dtype=self.dtype, device=self.device)
        else:
            self.x_opt = torch.as_tensor(self.xopt_val, dtype=self.dtype, device=self.device)
            self.D = int(self.x_opt.numel())

        self.lower = torch.full((self.D,), -5.0, dtype=self.dtype, device=self.device)
        self.upper = torch.full((self.D,), 5.0, dtype=self.dtype, device=self.device)

    def cal_obj(self, x):
        n, d = x.shape
        c = (self.upper - self.lower) / (2 * d)
        x_shifted = x + c.reshape(1, -1)
        z = tosz(x_shifted - self.x_opt)
        exponent = 0.5 * torch.arange(d, device=self.device, dtype=self.dtype) / (d - 1)
        s = (10 ** exponent).reshape(1, -1).repeat(n, 1)
        odd_dims = torch.arange(0, d, 2, device=self.device)
        scale_mask = (z[:, odd_dims] > 0).to(self.dtype)
        s[:, odd_dims] = (10 ** scale_mask) * s[:, odd_dims]
        z = s * z
        rastrigin = 10 * (d - torch.sum(torch.cos(2 * torch.pi * z), dim=1)) + torch.sum(z ** 2, dim=1)
        return (rastrigin + 100 * fpen(x_shifted)).reshape(-1, 1)
