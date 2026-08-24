import numpy as np
import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem
from NeuroEA_GEA_torch.Problems.problem_ops import tasy, tosz


class BBOB_F3(Problem):
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
        c = (self.upper - self.lower) / (2 * self.D)
        diff = x + c.reshape(1, -1) - self.x_opt
        z = (10 ** (0.5 * torch.arange(self.D, device=self.device, dtype=self.dtype) / (self.D - 1))).reshape(1, -1)
        z = z * tasy(0.2, tosz(diff))
        part1 = 10 * (self.D - torch.sum(torch.cos(2 * torch.pi * z), dim=1))
        part2 = torch.sum(z ** 2, dim=1)
        return (part1 + part2).reshape(-1, 1)
