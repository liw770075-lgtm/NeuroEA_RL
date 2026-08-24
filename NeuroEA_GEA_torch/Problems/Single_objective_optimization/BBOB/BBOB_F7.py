import numpy as np
import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem
from NeuroEA_GEA_torch.Problems.problem_ops import fpen, make_orthogonal_matrix


class BBOB_F7(Problem):
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
        self.Q = make_orthogonal_matrix(self.D, 1, self.device, self.dtype)
        self.R = make_orthogonal_matrix(self.D, 2, self.device, self.dtype)

    def cal_obj(self, x):
        _, d = x.shape
        c = (self.upper - self.lower) / (2 * d)
        diff = x + c.reshape(1, -1) - self.x_opt
        lam = torch.diag(10 ** (0.5 * torch.arange(d, device=self.device, dtype=self.dtype) / (d - 1)))
        z = diff @ (lam @ self.R)
        zh = torch.floor(0.5 + 10 * z) / 10
        zh = torch.where(torch.abs(z) > 0.5, torch.floor(0.5 + z), zh)
        z_rot = z @ self.Q
        term1 = torch.abs(zh[:, 0]) / 1.0e4
        weights = 10 ** (2 * torch.arange(d, device=self.device, dtype=self.dtype) / (d - 1))
        term2 = torch.sum(weights.reshape(1, -1) * (z_rot ** 2), dim=1)
        return (0.1 * torch.maximum(term1, term2) + fpen(x + c.reshape(1, -1))).reshape(-1, 1)
