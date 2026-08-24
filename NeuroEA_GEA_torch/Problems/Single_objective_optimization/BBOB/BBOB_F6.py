import numpy as np
import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem
from NeuroEA_GEA_torch.Problems.problem_ops import make_orthogonal_matrix, tosz


class BBOB_F6(Problem):
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
        d = x.shape[1]
        shift = torch.linspace(0, float(self.upper[0].item()), d, device=self.device, dtype=self.dtype)
        scaling = 10 ** (0.5 * torch.arange(d, device=self.device, dtype=self.dtype) / (d - 1))
        transform = self.Q @ torch.diag(scaling) @ self.R
        z = (x - shift - self.x_opt) @ transform
        s = torch.where(z * self.x_opt.reshape(1, -1) > 0, torch.full_like(z, 100.0), torch.ones_like(z))
        sum_sq = torch.sum((s * z) ** 2, dim=1)
        return (tosz(sum_sq) ** 0.9).reshape(-1, 1)
