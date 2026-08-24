import numpy as np
import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem
from NeuroEA_GEA_torch.Problems.problem_ops import make_orthogonal_matrix


class BBOB_F9(Problem):
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
        self.R = make_orthogonal_matrix(self.D, 2, self.device, self.dtype)

    def cal_obj(self, x):
        _, d = x.shape
        shift = torch.linspace(0, float(self.upper[0].item()), d, device=self.device, dtype=self.dtype)
        scale = max(1.0, float(np.sqrt(d) / 8.0))
        z = scale * ((x - shift - self.x_opt) @ self.R) + 0.5
        term1 = 100.0 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2
        term2 = (z[:, :-1] - 1.0) ** 2
        return torch.sum(term1 + term2, dim=1, keepdim=True)
