import numpy as np
import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem


class BBOB_F1(Problem):
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
        shift = torch.linspace(0, float(self.upper[0].item()), self.D, device=self.device, dtype=self.dtype)
        z = x - shift - self.x_opt
        return torch.sum(z ** 2, dim=1, keepdim=True)
