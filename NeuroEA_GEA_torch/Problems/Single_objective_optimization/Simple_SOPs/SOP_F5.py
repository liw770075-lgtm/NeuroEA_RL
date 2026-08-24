import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem


class SOP_F5(Problem):
    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30

        self.lower = torch.full((self.D,), -30.0, dtype=self.dtype, device=self.device)
        self.upper = torch.full((self.D,), 30.0, dtype=self.dtype, device=self.device)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        offset = torch.linspace(0, float(self.upper[0].item()), self.D, device=self.device, dtype=self.dtype)
        shifted = pop_dec - offset
        x_i = shifted[:, :-1]
        x_next = shifted[:, 1:]
        return torch.sum(100.0 * (x_next - x_i ** 2) ** 2 + (x_i - 1.0) ** 2, dim=1, keepdim=True)
