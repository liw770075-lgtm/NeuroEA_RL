import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem


class SOP_F9(Problem):
    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30

        self.lower = torch.full((self.D,), -5.12, dtype=self.dtype, device=self.device)
        self.upper = torch.full((self.D,), 5.12, dtype=self.dtype, device=self.device)

    def cal_obj(self, pop_dec):
        shift = torch.linspace(0, float(self.upper[0].item()), self.D, device=self.device, dtype=self.dtype)
        temp = pop_dec - shift
        inner = temp ** 2 - 10 * torch.cos(2 * torch.pi * temp) + 10
        return torch.sum(inner, dim=1, keepdim=True)
