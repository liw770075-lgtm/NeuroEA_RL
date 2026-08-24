import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem


class SOP_F6(Problem):
    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30

        self.lower = torch.full((self.D,), -100.0, dtype=self.dtype, device=self.device)
        self.upper = torch.full((self.D,), 100.0, dtype=self.dtype, device=self.device)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        offset = torch.linspace(0, float(self.upper[0].item()), self.D, device=self.device, dtype=self.dtype)
        shifted = pop_dec - offset
        stepped = torch.floor(shifted + 0.5)
        return torch.sum(stepped ** 2, dim=1, keepdim=True)
