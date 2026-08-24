import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem


class SOP_F10(Problem):
    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30

        self.lower = torch.full((self.D,), -32.0, dtype=self.dtype, device=self.device)
        self.upper = torch.full((self.D,), 32.0, dtype=self.dtype, device=self.device)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        offset = torch.linspace(0, float(self.upper[0].item()), self.D, device=self.device, dtype=self.dtype)
        shifted = pop_dec - offset
        term1 = -20.0 * torch.exp(-0.2 * torch.sqrt(torch.mean(shifted ** 2, dim=1)))
        term2 = -torch.exp(torch.mean(torch.cos(2 * torch.pi * shifted), dim=1))
        return (term1 + term2 + 20.0 + torch.exp(torch.tensor(1.0, device=self.device, dtype=self.dtype))).reshape(-1, 1)
