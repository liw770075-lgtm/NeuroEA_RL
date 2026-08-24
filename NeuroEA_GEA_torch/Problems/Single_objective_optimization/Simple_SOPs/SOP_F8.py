import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem


class SOP_F8(Problem):
    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30

        self.lower = torch.full((self.D,), -500.0, dtype=self.dtype, device=self.device)
        self.upper = torch.full((self.D,), 500.0, dtype=self.dtype, device=self.device)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        offset = torch.linspace(0, float(self.upper[0].item()), self.D, device=self.device, dtype=self.dtype)
        shifted = pop_dec - offset
        inner = shifted * torch.sin(torch.sqrt(torch.abs(shifted)))
        return (-torch.sum(inner, dim=1)).reshape(-1, 1)

    def get_optimum(self, N=None):
        return torch.tensor([-418.9829 * self.D], device=self.device, dtype=self.dtype)
