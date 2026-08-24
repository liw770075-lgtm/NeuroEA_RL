import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem


class SOP_F2(Problem):
    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30

        self.lower = torch.full((self.D,), -10.0, dtype=self.dtype, device=self.device)
        self.upper = torch.full((self.D,), 10.0, dtype=self.dtype, device=self.device)

    def cal_obj(self, pop_dec):
        shift = torch.linspace(0, float(self.upper[0].item()), self.D, device=self.device, dtype=self.dtype)
        temp = torch.abs(pop_dec - shift)
        return (torch.sum(temp, dim=1) + torch.prod(temp, dim=1)).reshape(-1, 1)
