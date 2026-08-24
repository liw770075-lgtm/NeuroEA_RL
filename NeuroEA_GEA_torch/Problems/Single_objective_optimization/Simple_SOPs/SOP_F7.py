import torch

from NeuroEA_GEA_torch.Problems.PROBLEM import Problem


class SOP_F7(Problem):
    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30

        self.lower = torch.full((self.D,), -1.28, dtype=self.dtype, device=self.device)
        self.upper = torch.full((self.D,), 1.28, dtype=self.dtype, device=self.device)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        offset = torch.linspace(0, float(self.upper[0].item()), self.D, device=self.device, dtype=self.dtype)
        shifted = pop_dec - offset
        weights = torch.arange(1, self.D + 1, device=self.device, dtype=self.dtype)
        deterministic = torch.sum(weights.unsqueeze(0) * (shifted ** 4), dim=1)
        noise = self.rng.rand(len(pop_dec))
        return (deterministic + noise).reshape(-1, 1)
