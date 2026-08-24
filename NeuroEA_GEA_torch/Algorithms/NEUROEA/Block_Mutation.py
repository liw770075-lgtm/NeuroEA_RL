import numpy as np
import torch

from NeuroEA_GEA_torch.Algorithms.NEUROEA.BLOCK import Block


class Block_Mutation(Block):
    def __init__(self, n_sets=5, **kwargs):
        super().__init__(**kwargs)
        self.n_sets = n_sets
        self.n_dec = 1

        self.lower = torch.as_tensor([0.0, 1e-20] * self.n_sets, device=self.device, dtype=self.dtype)
        self.upper = torch.as_tensor([1.0, 2.0] * self.n_sets, device=self.device, dtype=self.dtype)
        self.parameter = self.rng.uniform(self.lower.cpu().numpy(), torch.ones_like(self.upper).cpu().numpy())
        self.weight = None
        self.fit = None
        self.ParameterAssign()

    def ParameterAssign(self):
        weight = self.parameter.reshape(self.n_sets, 2).clone()
        weight[:, 1] = weight[:, 1] / self.n_dec
        remain_prob = torch.clamp(1.0 - torch.sum(weight[:, 1]), min=0.0)
        tail = torch.tensor([[0.0, remain_prob.item()]], device=self.device, dtype=self.dtype)
        self.weight = torch.cat([weight, tail], dim=0)
        cumsum = torch.cumsum(self.weight[:, 1], dim=0)
        self.fit = cumsum / torch.max(cumsum)

    def para_sampling(self, shape, weights, fit, device):
        rand_np = self.rng.rand_numpy(shape)
        fit_np = fit.detach().cpu().numpy()
        types_np = torch.as_tensor(np.searchsorted(fit_np, rand_np, side="left"), device=device, dtype=torch.long)
        out = self.rng.standard_normal(shape).to(device=device, dtype=self.dtype)
        for i in range(len(fit_np)):
            mask = types_np == i
            if mask.any():
                out[mask] = out[mask] * weights[i]
        return out

    def Main(self, problem, precursors, ratio):
        parent_dec = self.Gather(problem, precursors, ratio, 2, 1)
        if parent_dec.numel() == 0:
            self.output = parent_dec
            return

        if parent_dec.shape[1] != self.n_dec:
            self.n_dec = parent_dec.shape[1]
            self.ParameterAssign()

        r = self.para_sampling(parent_dec.shape, self.weight[:, 0], self.fit, problem.device)
        range_dec = problem.upper - problem.lower
        offspring = parent_dec + range_dec.reshape(1, -1) * r
        self.output = torch.maximum(
            torch.minimum(offspring, problem.upper.reshape(1, -1)),
            problem.lower.reshape(1, -1),
        )
