import numpy as np
import torch

from NeuroEA_GEA_torch.Algorithms.NEUROEA.BLOCK import Block


class Block_Exchange(Block):
    def __init__(self, n_parents=2, **kwargs):
        super().__init__(**kwargs)
        self.n_parents = n_parents
        self.lower = torch.full((n_parents,), 1e-20, device=self.device, dtype=self.dtype)
        self.upper = torch.ones(n_parents, device=self.device, dtype=self.dtype)
        self.parameter = self.rng.uniform(self.lower.cpu().numpy(), self.upper.cpu().numpy())
        self.fitness = None
        self.ParameterAssign()

    def ParameterAssign(self):
        self.fitness = torch.cumsum(self.parameter, dim=0)
        self.fitness = self.fitness / self.fitness[-1]

    def Main(self, problem, precursors, ratio):
        parent_dec = self.Gather(problem, precursors, ratio, 2, self.n_parents)
        if parent_dec.numel() == 0:
            self.output = parent_dec
            return

        n_offspring = parent_dec.shape[0] // self.n_parents
        n_vars = parent_dec.shape[1]
        rand_vals = self.rng.rand_numpy((n_offspring, n_vars))
        selector_np = np.searchsorted(self.fitness.detach().cpu().numpy(), rand_vals, side="left")
        parent_selector = torch.as_tensor(selector_np, device=problem.device, dtype=torch.long)

        reshaped = parent_dec.reshape(n_offspring, self.n_parents, n_vars)
        off_idx = torch.arange(n_offspring, device=problem.device).reshape(-1, 1)
        var_idx = torch.arange(n_vars, device=problem.device)
        self.output = reshaped[off_idx, parent_selector, var_idx]
