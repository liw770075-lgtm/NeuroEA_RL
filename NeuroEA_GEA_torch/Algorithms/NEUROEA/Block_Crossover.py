import numpy as np
import torch

from NeuroEA_GEA_torch.Algorithms.NEUROEA.BLOCK import Block


class Block_Crossover(Block):
    def __init__(self, n_parents=2, n_sets=5, **kwargs):
        super().__init__(**kwargs)
        self.n_parents = n_parents
        self.n_sets = n_sets

        param_dim = (self.n_parents - 1) * self.n_sets * 3
        self.lower = torch.as_tensor(
            np.tile([0.0, -1.0, 1e-20], (self.n_parents - 1) * self.n_sets),
            device=self.device,
            dtype=self.dtype,
        )
        self.upper = torch.ones(param_dim, device=self.device, dtype=self.dtype)

        self.parameter = self.rng.uniform(self.lower.cpu().numpy(), self.upper.cpu().numpy())
        self.weight = None
        self.fit = None
        self.ParameterAssign()

    def ParameterAssign(self):
        reshaped = self.parameter.reshape(self.n_sets, -1)
        self.weight = reshaped
        probs = reshaped[:, 2::3]
        cumsum = torch.cumsum(probs, dim=0)
        self.fit = cumsum / cumsum[-1:, :]

    def para_sampling(self, shape, weight, fit, device):
        rand_np = self.rng.rand_numpy(shape)
        fit_np = fit.detach().cpu().numpy()
        types_np = np.searchsorted(fit_np, rand_np, side="left")
        types = torch.as_tensor(types_np, device=device, dtype=torch.long)

        out = self.rng.standard_normal(shape).to(device=device, dtype=self.dtype)
        for i in range(len(fit_np)):
            mask = types == i
            if mask.any():
                out[mask] = out[mask] * weight[i, 0] + weight[i, 1]
        return out

    def Main(self, problem, precursors, ratio):
        parent_dec = self.Gather(problem, precursors, ratio, 2, self.n_parents)
        if parent_dec.numel() == 0:
            self.output = parent_dec
            return

        n_offspring = parent_dec.shape[0] // self.n_parents
        n_vars = parent_dec.shape[1]

        r_list = []
        for i in range(self.n_parents - 1):
            w_sub = self.weight[:, i * 3 : i * 3 + 2]
            f_sub = self.fit[:, i]
            r_sub = self.para_sampling((n_offspring, n_vars), w_sub, f_sub, problem.device)
            r_list.append(r_sub)

        r_weights = torch.stack(r_list, dim=1)
        parents = parent_dec.reshape(n_offspring, self.n_parents, n_vars)
        w_first = 1.0 - torch.sum(r_weights, dim=1, keepdim=True)
        weights = torch.cat([w_first, r_weights], dim=1)
        offspring = torch.sum(weights * parents, dim=1)

        if offspring.ndim == 1:
            offspring = offspring.reshape(-1, n_vars)
        elif offspring.shape[1] != n_vars:
            offspring = offspring.reshape(-1, n_vars)

        self.output = offspring
