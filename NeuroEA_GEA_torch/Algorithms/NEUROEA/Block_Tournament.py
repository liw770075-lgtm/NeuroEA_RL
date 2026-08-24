import numpy as np
import torch

from NeuroEA_GEA_torch.Algorithms.NEUROEA.BLOCK import Block
from NeuroEA_GEA_torch.utils.get_fitness_single import get_fitness_single
from NeuroEA_GEA_torch.utils.nd_sort import nd_sort


class Block_Tournament(Block):
    def __init__(self, n_parents=100, upper_k=2, **kwargs):
        super().__init__(**kwargs)
        self.n_parents = n_parents
        self.lower = torch.tensor([1.0], device=self.device, dtype=self.dtype)
        self.upper = torch.tensor([float(upper_k)], device=self.device, dtype=self.dtype)
        self.parameter = self.rng.uniform(self.lower.cpu().numpy(), self.upper.cpu().numpy())
        self.n_tournament = 1
        self.ParameterAssign()

    def ParameterAssign(self):
        self.n_tournament = int(torch.round(self.parameter[0]).item())

    def Main(self, problem, precursors, ratio):
        population = self.Gather(problem, precursors, ratio, 1, 1)
        if len(population) == 0:
            self.output = population
            return

        n_pop = len(population)
        k = min(self.n_tournament, n_pop)

        if problem.m == 1:
            fitness = get_fitness_single(population)
            fitness_np = fitness.detach().cpu().numpy()
            rank = torch.as_tensor(np.argsort(np.argsort(fitness_np)), device=problem.device, dtype=torch.long)
        else:
            rank, _ = nd_sort(population.obj, population.con, n_pop)

        candidates_np = self.rng.randint_numpy(0, n_pop, size=(k, self.n_parents))
        candidates = torch.as_tensor(candidates_np, device=problem.device, dtype=torch.long)
        candidate_ranks = rank[candidates]
        best_rows = torch.argmin(candidate_ranks, dim=0)
        cols = torch.arange(self.n_parents, device=problem.device)
        final_indices = candidates[best_rows, cols]
        self.output = population.take(final_indices)
