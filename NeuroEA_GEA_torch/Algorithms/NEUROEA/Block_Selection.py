import numpy as np
import torch

from NeuroEA_GEA_torch.Algorithms.NEUROEA.BLOCK import Block
from NeuroEA_GEA_torch.utils.nd_sort import nd_sort


class Block_Selection(Block):
    def __init__(self, n_solutions=100, **kwargs):
        super().__init__(**kwargs)
        self.n_solutions = n_solutions

    def Main(self, problem, precursors, ratio):
        population = self.Gather(problem, precursors, ratio, 1, 1)
        if len(population) == 0:
            self.output = population
            return

        n_pop = len(population)
        num_to_keep = min(n_pop, self.n_solutions)

        if problem.m == 1:
            fitness = self._get_fitness_single(population)
            rank = torch.as_tensor(
                np.argsort(fitness.detach().cpu().numpy()),
                device=problem.device,
                dtype=torch.long,
            )
            self.output = population.take(rank[:num_to_keep])
            return

        front_no, max_f_no = nd_sort(population.obj, population.con, num_to_keep)
        next_idx = torch.where(front_no < max_f_no)[0]
        last_idx = torch.where(front_no == max_f_no)[0]

        k = len(next_idx) + len(last_idx) - num_to_keep
        if k > 0:
            del_mask = self._truncation(population.obj.index_select(0, last_idx), population.obj, k)
            selected_last = last_idx[~del_mask]
            self.output = population.take(torch.cat([next_idx, selected_last], dim=0))
        else:
            self.output = population.take(torch.cat([next_idx, last_idx], dim=0))

    def _get_fitness_single(self, population):
        objs = population.obj.reshape(-1)
        cons = population.con.reshape(-1)
        return objs + torch.clamp(cons, min=0)

    def _truncation(self, pop_obj_last, pop_obj_all, k):
        n_last = pop_obj_last.shape[0]
        del_mask = torch.zeros(n_last, device=pop_obj_last.device, dtype=torch.bool)
        distance = torch.cdist(pop_obj_last, pop_obj_all)

        for i in range(n_last):
            same = torch.all(pop_obj_all == pop_obj_last[i], dim=1)
            distance[i, same] = float("inf")

        while int(del_mask.sum().item()) < k:
            remain = torch.where(~del_mask)[0]
            temp_dist = torch.sort(distance.index_select(0, remain), dim=1).values.detach().cpu().numpy()
            worst_idx = np.lexsort(temp_dist.T[::-1])[0]
            del_mask[remain[worst_idx]] = True

        return del_mask
