import torch

from NeuroEA_GEA_torch.Algorithms.ALGORITHM import Algorithm
from NeuroEA_GEA_torch.Algorithms.GA.operators import operator_ga, tournament_selection
from NeuroEA_GEA_torch.Problems.POPULATION import Population
from NeuroEA_GEA_torch.utils.get_fitness_single import get_fitness_single


class GA(Algorithm):
    """PlatEMO-style genetic algorithm."""

    def main(self, problem):
        pro_c, dis_c, pro_m, dis_m = self.parameter_set(1.0, 20.0, 1.0, 20.0)
        population = problem.initialization()

        while self.not_terminated(population):
            mating_pool = tournament_selection(
                2,
                problem.N,
                get_fitness_single(population),
                rng=problem.rng,
                device=problem.device,
            )
            offspring = operator_ga(
                problem,
                population.take(mating_pool),
                (pro_c, dis_c, pro_m, dis_m),
            )
            population = self._survival(population, offspring, problem.N)

    def _survival(self, population, offspring, n_keep):
        merged = Population.cat([population, offspring])
        fitness = get_fitness_single(merged)
        rank = torch.argsort(fitness)
        return merged.take(rank[:n_keep])
