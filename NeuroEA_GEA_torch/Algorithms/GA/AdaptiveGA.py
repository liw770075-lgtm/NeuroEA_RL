import numpy as np
import torch

from NeuroEA_GEA_torch.Algorithms.GA.operators import operator_ga, tournament_selection
from NeuroEA_GEA_torch.Algorithms.GENERATION_ALGORITHM import GenerationAlgorithm
from NeuroEA_GEA_torch.Problems.POPULATION import Population
from NeuroEA_GEA_torch.utils.get_fitness_single import get_fitness_single


class AdaptiveGA(GenerationAlgorithm):
    """
    支持逐代调参的 GA。

    这个类基于 `GenerationAlgorithm`，因此既可以像普通算法一样直接
    `solve(problem)` 跑完整个过程，也可以按“代”推进。

    常见用法示例

    1. 固定参数，直接完整运行：

    ```python
    alg = AdaptiveGA(parameter=[1.0, 20.0, 1.0, 20.0], tournament_k=2)
    alg.solve(problem)
    ```

    2. 固定参数，但手动一代一代运行：

    ```python
    alg = AdaptiveGA(parameter=[1.0, 20.0, 1.0, 20.0], tournament_k=2)
    alg.initialize(problem)

    while not alg.is_done():
        alg.run_generation()
        state = alg.get_state()
    ```

    3. 每一代都重新给参数：

    ```python
    alg = AdaptiveGA()
    alg.initialize(problem)

    while not alg.is_done():
        action = {
            "pro_c": 1.0,
            "dis_c": 20.0,
            "pro_m": 0.5,
            "dis_m": 10.0,
            "tournament_k": 3,
        }
        state = alg.advance(action)
    ```

    4. 如果只是想先更新参数，但暂时不执行这一代：

    ```python
    alg.apply_update({"pro_m": 0.5, "dis_m": 10.0, "tournament_k": 3})
    state = alg.get_state()
    ```

    其中：
    - `advance(update)` 更像 RL 环境里的 `step(action)`
    - `run_generation(update)` 和 `advance(update)` 都会先调用 `apply_update(update)`，再执行这一代
    """

    def __init__(self, tournament_k=2, **kwargs):
        super().__init__(**kwargs)
        self.tournament_k = max(1, int(tournament_k))

    def initialize_population(self, problem):
        return problem.initialization()

    def evolve_one_generation(self, population):
        
        pro_c, dis_c, pro_m, dis_m = self.parameter_set(1.0, 20.0, 1.0, 20.0)
        mating_pool = tournament_selection(
            self.tournament_k,
            self.problem.N,
            get_fitness_single(population),
            rng=self.problem.rng,
            device=self.problem.device,
        )
        offspring = operator_ga(
            self.problem,
            population.take(mating_pool),
            (pro_c, dis_c, pro_m, dis_m),
        )
        return self._survival(population, offspring, self.problem.N)

    def set_operator_parameters(self, pro_c=None, dis_c=None, pro_m=None, dis_m=None):
        current = list(self.parameter_set(1.0, 20.0, 1.0, 20.0))
        updates = [pro_c, dis_c, pro_m, dis_m]
        for idx, value in enumerate(updates):
            if value is not None:
                current[idx] = value
        self.parameter = current
        return tuple(current)

    def set_tournament_size(self, tournament_k):
        self.tournament_k = max(1, int(tournament_k))
        return self.tournament_k

    def apply_update(self, update):
        if update is None:
            return self.get_control_state()

        update_dict = self._normalize_update(update)

        operator_kwargs = {}
        for key in ("pro_c", "dis_c", "pro_m", "dis_m"):
            if key in update_dict:
                operator_kwargs[key] = update_dict[key]

        if operator_kwargs:
            self.set_operator_parameters(**operator_kwargs)

        if "tournament_k" in update_dict:
            self.set_tournament_size(update_dict["tournament_k"])

        return self.get_control_state()

    def get_generation_state(self):
        state = super().get_generation_state()
        state["best_fitness"] = self._best_fitness()
        return state

    def get_control_state(self):
        pro_c, dis_c, pro_m, dis_m = self.parameter_set(1.0, 20.0, 1.0, 20.0)
        return {
            "pro_c": float(pro_c),
            "dis_c": float(dis_c),
            "pro_m": float(pro_m),
            "dis_m": float(dis_m),
            "tournament_k": int(self.tournament_k),
        }

    def _survival(self, population, offspring, n_keep):
        merged = Population.cat([population, offspring])
        fitness = get_fitness_single(merged)
        rank = torch.argsort(fitness)
        return merged.take(rank[:n_keep])

    def _best_fitness(self):
        if self.population is None or len(self.population) == 0:
            return float("inf")
        return float(torch.min(self.population.obj).item())

    def _normalize_update(self, update):
        if torch.is_tensor(update):
            update = update.detach().cpu().tolist()
        elif isinstance(update, np.ndarray):
            update = update.tolist()

        if isinstance(update, dict):
            if "operator_parameters" in update:
                operator_update = self._normalize_update(update["operator_parameters"])
                merged = {k: v for k, v in update.items() if k != "operator_parameters"}
                merged.update(operator_update)
                return merged
            return dict(update)

        if isinstance(update, (list, tuple)):
            if len(update) not in (4, 5):
                raise ValueError("AdaptiveGA update sequence must have length 4 or 5.")
            keys = ["pro_c", "dis_c", "pro_m", "dis_m"]
            values = list(update[:4])
            update_dict = {key: float(value) for key, value in zip(keys, values)}
            if len(update) == 5:
                update_dict["tournament_k"] = int(update[4])
            return update_dict

        raise TypeError("AdaptiveGA update must be a dict, list, tuple, tensor, or numpy array.")
