import time
from abc import ABC, abstractmethod

from NeuroEA_GEA_torch.Algorithms.ALGORITHM import Algorithm


class GenerationAlgorithm(Algorithm, ABC):
    """
    支持“初始化 -> 逐代推进”的算法基类。

    这套接口的核心约定是：
    - `initialize(problem, update=None)`：初始化算法；如果给了 `update`，会先调用 `apply_update(update)`
    - `run_generation(update=None)`：推进一代；如果给了 `update`，会先调用 `apply_update(update)`，再执行这一代
    - `advance(update=None)`：只是 `run_generation(update)` 的一个便捷包装，但返回的是完整 state

    也就是说，对大多数逐代控制场景：
    - 如果你想“更新参数并执行一代”，直接用 `run_generation(update)` 或 `advance(update)`
    - `apply_update(update)` 只在“想先更新，但暂时不执行这一代”时单独调用
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.population = None
        self.generation = 0
        self.initialized = False
        self.finished = False

    def solve(self, problem):
        self._run_to_completion(problem)

    def main(self, problem):
        self._run_to_completion(problem)

    def initialize(self, problem, update=None):
        self.result = []
        self.metric = {"runtime": 0.0}
        self.problem = problem
        self.problem.fe = 0
        self.start_time = time.time()

        if update is not None:
            self.apply_update(update)

        self.population = self.initialize_population(problem)
        self.generation = 0
        self.initialized = True
        self.finished = not self._record_generation(self.population)
        return self.population

    def reset(self, problem, update=None):
        self.initialize(problem, update=update)
        return self.get_state()

    def run_generation(self, update=None):
        """
        执行一代。

        如果传入 `update`，这里会先调用 `apply_update(update)`，
        然后再真正执行 `evolve_one_generation(...)`。
        """
        if not self.initialized:
            raise RuntimeError("Call initialize(problem) before run_generation().")
        if self.finished:
            return False

        if update is not None:
            self.apply_update(update)

        self.population = self.evolve_one_generation(self.population)
        self.generation += 1
        self.finished = not self._record_generation(self.population)
        return not self.finished

    def advance(self, update=None):
        """执行一代并返回最新状态。`update` 的处理规则与 `run_generation()` 完全一致。"""
        self.run_generation(update=update)
        return self.get_state()

    @abstractmethod
    def initialize_population(self, problem):
        pass

    @abstractmethod
    def evolve_one_generation(self, population):
        pass

    def apply_update(self, update):
        if update is None:
            return None
        raise NotImplementedError(f"{self.__class__.__name__} does not support external updates.")

    def get_population(self):
        return self.population

    def is_done(self):
        return self.finished

    def get_generation_state(self):
        """
        返回当前代的通用运行状态。

        子类如果希望补充“种群统计、最优值、额外观测”，优先重写这个函数。
        """
        return {
            "population": self.population,
            "generation": self.generation,
            "fe": self.problem.fe if self.problem is not None else 0,
            "finished": self.finished,
            "runtime": float(self.metric.get("runtime", 0.0)),
        }

    def get_control_state(self):
        """
        返回控制层状态。

        基类默认没有额外控制量，子类如果有“算法参数、结构、上下界”等信息，
        可以重写这个函数。
        """
        return {}

    def get_state(self):
        """返回完整状态，默认由 generation state 和 control state 合并得到。"""
        state = self.get_generation_state()
        state.update(self.get_control_state())
        return state

    def _record_generation(self, population):
        elapsed = time.time() - self.start_time
        self.metric["runtime"] += elapsed

        if self.problem.max_runtime < float("inf") and self.metric["runtime"] > 0:
            self.problem.max_fe = self.problem.fe * self.problem.max_runtime / self.metric["runtime"]

        if self.problem.max_fe > 0:
            self.result.append([self.problem.fe, population])

        no_finish = self.problem.fe < self.problem.max_fe
        self.start_time = time.time()
        return no_finish

    def _run_to_completion(self, problem):
        """运行到终止。`solve()` 和 `main()` 共用这条路径。"""
        self.initialize(problem)
        while self.run_generation():
            pass

