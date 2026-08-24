import time
from abc import ABC, abstractmethod


class Algorithm(ABC):
    def __init__(self, parameter=None, save_count=-10, run_id=None, met_name=None):
        self.parameter = parameter if parameter is not None else []
        self.save_count = save_count
        self.run_id = run_id
        self.met_name = met_name if met_name is not None else []

        self.problem = None
        self.result = []
        self.metric = {"runtime": 0.0}
        self.start_time = 0.0
        self.output_fcn = self.default_output

    def solve(self, problem):
        try:
            self.result = []
            self.metric = {"runtime": 0.0}
            self.problem = problem
            self.problem.fe = 0
            self.start_time = time.time()
            self.main(self.problem)
        except Exception as exc:
            if str(exc) != "Termination":
                raise

    @abstractmethod
    def main(self, problem):
        pass

    def not_terminated(self, population):
        elapsed = time.time() - self.start_time
        self.metric["runtime"] += elapsed

        if self.problem.max_runtime < float("inf"):
            self.problem.max_fe = self.problem.fe * self.problem.max_runtime / self.metric["runtime"]

        if self.problem.max_fe > 0:
            self.result.append([self.problem.fe, population])

        no_finish = self.problem.fe < self.problem.max_fe
        self.start_time = time.time()
        if not no_finish:
            raise Exception("Termination")
        return no_finish

    def parameter_set(self, *default_values):
        results = list(default_values)
        for i in range(min(len(self.parameter), len(results))):
            if self.parameter[i] is not None:
                results[i] = self.parameter[i]
        return results

    def default_output(self, problem):
        progress = (problem.fe / problem.max_fe) * 100 if problem.max_fe > 0 else 0
        print(
            f"{self.__class__.__name__} on {problem.m}_obj {problem.d}_var "
            f"{problem.__class__.__name__} ({progress:6.2f}%), {self.metric['runtime']:.2f}s passed..."
        )
