from abc import ABC, abstractmethod

import torch

from NeuroEA_GEA_torch.Problems.POPULATION import Population
from NeuroEA_GEA_torch.utils.rng import NumpyTorchRNG


class Problem(ABC):
    """Torch version of the problem base class."""

    def __init__(
        self,
        N=100,
        M=1,
        D=10,
        max_fe=10000,
        max_runtime=float("inf"),
        parameter=None,
        device="cpu",
        dtype=torch.float64,
        rng=None,
    ):
        self.N = N
        self.M = M
        self.D = D
        self.max_fe = max_fe
        self.fe = 0
        self.max_runtime = max_runtime

        self.encoding = 1
        self.lower = torch.zeros(D, dtype=dtype)
        self.upper = torch.ones(D, dtype=dtype)
        self.optimum = None
        self.pf = None
        self.parameter = parameter if parameter is not None else []

        self.device = torch.device(device)
        self.dtype = dtype
        self.rng = rng if rng is not None else NumpyTorchRNG(seed=None, device=self.device, dtype=self.dtype)

        self.setting()

        self.lower = self._tensor(self.lower)
        self.upper = self._tensor(self.upper)
        if self.optimum is None:
            self.optimum = self.get_optimum(10000)
        self.optimum = self._tensor(self.optimum)

    @property
    def m(self):
        return self.M

    @property
    def d(self):
        return self.D

    def _tensor(self, value, dtype=None):
        tensor_dtype = self.dtype if dtype is None else dtype
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=tensor_dtype)
        return torch.as_tensor(value, device=self.device, dtype=tensor_dtype)

    @abstractmethod
    def setting(self):
        pass

    def initialization(self, N=None):
        if N is None:
            N = self.N

        rand = self.rng.rand(N, self.D)
        pop_dec = self.lower.unsqueeze(0) + (self.upper - self.lower).unsqueeze(0) * rand
        if self.encoding in [2, 3]:
            pop_dec = torch.round(pop_dec)
        return self.evaluation(pop_dec)

    def evaluation(self, pop_dec, *args):
        pop_dec = self.cal_dec(pop_dec)
        pop_obj = self.cal_obj(pop_dec)
        pop_con = self.cal_con(pop_dec)
        population = Population(pop_dec, pop_obj, pop_con, None)
        self.fe += len(population)
        return population

    def cal_dec(self, pop_dec):
        pop_dec = self._tensor(pop_dec)
        if pop_dec.ndim == 2 and pop_dec.shape[0] == 1 and pop_dec.shape[1] != self.D:
            pop_dec = pop_dec.reshape(-1, self.D)
        elif pop_dec.ndim == 1:
            pop_dec = pop_dec.reshape(-1, self.D)

        lower = self.lower.reshape(1, -1)
        upper = self.upper.reshape(1, -1)
        return torch.maximum(torch.minimum(pop_dec, upper), lower)

    @abstractmethod
    def cal_obj(self, pop_dec):
        return torch.zeros((len(pop_dec), self.M), device=self.device, dtype=self.dtype)

    def cal_con(self, pop_dec):
        return torch.zeros((len(pop_dec), 1), device=self.device, dtype=self.dtype)

    def get_optimum(self, N):
        if self.M > 1:
            return torch.ones((1, self.M), device=self.device, dtype=self.dtype)
        return torch.zeros(1, device=self.device, dtype=self.dtype)

    def parameter_set(self, *default_values):
        results = list(default_values)
        for i, p in enumerate(self.parameter):
            if p is not None:
                results[i] = p
        return results

    def cal_metric(self, metric_name, result):
        if not result:
            return float("inf")

        last_pop = result[-1][1]
        if metric_name == "Min_value":
            return float(torch.min(last_pop.obj).item())
        if metric_name == "HV":
            raise NotImplementedError("HV is not implemented in the torch port.")
        return 0.0
