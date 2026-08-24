import multiprocessing as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from NeuroEA_GEA_torch.Algorithms.NEUROEA.BLOCK import Block
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Crossover import Block_Crossover
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Exchange import Block_Exchange
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Mutation import Block_Mutation
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Population import Block_Population
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Selection import Block_Selection
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Tournament import Block_Tournament
from NeuroEA_GEA_torch.Algorithms.NEUROEA.NeuroEA import NeuroEA
from NeuroEA_GEA_torch.Problems import get_problem_instance
from NeuroEA_GEA_torch.utils.rng import NumpyTorchRNG


DEFAULT_PROBLEM_PARA = (100, 1, 10, 10000)


def run_single_evaluation_worker(payload):
    problem, blocks, graph = payload
    alg = NeuroEA(blocks=blocks, graph=graph)
    alg.solve(problem)

    if problem.m == 1:
        return float(problem.cal_metric("Min_value", alg.result))
    return float(-problem.cal_metric("HV", alg.result))


class ParameterTraining:
    """
    NeuroEA 参数训练与测试管理器（PyTorch 版本）
    """

    def __init__(
        self,
        problem_name,
        run_times,
        save_root,
        device="auto",
        dtype=None,
        seed=0,
        parallel=False,
        max_workers=None,
    ):
        self.run_times = run_times
        self.save_root = save_root
        self.problem_name = problem_name
        self.prob_para = tuple(DEFAULT_PROBLEM_PARA)
        self.seed = seed
        self.is_parallel = parallel
        self.max_workers = max_workers

        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)

        # 1. 默认问题参数设置 [N, M, D, maxFE]
        para = self.prob_para

        # 2. 构造待优化的目标问题 (Problem2Train)
        self.rng = NumpyTorchRNG(seed=self.seed, device=self.device, dtype=self.dtype)
        self.problem_to_train = self._build_problem(problem_name, para)

        # 3. 构造 GEA (NeuroEA) 的架构
        if Block_Population is None:
            print("警告：Block_Population 导入失败，当前为 None！")

        self.blocks = [
            Block_Population(rng=self.rng, device=self.device, dtype=self.dtype),  # 0
            Block_Tournament(n_parents=200, upper_k=10, rng=self.rng, device=self.device, dtype=self.dtype),  # 1
            Block_Tournament(200, 10, rng=self.rng, device=self.device, dtype=self.dtype),  # 2
            Block_Tournament(200, 10, rng=self.rng, device=self.device, dtype=self.dtype),  # 3
            Block_Exchange(n_parents=3, rng=self.rng, device=self.device, dtype=self.dtype),  # 4
            Block_Exchange(3, rng=self.rng, device=self.device, dtype=self.dtype),  # 5
            Block_Exchange(3, rng=self.rng, device=self.device, dtype=self.dtype),  # 6
            Block_Exchange(3, rng=self.rng, device=self.device, dtype=self.dtype),  # 7
            Block_Crossover(n_parents=2, n_sets=3, rng=self.rng, device=self.device, dtype=self.dtype),  # 8
            Block_Mutation(n_sets=3, rng=self.rng, device=self.device, dtype=self.dtype),  # 9
            Block_Selection(n_solutions=100, rng=self.rng, device=self.device, dtype=self.dtype),  # 10
        ]

        # 定义邻接矩阵 Graph
        num_blocks = len(self.blocks)
        self.graph = np.zeros((num_blocks, num_blocks))

        # 设置连接关系和流动比例 (Ratio)
        self.graph[0, [1, 2, 3, 10]] = 1.0
        self.graph[1:4, 4:8] = 0.25
        self.graph[4:8, 8] = 1.0
        self.graph[8, 9] = 1.0
        self.graph[9, 10] = 1.0
        self.graph[10, 0] = 1.0

        Block.Validity(self.blocks, self.graph)

    def testing(self, parameters, run_times=None, problem_name=None, prob_para=None):
        """
        测试一组特定参数在某个问题上的表现
        """
        actual_problem_name = self.problem_name if problem_name is None else problem_name
        actual_prob_para = self.prob_para if prob_para is None else tuple(prob_para)
        actual_run_times = self.run_times if run_times is None else run_times
        if actual_problem_name == self.problem_name and actual_prob_para == self.prob_para:
            problem = self.problem_to_train
        else:
            problem = self._build_problem(actual_problem_name, actual_prob_para)

        self._distribute_parameters(parameters)

        if self.is_parallel and actual_run_times > 1 and self.device.type == "cpu":
            return self._testing_parallel(problem, actual_run_times)
        fitness_results = [self._run_single_evaluation(problem) for _ in range(actual_run_times)]


        return float(np.median(fitness_results))

    def _testing_parallel(self, problem, run_times):
        max_workers = self.max_workers or min(run_times, mp.cpu_count() or 1)
        max_workers = 4
        payloads = [(problem, self.blocks, self.graph) for _ in range(run_times)]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            fitness_results = list(executor.map(run_single_evaluation_worker, payloads))

        return float(np.median(fitness_results))

    def _run_single_evaluation(self, problem):
        """
        运行一次完整的 EA 流程并计算指标
        """
        alg = NeuroEA(blocks=self.blocks, graph=self.graph)
        alg.solve(problem)

        if problem.m == 1:
            return float(problem.cal_metric("Min_value", alg.result))
        return float(-problem.cal_metric("HV", alg.result))

    def _distribute_parameters(self, par_vector):
        """将一维参数向量 Parameters 依次分发给各个 Block"""
        par_vector = np.asarray(par_vector, dtype=np.float64).reshape(-1)
        expected = int(Block.parameters(self.blocks).numel())
        if par_vector.size != expected:
            raise ValueError(f"Expected {expected} block parameters, got {par_vector.size}.")
        Block.ParameterSet(self.blocks, par_vector)

    def _build_problem(self, problem_name, prob_para):
        return get_problem_instance(
            problem_name,
            *prob_para,
            device=self.device,
            dtype=self.dtype,
            rng=self.rng,
        )

    def _resolve_device(self, device):
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _resolve_dtype(self, dtype):
        if dtype is None:
            return torch.float32 if self.device.type != "cpu" else torch.float64
        if isinstance(dtype, str):
            return getattr(torch, dtype)
        return dtype


EXAMPLE_PARAMETERS = [
    7.328845916458512,
    9.519452716194290,
    6.699378970058081,
    0.504833860696245,
    0.871154112250468,
    0.451289503013029,
    0.277377781142785,
    0.999699252289889,
    0.777907300954096,
    0.0404806672600856,
    0.368156506713865,
    0.728384642089435,
    0.257697135783844,
    0.854586114401756,
    0.404668247736162,
    0.513535549249738,
    -0.316033255651616,
    0.158972592688388,
    0.304425102233896,
    0.854778369939324,
    0.304690855887451,
    0.245240419885119,
    0.190764560248536,
    0.0911426508854076,
    0.0884953899671375,
    0.0363972714169237,
    0.0,
    0.200203031642368,
    0.0327544992994410,
    0.00772647381847839,
]


if __name__ == "__main__":
    problem = "BBOB_F1"
    run_times = 30
    save_root = "results"

    trainer = ParameterTraining(
        problem,
        run_times,
        save_root,
        device="auto",
        dtype="float64" if not torch.cuda.is_available() else "float32",
        seed=0,
        parallel=False,
    )
    fitness = trainer.testing(EXAMPLE_PARAMETERS)
    print(f"torch最优适应度: {fitness}")
