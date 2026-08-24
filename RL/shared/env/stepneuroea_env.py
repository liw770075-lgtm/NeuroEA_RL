"""`StepNeuroEA` 的专用环境封装。

这里把默认 block 结构、问题构造和环境拼装放在一起，
方便直接看懂 StepNeuroEA 是如何被接进 RL 的。
"""

from __future__ import annotations

import numpy as np

from NeuroEA_GEA_torch.Algorithms.NEUROEA.BLOCK import Block
from RL.shared.env.stepwise_ea_env import (
    ImprovementReward,
    ParameterVectorActionAdapter,
    PopulationObservationBuilder,
    StepwiseEAEnv,
)
from RL.shared.env.problem_utils import (
    build_problem_instance,
    build_task_context,
    normalize_problem_config,
    normalize_tasks,
    resolve_device,
    resolve_dtype,
    select_task,
)
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Crossover import Block_Crossover
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Exchange import Block_Exchange
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Mutation import Block_Mutation
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Population import Block_Population
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Selection import Block_Selection
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Tournament import Block_Tournament
from NeuroEA_GEA_torch.Algorithms.NEUROEA.StepNeuroEA import StepNeuroEA
from NeuroEA_GEA_torch.main_test import DEFAULT_PROBLEM_PARA, EXAMPLE_PARAMETERS
from NeuroEA_GEA_torch.utils.rng import NumpyTorchRNG


def build_stepneuroea_algorithm(task, device="auto", dtype=None, seed=0):
    """
    构造 `StepNeuroEA` 算法实例。

    当前默认结构直接沿用 `main_test.py` 中那套 block + graph 定义。
    如果后面你要换默认结构，优先改这里，而不是再去追其他包装层。
    """
    problem_config = tuple(task.problem_config)
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    rng = NumpyTorchRNG(seed=seed, device=device, dtype=dtype)

    # 这里直接复用 `main_test.py` 那套默认 block 结构。
    # 这样训练环境和已有测试入口共享同一套 NeuroEA 结构假设。
    population_size = int(problem_config[0])
    blocks = [
        Block_Population(rng=rng, device=device, dtype=dtype),
        Block_Tournament(n_parents=2 * population_size, upper_k=10, rng=rng, device=device, dtype=dtype),
        Block_Tournament(n_parents=2 * population_size, upper_k=10, rng=rng, device=device, dtype=dtype),
        Block_Tournament(n_parents=2 * population_size, upper_k=10, rng=rng, device=device, dtype=dtype),
        Block_Exchange(n_parents=3, rng=rng, device=device, dtype=dtype),
        Block_Exchange(n_parents=3, rng=rng, device=device, dtype=dtype),
        Block_Exchange(n_parents=3, rng=rng, device=device, dtype=dtype),
        Block_Exchange(n_parents=3, rng=rng, device=device, dtype=dtype),
        Block_Crossover(n_parents=2, n_sets=3, rng=rng, device=device, dtype=dtype),
        Block_Mutation(n_sets=3, rng=rng, device=device, dtype=dtype),
        Block_Selection(n_solutions=population_size, rng=rng, device=device, dtype=dtype),
    ]

    graph = np.zeros((len(blocks), len(blocks)))
    graph[0, [1, 2, 3, 10]] = 1.0
    graph[1:4, 4:8] = 0.25
    graph[4:8, 8] = 1.0
    graph[8, 9] = 1.0
    graph[9, 10] = 1.0
    graph[10, 0] = 1.0

    Block.Validity(blocks, graph)
    return StepNeuroEA(blocks=blocks, graph=graph)


def resolve_stepneuroea_initial_update(initialization, algorithm, problem):
    """
    解析 episode 开始时使用的初始参数。

    常见写法：
    - `"current"`：直接用 block 当前自带参数
    - `"example"`：复用 main_test 里的 example 参数
    - callable：外部自定义初始化策略
    """
    if callable(initialization):
        return initialization(algorithm, problem)
    if initialization is None or initialization == "current":
        return algorithm.get_block_parameters()
    if initialization == "example":
        return np.asarray(EXAMPLE_PARAMETERS, dtype=np.float64)
    return initialization


class StepNeuroEAEnv(StepwiseEAEnv):
    """
    基于 `StepNeuroEA` 的强化学习环境。

    默认约定：
    - agent 动作是完整参数向量
    - 动作空间默认归一化到 [-1, 1]
    - 环境每次 `step()` 对应 StepNeuroEA 的“一代”
    - 如果传入 `tasks=[...]`，环境会在每个 episode 开始时选择一个任务

    说明：
    - 这里把 “默认问题构造 + 默认算法构造 + 环境封装” 放在一个文件里
    - 这样看 `StepNeuroEA` 对接逻辑时，只需要读这一个文件，不需要在 env/builder 间来回跳
    """

    def __init__(
        self,
        problem_name="BBOB_F1",
        problem_config=None,
        tasks=None,
        task_mode="cycle",
        initialization="current",
        device="auto",
        dtype=None,
        seed=0,
        action_adapter=None,
        observation_builder=None,
        reward_builder=None,
    ):
        """
        参数说明：
        - `problem_name/problem_config`：单任务入口
        - `tasks`：多任务入口，列表中的每一项至少要包含 `problem_name`，可选 `problem_config`
        - `task_mode`：`cycle` 或 `random`

        如果你要做多问题训练，推荐直接用 `tasks=[...]`。
        如果不同任务的维度不同，推荐使用：
        `PopulationObservationBuilder(include_population=False)`。
        """
        default_problem_config = normalize_problem_config(problem_config, default_problem_config=DEFAULT_PROBLEM_PARA)
        task_list = normalize_tasks(
            tasks=tasks,
            problem_name=problem_name,
            problem_config=default_problem_config,
            default_problem_config=DEFAULT_PROBLEM_PARA,
        )

        observation_builder = observation_builder or PopulationObservationBuilder()
        self.tasks = task_list
        self.task_mode = task_mode

        if observation_builder.include_population:
            # 如果 observation 里保留完整种群张量，那么所有任务的张量形状就必须一致。
            population_shapes = {(task.problem_config[0], task.problem_config[1], task.problem_config[2]) for task in task_list}
            if len(population_shapes) > 1:
                raise ValueError(
                    "When `include_population=True`, all tasks must share the same (N, M, D). "
                    "For multi-task training with varying dimensions, use `PopulationObservationBuilder(include_population=False)`."
                )

        algorithm_factory = lambda current_seed, task: build_stepneuroea_algorithm(
            task=task,
            device=device,
            dtype=dtype,
            seed=current_seed,
        )
        problem_factory = lambda current_seed, task: build_problem_instance(task=task, device=device, dtype=dtype, seed=current_seed)
        initial_update = lambda algorithm, problem: resolve_stepneuroea_initial_update(initialization, algorithm, problem)
        task_sampler = lambda current_seed, episode_index=0, options=None, prototype=False: select_task(
            task_list,
            episode_index=episode_index,
            seed=current_seed,
            mode=task_mode,
            options=options,
            prototype=prototype,
        )
        task_state_builder = lambda task, task_index: {
            # 这里把“任务名 + 配置 + one-hot/normalized context”写进 state，
            # 后面 reward / observation / logging 都可以直接复用。
            "task_name": task.problem_name,
            "task_config": np.asarray(task.problem_config, dtype=np.float32),
            "task_index": int(-1 if task_index is None else task_index),
            "task_context": build_task_context(task, -1 if task_index is None else task_index, task_list),
        }

        super().__init__(
            algorithm_factory=algorithm_factory,
            problem_factory=problem_factory,
            initial_update=initial_update,
            action_adapter=action_adapter or ParameterVectorActionAdapter(normalized=True),
            observation_builder=observation_builder,
            reward_builder=reward_builder or ImprovementReward(maximize=False),
            task_sampler=task_sampler,
            task_state_builder=task_state_builder,
            seed=seed,
        )


def create_stepneuroea_env(**kwargs):
    """便捷构造函数。"""
    return StepNeuroEAEnv(**kwargs)
