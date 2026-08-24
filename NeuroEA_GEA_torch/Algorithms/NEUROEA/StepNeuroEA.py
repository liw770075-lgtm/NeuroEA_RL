import numpy as np
import torch

from NeuroEA_GEA_torch.Algorithms.GENERATION_ALGORITHM import GenerationAlgorithm
from NeuroEA_GEA_torch.Algorithms.NEUROEA.BLOCK import Block
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Population import Block_Population


class StepNeuroEA(GenerationAlgorithm):
    """
    按“代”推进的 NeuroEA。

    这个类把原始 NeuroEA 的“一次跑完整个优化过程”拆成了逐代接口：
    - `initialize()`：初始化问题和初始种群
    - `advance()` / `run_generation()`：只推进一代
    - `apply_update()`：在下一代前更新参数或结构

    说明：
    - `reset()` 也是可用的，但它不是在这个类里单独定义的，而是继承自 `GenerationAlgorithm`
    - 如果参数固定、不需要逐代干预，那么最简单的用法其实就是直接 `solve(problem)`

    最常用的三个状态接口：
    - `get_control_state()`：只看控制量，如 graph、block 参数、上下界
    - `get_generation_state()`：只看当前代运行状态和当前种群
    - `get_state()`：完整总状态，等于上面两类状态的合并

    常见用法示例

    1. 固定参数，直接完整运行：

    ```python
    alg = StepNeuroEA(blocks=blocks, graph=graph)
    alg.set_block_parameters(param_vector)
    alg.solve(problem)
    ```

    2. 固定参数，但手动一代一代运行：

    ```python
    alg = StepNeuroEA(blocks=blocks, graph=graph)
    alg.initialize(problem, update=param_vector)

    while not alg.is_done():
        alg.run_generation(param_vector)
        state = alg.get_state()
        dec = state["population_dec"]
        obj = state["population_obj"]
    ```

    3. 每一代都重新给参数，例如接 RL：

    ```python
    alg = StepNeuroEA(blocks=blocks, graph=graph)
    alg.initialize(problem, update=initial_param_vector)

    while not alg.is_done():
        action = agent.act(alg.get_state())
        state = alg.advance(action)
    ```

    4. 如果你只是想先更新，但暂时不执行这一代：

    ```python
    alg.apply_update({
        "blocks": new_blocks,
        "graph": new_graph,
        "block_parameters": new_param_vector,
    })
    state = alg.get_state()
    ```

    其中：
    - `advance()` 更像 RL 环境的 `step()`
    - `reset()` 如果要用，它来自 `GenerationAlgorithm`
    - `run_generation(update)` 和 `advance(update)` 都会先调用 `apply_update(update)`，再执行这一代
    """

    def __init__(self, blocks=None, graph=None, **kwargs):
        super().__init__(**kwargs)

        # blocks + graph 定义了一条具体的 NeuroEA 结构。
        self.blocks = [] if blocks is None else list(blocks)
        self.graph = self._normalize_graph(graph, len(self.blocks))

        # 标记哪些 block 是 population block。
        # 一代执行完成的判断是：所有 population block 都重新被激活。
        self._is_population = np.zeros(len(self.blocks), dtype=bool)

        if self.blocks:
            self._refresh_configuration(validate=True)

    # ------------------------------------------------------------------
    # 对外状态接口
    # ------------------------------------------------------------------
    def get_control_state(self):
        """
        返回控制层状态。

        这里不关心当前种群，只返回“你正在控制什么”：
        - 当前 blocks
        - 当前 graph
        - 当前 block 参数
        - 参数上下界
        """
        if not self.blocks:
            empty = torch.empty(0)
            return {
                "blocks": [],
                "graph": self.graph.copy(),
                "block_parameters": empty,
                "lower_bounds": empty.clone(),
                "upper_bounds": empty.clone(),
            }

        return {
            "blocks": list(self.blocks),
            "graph": self.graph.copy(),
            "block_parameters": Block.parameters(self.blocks).detach().clone(),
            "lower_bounds": Block.lowers(self.blocks).detach().clone(),
            "upper_bounds": Block.uppers(self.blocks).detach().clone(),
        }

    def get_population_state(self):
        """
        返回当前种群相关状态。

        这里专门把种群拆出来，避免都堆在 `get_state()` 里看不清。
        """
        if self.population is None:
            return {
                "population": None,
                "population_dec": None,
                "population_obj": None,
                "population_con": None,
                "best_fitness": float("inf"),
            }

        return {
            "population": self.population,
            "population_dec": self.population.dec.detach().clone(),
            "population_obj": self.population.obj.detach().clone(),
            "population_con": self.population.con.detach().clone(),
            "best_fitness": self._best_fitness(),
        }

    def get_generation_state(self):
        """
        返回“当前这一代”的运行状态。

        这里关注的是：
        - 当前 population
        - 当前 generation / FE / finished / runtime
        - 当前最优值

        不包含 graph 和参数上下界这类控制层信息。
        """
        state = super().get_generation_state()
        state.update(self.get_population_state())
        return state

    def get_block_parameters(self):
        """返回当前所有 block 参数拼接后的长向量。"""
        if not self.blocks:
            return torch.empty(0)
        return Block.parameters(self.blocks).detach().clone()

    # ------------------------------------------------------------------
    # 对外更新接口
    # ------------------------------------------------------------------
    def apply_update(self, update):
        """
        在下一代开始前应用外部更新。

        支持三类输入：
        1. `tensor / ndarray / list / tuple`：直接作为 block 参数向量
        2. `dict`：显式更新 `blocks / graph / block_parameters`

        注意：这里“只更新，不执行”。
        对常规逐代控制，更推荐直接使用：
        - `run_generation(update)`
        - `advance(update)`

        因为这两个接口本身就已经会先调用 `apply_update(update)`，
        然后再执行这一代。

        使用示例：
        - 只更新参数向量，但暂时不执行：
          `alg.apply_update(param_vector)`
        - 用 tensor 更新参数向量，但暂时不执行：
          `alg.apply_update(torch.tensor(param_vector))`
        - 同时更新结构和参数，但暂时不执行：
          `alg.apply_update({"blocks": new_blocks, "graph": new_graph, "block_parameters": new_params})`
        - 推荐的逐代执行方式：
          `alg.run_generation(param_vector)`
          `alg.advance(param_vector)`
        """
        if update is None:
            return self.get_control_state()

        if torch.is_tensor(update):
            self.set_block_parameters(update.detach().cpu())
            return self.get_control_state()

        if isinstance(update, np.ndarray):
            self.set_block_parameters(update)
            return self.get_control_state()

        if isinstance(update, (list, tuple)):
            self.set_block_parameters(update)
            return self.get_control_state()

        if not isinstance(update, dict):
            raise TypeError("StepNeuroEA update must be a dict, list, tuple, tensor, or numpy array.")

        # 先处理结构更新，再处理参数更新。
        configuration = update.get("configuration")
        if configuration is not None:
            if not isinstance(configuration, dict):
                raise TypeError("StepNeuroEA configuration update must be a dict.")
            self.set_configuration(
                blocks=configuration.get("blocks"),
                graph=configuration.get("graph"),
            )

        blocks = update.get("blocks")
        graph = update.get("graph")
        if blocks is not None or graph is not None:
            self.set_configuration(blocks=blocks, graph=graph)

        parameters = update.get("block_parameters")
        if parameters is None:
            parameters = update.get("parameters")
        if parameters is None:
            parameters = update.get("parameter_vector")
        if parameters is not None:
            self.set_block_parameters(parameters)

        return self.get_control_state()

    def set_configuration(self, blocks=None, graph=None):
        """
        更新当前 NeuroEA 的结构定义。

        这允许你在两代之间替换：
        - block 列表
        - graph 邻接矩阵

        如果算法已经初始化过，那么更新结构后，会把“当前种群”重新写入
        新的 population block，使下一代从当前种群继续往下跑。
        """
        if blocks is not None:
            self.blocks = list(blocks)

        if graph is not None:
            self.graph = self._normalize_graph(graph, len(self.blocks))
        elif self.blocks and self.graph.size == 0:
            self.graph = self._normalize_graph(None, len(self.blocks))

        self._refresh_configuration(validate=True)

        if self.initialized and self.population is not None:
            self._reset_block_runtime_state()
            self._initialize_population_blocks(self.population)

        return self.get_control_state()

    def set_block_parameters(self, parameters):
        """按 block 顺序把一维参数向量分发给每个 block。"""
        if not self.blocks:
            raise ValueError("StepNeuroEA blocks are not configured.")
        Block.ParameterSet(self.blocks, parameters)
        return self.get_block_parameters()

    # ------------------------------------------------------------------
    # GenerationAlgorithm 需要子类实现的核心接口
    # ------------------------------------------------------------------
    def initialize_population(self, problem):
        """
        初始化 StepNeuroEA 的第一代。

        这里做三件事：
        1. 检查结构是否合法
        2. 清空上次运行遗留的 block 输出缓存
        3. 生成初始种群，并写入 population block
        """
        self._refresh_configuration(validate=True)
        self._reset_block_runtime_state()

        initial_population = problem.initialization()
        self._initialize_population_blocks(initial_population)
        return self.blocks[0].output

    def evolve_one_generation(self, population):
        """
        执行一代完整的 block 图调度。

        整体逻辑沿用原始 NeuroEA：
        - 在图上找“前驱都准备好”的 block
        - 执行该 block
        - 重复直到所有 population block 都被重新激活

        返回值是新一代主种群，即 `blocks[0].output`。
        """
        if population is None:
            raise RuntimeError("StepNeuroEA population is not initialized.")

        num_blocks = len(self.blocks)
        activated = np.zeros(num_blocks, dtype=bool)

        while not np.all(activated[self._is_population]):
            progressed = False
            for i in range(num_blocks):
                if activated[i]:
                    continue

                predecessors_mask = self.graph[:, i] > 0
                if np.all(activated[predecessors_mask] | self._is_population[predecessors_mask]):
                    precursors = [self.blocks[j] for j in range(num_blocks) if predecessors_mask[j]]
                    ratios = self.graph[predecessors_mask, i]
                    self.blocks[i].Main(self.problem, precursors, ratios)
                    activated[i] = True
                    progressed = True

            if not progressed:
                raise RuntimeError("StepNeuroEA generation execution stalled due to invalid activation order.")

        return self.blocks[0].output

    # ------------------------------------------------------------------
    # 内部辅助函数
    # ------------------------------------------------------------------
    def _refresh_configuration(self, validate):
        """根据当前 blocks 重建辅助缓存，并在需要时检查 graph 合法性。"""
        if not self.blocks:
            raise ValueError("StepNeuroEA requires configured blocks.")

        if self.graph.size == 0:
            raise ValueError("StepNeuroEA requires a non-empty graph.")

        self._is_population = np.asarray(
            [isinstance(block, Block_Population) for block in self.blocks],
            dtype=bool,
        )

        if validate:
            Block.Validity(self.blocks, self.graph)

    def _initialize_population_blocks(self, population):
        """让所有 population block 都从同一份当前种群开始。"""
        for i, is_population in enumerate(self._is_population):
            if is_population:
                self.blocks[i].initialization(population)

    def _reset_block_runtime_state(self):
        """清空 block 运行时缓存，但不修改 block 参数。"""
        for block in self.blocks:
            block.output = None
            block.next_out = 0

    def _best_fitness(self):
        """当前主要面向单目标场景，因此直接返回目标值最小值。"""
        if self.population is None or len(self.population) == 0:
            return float("inf")
        return float(torch.min(self.population.obj).item())

    @staticmethod
    def _normalize_graph(graph, n_blocks):
        """统一把 graph 转成 numpy 邻接矩阵。"""
        if graph is None:
            return np.empty((0, 0), dtype=float) if n_blocks == 0 else np.zeros((n_blocks, n_blocks), dtype=float)
        return np.asarray(graph, dtype=float)
