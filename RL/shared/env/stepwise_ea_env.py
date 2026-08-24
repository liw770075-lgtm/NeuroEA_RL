"""通用 step-wise EA 环境。

只要某个进化算法支持：
- reset/initialize
- run_generation/advance
- get_state

就可以通过这个环境层接进 RL。
"""

from __future__ import annotations

from collections import defaultdict, deque
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from RL.shared.env.spaces import EnvBase, spaces


def _to_numpy(value, dtype=np.float32):
    """把 tensor / list / scalar 统一转成 numpy，方便环境层做数值处理。"""
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _extract_best_value(state: Dict[str, Any]) -> float:
    """
    从算法 state 中提取“当前最好值”。

    约定优先级：
    1. 如果算法已经显式提供 `best_fitness`
    2. 否则从 `population_obj` 里取最小值
    """
    if "best_fitness" in state and state["best_fitness"] is not None:
        return float(state["best_fitness"])

    population_obj = state.get("population_obj")
    if population_obj is None:
        return 0.0
    population_obj = _to_numpy(population_obj, dtype=np.float64)
    if population_obj.size == 0:
        return 0.0
    return float(np.min(population_obj))


def _extract_task_key(state: Dict[str, Any]):
    """
    为多任务 reward 提供一个稳定的 task key。

    优先使用 `task_name`，如果没有，再退回 `task_index`。
    如果当前是单任务环境，这里会返回 `"default"`。
    """
    if state.get("task_name") is not None:
        return str(state["task_name"])
    if state.get("task_index") is not None:
        return f"task_{int(state['task_index'])}"
    return "default"


def _resolve_optimum_value(state: Dict[str, Any], optimum=0.0, optimum_by_task=None):
    """
    解析“已知最优值”。

    支持：
    - 一个全局标量 optimum
    - 一个按 task_name / task_index 分配的 dict
    """
    if optimum_by_task is None:
        return float(optimum)

    task_name = state.get("task_name")
    if task_name is not None and task_name in optimum_by_task:
        return float(optimum_by_task[task_name])

    task_index = state.get("task_index")
    if task_index is not None and task_index in optimum_by_task:
        return float(optimum_by_task[task_index])

    return float(optimum)


def _safe_gap(value: float, optimum: float = 0.0, eps: float = 1e-30):
    """
    把目标值转成“到最优值的 gap”。

    - 已知最优值时：gap = |f - f*|
    - 未知最优值但假设最优接近 0 时：gap = |f|
    """
    return max(abs(float(value) - float(optimum)), float(eps))


class ParameterVectorActionAdapter:
    """
    将 agent 动作映射成算法 update。

    默认模式下，agent 输出的是 [-1, 1] 区间内的连续动作，
    环境会根据算法暴露的 lower/upper bounds 反归一化为真实参数向量。
    """

    def __init__(self, normalized: bool = True, clip: bool = True):
        self.normalized = normalized
        self.clip = clip

    def build_space(self, state: Dict[str, Any]):
        # 动作空间直接由算法当前参数上下界决定。
        lower = self._require_vector(state, "lower_bounds")
        upper = self._require_vector(state, "upper_bounds")
        if self.normalized:
            return spaces.Box(low=-1.0, high=1.0, shape=lower.shape, dtype=np.float32)
        return spaces.Box(low=lower, high=upper, shape=lower.shape, dtype=np.float32)

    def to_update(self, action, state: Dict[str, Any]):
        # 在 step() 时，把 agent 的动作变回算法真正需要的参数向量。
        lower = self._require_vector(state, "lower_bounds")
        upper = self._require_vector(state, "upper_bounds")
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.shape != lower.shape:
            raise ValueError(f"Expected action shape {lower.shape}, got {action.shape}.")

        if self.normalized:
            if self.clip:
                action = np.clip(action, -1.0, 1.0)
            return lower + (action + 1.0) * 0.5 * (upper - lower)

        if self.clip:
            action = np.clip(action, lower, upper)
        return action

    @staticmethod
    def _require_vector(state: Dict[str, Any], key: str):
        value = state.get(key)
        if value is None:
            raise ValueError(f"Action adapter requires `{key}` in algorithm state.")
        value = _to_numpy(value, dtype=np.float32).reshape(-1)
        return value


class PopulationObservationBuilder:
    """
    默认 observation 构造器。

    输出一个字典：
    - `summary`: 当前代的统计特征
    - `control`: 当前控制向量（通常是参数向量，已归一化）
    - `population_dec / obj / con`: 当前种群张量
    """

    def __init__(self, include_population: bool = True, normalize_control: bool = True):
        self.include_population = include_population
        self.normalize_control = normalize_control

    def build_space(self, state: Dict[str, Any], history_states=None):
        # 用一次真实 observation 作为模板，自动推导 observation_space。
        observation = self(state, history_states=history_states)
        return spaces.Dict(
            {
                key: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=np.asarray(value).shape,
                    dtype=np.float32,
                )
                for key, value in observation.items()
            }
        )

    def __call__(self, state: Dict[str, Any], history_states=None):
        # 这里把算法 state 拆成“固定长度统计特征 + 可选的完整种群张量”。
        population_dec = _to_numpy(state.get("population_dec"))
        population_obj = _to_numpy(state.get("population_obj"))
        population_con = _to_numpy(state.get("population_con"))
        task_context = _to_numpy(state.get("task_context"))

        control = self._build_control_vector(state)
        summary = self._build_summary(state, population_dec, population_obj, population_con, control)

        observation = {"summary": summary.astype(np.float32)}
        if control is not None:
            observation["control"] = control.astype(np.float32)

        if self.include_population:
            if population_dec is not None:
                observation["population_dec"] = population_dec.astype(np.float32)
            if population_obj is not None:
                observation["population_obj"] = population_obj.astype(np.float32)
            if population_con is not None:
                observation["population_con"] = population_con.astype(np.float32)

        if task_context is not None:
            observation["task"] = task_context.astype(np.float32).reshape(-1)

        return observation

    def _build_control_vector(self, state: Dict[str, Any]):
        if "block_parameters" in state:
            control = _to_numpy(state["block_parameters"])
            if self.normalize_control and "lower_bounds" in state and "upper_bounds" in state:
                lower = _to_numpy(state["lower_bounds"])
                upper = _to_numpy(state["upper_bounds"])
                denom = np.where(np.abs(upper - lower) < 1e-12, 1.0, upper - lower)
                control = 2.0 * (control - lower) / denom - 1.0
            return control.reshape(-1)

        scalars = []
        for key, value in state.items():
            if key in {
                "population",
                "population_dec",
                "population_obj",
                "population_con",
                "task_context",
                "task_name",
                "task_index",
                "task_config",
                "blocks",
                "graph",
                "lower_bounds",
                "upper_bounds",
            }:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                scalars.append(float(value))
        if scalars:
            return np.asarray(scalars, dtype=np.float32)
        return None

    def _build_summary(self, state, population_dec, population_obj, population_con, control):
        generation = float(state.get("generation", 0))
        fe = float(state.get("fe", 0))
        best = _extract_best_value(state)

        mean_obj = float(np.mean(population_obj)) if population_obj is not None and population_obj.size > 0 else 0.0
        std_obj = float(np.std(population_obj)) if population_obj is not None and population_obj.size > 0 else 0.0
        mean_con = float(np.mean(population_con)) if population_con is not None and population_con.size > 0 else 0.0
        diversity = float(np.mean(np.std(population_dec, axis=0))) if population_dec is not None and population_dec.size > 0 else 0.0
        control_norm = float(np.linalg.norm(control)) if control is not None and control.size > 0 else 0.0
        done_flag = 1.0 if state.get("finished", False) else 0.0

        return np.asarray(
            [generation, fe, best, mean_obj, std_obj, mean_con, diversity, control_norm, done_flag],
            dtype=np.float32,
        )


class ImprovementReward:
    """默认奖励：本代最优值相对上一代的改进量。默认按最小化问题处理。"""

    def __init__(self, maximize: bool = False):
        self.maximize = maximize

    def __call__(self, previous_state: Dict[str, Any], current_state: Dict[str, Any], update: Any = None) -> float:
        prev_best = _extract_best_value(previous_state)
        curr_best = _extract_best_value(current_state)
        if self.maximize:
            return float(curr_best - prev_best)
        return float(prev_best - curr_best)


class RelativeImprovementReward:
    """
    相对改进奖励。

    对多任务更友好，因为不同任务的目标值量级可能差很多。
    这里会用“改进量 / 上一代 best 的绝对值”来做归一化。
    """

    def __init__(self, maximize: bool = False, eps: float = 1e-8, clip: float | None = None):
        self.maximize = maximize
        self.eps = float(eps)
        self.clip = None if clip is None else float(clip)

    def __call__(self, previous_state: Dict[str, Any], current_state: Dict[str, Any], update: Any = None) -> float:
        prev_best = _extract_best_value(previous_state)
        curr_best = _extract_best_value(current_state)
        scale = max(abs(prev_best), self.eps)

        if self.maximize:
            reward = (curr_best - prev_best) / scale
        else:
            reward = (prev_best - curr_best) / scale

        if self.clip is not None:
            reward = float(np.clip(reward, -self.clip, self.clip))
        return float(reward)


class LogGapReward:
    """
    基于 log-gap 的奖励。

    适用场景：
    - 问题通常是最小化
    - 最优值通常接近 0
    - 希望“减少一个数量级”和“减少另一个数量级”有相似奖励

    公式：
    reward = log(gap_prev) - log(gap_curr)
    其中 gap = max(|best|, eps)

    例子：
    - 1e-2 -> 1e-3
    - 1e-12 -> 1e-13

    这两种都会得到接近的正奖励，因为它们本质上都表示“减少了一个数量级”。
    """

    def __init__(self, maximize: bool = False, eps: float = 1e-30, clip: float | None = None):
        self.maximize = maximize
        self.eps = float(eps)
        self.clip = None if clip is None else float(clip)

    def __call__(self, previous_state: Dict[str, Any], current_state: Dict[str, Any], update: Any = None) -> float:
        prev_gap = _safe_gap(_extract_best_value(previous_state), optimum=0.0, eps=self.eps)
        curr_gap = _safe_gap(_extract_best_value(current_state), optimum=0.0, eps=self.eps)

        if self.maximize:
            reward = np.log(curr_gap) - np.log(prev_gap)
        else:
            reward = np.log(prev_gap) - np.log(curr_gap)

        if self.clip is not None:
            reward = float(np.clip(reward, -self.clip, self.clip))
        return float(reward)


class KnownOptimumLogGapReward:
    """
    已知最优值时使用的 log-gap 奖励。

    公式：
    reward = log(|f_prev - f*|) - log(|f_curr - f*|)

    其中 `f*` 可以是：
    - 一个全局标量 optimum
    - 一个按任务名/任务索引区分的 `optimum_by_task`

    这个版本比 `LogGapReward` 更通用，因为它不强制要求最优值一定接近 0。
    """

    def __init__(
        self,
        optimum: float = 0.0,
        optimum_by_task: Dict[Any, float] | None = None,
        maximize: bool = False,
        eps: float = 1e-30,
        clip: float | None = None,
    ):
        self.optimum = float(optimum)
        self.optimum_by_task = optimum_by_task
        self.maximize = maximize
        self.eps = float(eps)
        self.clip = None if clip is None else float(clip)

    def __call__(self, previous_state: Dict[str, Any], current_state: Dict[str, Any], update: Any = None) -> float:
        optimum = _resolve_optimum_value(current_state, optimum=self.optimum, optimum_by_task=self.optimum_by_task)
        prev_gap = _safe_gap(_extract_best_value(previous_state), optimum=optimum, eps=self.eps)
        curr_gap = _safe_gap(_extract_best_value(current_state), optimum=optimum, eps=self.eps)

        if self.maximize:
            reward = np.log(curr_gap) - np.log(prev_gap)
        else:
            reward = np.log(prev_gap) - np.log(curr_gap)

        if self.clip is not None:
            reward = float(np.clip(reward, -self.clip, self.clip))
        return float(reward)


class AdaptiveWindowScaledLogGapReward:
    """
    基于“近期改进速度”自适应缩放的 log-gap 奖励。

    设计动机：
    - 如果一个问题最近几代进步很快，说明它当前更“容易”
    - 如果一个问题最近几代进步很慢，说明它当前更“困难”
    - 那么同样的 log-gap 改进量，在困难问题上应该得到更大的 reward

    做法：
    1. 先计算本代的基础 reward：
       base = log(gap_prev) - log(gap_curr)
    2. 再收集同一个任务最近 `window` 代的 `|base|`
    3. 用这些历史改进量的中位数作为 scale
    4. 最终 reward = base / scale

    这样：
    - 简单问题：近期改进大，scale 大，reward 会被压低
    - 困难问题：近期改进小，scale 小，reward 会被放大

    注意：
    - 这是一个启发式 reward，不是严格理论最优
    - 它适合“最优值未知，但想根据近期收敛速度估计问题难度”的场景
    - 目前先提供实现，默认不启用
    """

    def __init__(
        self,
        window: int = 10,
        maximize: bool = False,
        eps: float = 1e-30,
        min_scale: float = 1e-3,
        warmup_scale: float = 1.0,
        clip: float | None = None,
    ):
        self.window = int(window)
        self.maximize = maximize
        self.eps = float(eps)
        self.min_scale = float(min_scale)
        self.warmup_scale = float(warmup_scale)
        self.clip = None if clip is None else float(clip)
        self._history = defaultdict(lambda: deque(maxlen=self.window))

    def __call__(self, previous_state: Dict[str, Any], current_state: Dict[str, Any], update: Any = None) -> float:
        task_key = _extract_task_key(current_state)
        history = self._history[task_key]

        prev_gap = _safe_gap(_extract_best_value(previous_state), optimum=0.0, eps=self.eps)
        curr_gap = _safe_gap(_extract_best_value(current_state), optimum=0.0, eps=self.eps)

        if self.maximize:
            base_reward = np.log(curr_gap) - np.log(prev_gap)
        else:
            base_reward = np.log(prev_gap) - np.log(curr_gap)

        if len(history) == 0:
            scale = self.warmup_scale
        else:
            scale = max(float(np.median(np.asarray(history, dtype=np.float64))), self.min_scale)

        reward = float(base_reward) / float(scale)
        if self.clip is not None:
            reward = float(np.clip(reward, -self.clip, self.clip))

        # 当前这代的基础改进会被记到该任务自己的窗口里，供后续几代估计难度。
        history.append(abs(float(base_reward)))
        return float(reward)


@dataclass
class EnvTransition:
    observation: Dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class StepwiseEAEnv(EnvBase):
    """
    通用 step-wise EA 环境。

    这个环境不绑定某一种算法，只要求算法对象遵循如下接口：
    - `initialize(problem, update=None)` 或 `reset(problem, update=None)`
    - `advance(update=None)` 或 `run_generation(update=None) + get_state()`
    - `is_done()`
    - `get_state()`

    对于 `StepNeuroEA`、`AdaptiveGA` 这种基于 `GenerationAlgorithm` 的算法，
    这套接口是直接兼容的。

    关于 `gymnasium`：
    - 如果环境里装了 `gymnasium`，这个类就是标准 Gym 环境
    - 如果没装，也仍然可以跑当前仓库自带的训练代码
    - 但没有 `gymnasium` 时，不建议直接拿去接依赖 Gym 标准生态的外部库

    约定：
    - 一个环境实例内部，动作空间和观测空间默认视为固定
    - 因此，如果你在 episode 中途把算法结构改到“参数维度也变化了”，
      那么当前 env 的 action_space / observation_space 就可能不再匹配
    - 对这种场景，更稳妥的做法是重新创建一个 env 实例
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        algorithm_factory: Callable[..., Any],
        problem_factory: Callable[..., Any],
        initial_update: Any = None,
        action_adapter: Optional[ParameterVectorActionAdapter] = None,
        observation_builder: Optional[PopulationObservationBuilder] = None,
        reward_builder: Optional[ImprovementReward] = None,
        task_sampler: Optional[Callable[..., Any]] = None,
        task_state_builder: Optional[Callable[..., Dict[str, Any]]] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.algorithm_factory = algorithm_factory
        self.problem_factory = problem_factory
        self.initial_update = initial_update
        self.action_adapter = action_adapter or ParameterVectorActionAdapter()
        self.observation_builder = observation_builder or PopulationObservationBuilder()
        self.reward_builder = reward_builder or ImprovementReward()
        self.task_sampler = task_sampler
        self.task_state_builder = task_state_builder
        self.base_seed = int(seed)

        self.algorithm = None
        self.problem = None
        self.current_task = None
        self.current_task_index = None
        self._state = None
        self._state_history = []
        self._episode_index = 0

        prototype_state = self._build_prototype_state()
        self.action_space = self.action_adapter.build_space(prototype_state)
        self.observation_space = self.observation_builder.build_space(
            prototype_state,
            history_states=[prototype_state],
        )

    def reset(self, *, seed=None, options=None):
        episode_seed = self.base_seed + self._episode_index if seed is None else int(seed)
        episode_index = self._episode_index
        self._episode_index += 1

        self.current_task, self.current_task_index = self._select_task(
            episode_seed,
            episode_index=episode_index,
            options=options,
        )

        self.algorithm = self._call_factory(self.algorithm_factory, episode_seed, self.current_task)
        self.problem = self._call_factory(self.problem_factory, episode_seed, self.current_task)

        initial_update = self._resolve_initial_update(self.algorithm, self.problem, options=options)
        self._state = self._augment_state(self._reset_algorithm(self.algorithm, self.problem, initial_update))
        self._state_history = [self._state]

        observation = self._build_observation()
        info = self._build_info(self._state, initial_update=initial_update, reward=0.0)
        return observation, info

    def step(self, action):
        if self.algorithm is None or self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self.algorithm.is_done():
            raise RuntimeError("The episode has already terminated. Call reset() to start a new episode.")

        previous_state = self._state
        update = self.action_adapter.to_update(action, previous_state)
        self._state = self._augment_state(self._advance_algorithm(self.algorithm, update))
        self._state_history.append(self._state)

        reward = float(self.reward_builder(previous_state, self._state, update))
        terminated = bool(self._state.get("finished", False))
        truncated = False

        observation = self._build_observation()
        info = self._build_info(self._state, update=update, reward=reward)
        return observation, reward, terminated, truncated, info

    def render(self):
        if self._state is None:
            print("Environment not initialized.")
            return
        print(
            f"generation={self._state.get('generation', 0)} "
            f"fe={self._state.get('fe', 0)} "
            f"best={_extract_best_value(self._state):.6e}"
        )

    def _build_prototype_state(self):
        task, task_index = self._select_task(self.base_seed, episode_index=0, options=None, prototype=True)
        self.current_task = task
        self.current_task_index = task_index
        algorithm = self._call_factory(self.algorithm_factory, self.base_seed, task)
        problem = self._call_factory(self.problem_factory, self.base_seed, task)
        initial_update = self._resolve_initial_update(algorithm, problem, options=None)
        return self._augment_state(self._reset_algorithm(algorithm, problem, initial_update))

    def _build_observation(self):
        return self.observation_builder(self._state, history_states=self._state_history)

    def _resolve_initial_update(self, algorithm, problem, options=None):
        if options and "initial_update" in options:
            value = options["initial_update"]
            return value(algorithm, problem) if callable(value) else value

        value = self.initial_update
        if callable(value):
            return value(algorithm, problem)
        return value

    def _reset_algorithm(self, algorithm, problem, initial_update):
        if hasattr(algorithm, "reset"):
            return algorithm.reset(problem, update=initial_update)
        algorithm.initialize(problem, update=initial_update)
        return algorithm.get_state()

    def _advance_algorithm(self, algorithm, update):
        if hasattr(algorithm, "advance"):
            return algorithm.advance(update)
        algorithm.run_generation(update)
        return algorithm.get_state()

    def _call_factory(self, factory, seed, task):
        signature = inspect.signature(factory)
        if len(signature.parameters) >= 2:
            return factory(seed, task)
        return factory(seed)

    def _select_task(self, episode_seed, episode_index=0, options=None, prototype=False):
        if self.task_sampler is None:
            return None, None

        task_selection = self.task_sampler(
            episode_seed,
            episode_index=episode_index,
            options=options,
            prototype=prototype,
        )
        if isinstance(task_selection, tuple) and len(task_selection) == 2:
            return task_selection
        return task_selection, None

    def _augment_state(self, state):
        if state is None:
            return state
        augmented = dict(state)

        if self.task_state_builder is not None and self.current_task is not None:
            task_state = self.task_state_builder(self.current_task, self.current_task_index)
            if task_state:
                augmented.update(task_state)

        return augmented

    def _build_info(self, state, update=None, initial_update=None, reward=0.0):
        return {
            "generation": int(state.get("generation", 0)),
            "fe": int(state.get("fe", 0)),
            "best_fitness": _extract_best_value(state),
            "reward": float(reward),
            "update": update,
            "initial_update": initial_update,
            "task": self.current_task,
            "task_index": self.current_task_index,
            "state": state,
        }
