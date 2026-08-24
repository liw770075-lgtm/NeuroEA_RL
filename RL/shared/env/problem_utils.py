"""问题任务与问题实例构造工具。

这个模块放的是“环境无关、算法无关”的通用能力：
- 把用户输入规整成 `ProblemTask`
- 负责真正构造 torch 版问题实例
- 负责多任务选择
- 提供问题发现工具，方便脚本自动遍历当前可用问题集
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import inspect
import pkgutil
from typing import Any, Iterable
import warnings

import numpy as np
import torch

from NeuroEA_GEA_torch import Problems
from NeuroEA_GEA_torch.Problems.PROBLEM import Problem
from NeuroEA_GEA_torch.Problems import get_problem_instance
from NeuroEA_GEA_torch.utils.rng import NumpyTorchRNG


@dataclass
class ProblemTask:
    """
    一个训练/评估任务的定义。

    `problem_config` 约定为：
    `(N, M, D, max_fe)`
    """

    problem_name: str
    problem_config: tuple[int, int, int, int]
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        # 训练配置落盘时，统一转成 JSON 友好的普通字典。
        return {
            "problem_name": self.problem_name,
            "problem_config": tuple(self.problem_config),
            "label": self.label,
            "metadata": dict(self.metadata),
        }


def resolve_device(device):
    """统一处理 env / problem / algorithm 里出现的 device 传参。"""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(dtype, device):
    """默认在 CPU 上优先用 float64，避免优化问题数值过早丢精度。"""
    if dtype is None:
        return torch.float32 if device.type != "cpu" else torch.float64
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


def normalize_problem_config(problem_config=None, default_problem_config=(100, 1, 10, 10000)):
    """把单任务配置规整成固定的 `(N, M, D, max_fe)` 四元组。"""
    if problem_config is None:
        return tuple(default_problem_config)
    return tuple(problem_config)


def make_problem_task(problem_name, problem_config=None, default_problem_config=(100, 1, 10, 10000), label=None, metadata=None):
    """把裸参数拼成 `ProblemTask`，供环境/脚本统一使用。"""
    return ProblemTask(
        problem_name=str(problem_name),
        problem_config=normalize_problem_config(problem_config, default_problem_config=default_problem_config),
        label=label,
        metadata={} if metadata is None else dict(metadata),
    )


def normalize_tasks(tasks=None, problem_name=None, problem_config=None, default_problem_config=(100, 1, 10, 10000)):
    """
    统一整理任务列表输入。

    支持三种写法：
    - 单任务：`problem_name + problem_config`
    - `ProblemTask`
    - `dict(problem_name=..., problem_config=..., label=..., metadata=...)`
    """
    if tasks is None:
        if problem_name is None:
            raise ValueError("Either `tasks` or `problem_name` must be provided.")
        return [make_problem_task(problem_name, problem_config, default_problem_config=default_problem_config)]

    normalized = []
    for task in tasks:
        if isinstance(task, ProblemTask):
            normalized.append(
                make_problem_task(
                    problem_name=task.problem_name,
                    problem_config=task.problem_config,
                    default_problem_config=default_problem_config,
                    label=task.label,
                    metadata=task.metadata,
                )
            )
            continue

        if isinstance(task, dict):
            normalized.append(
                make_problem_task(
                    problem_name=task["problem_name"],
                    problem_config=task.get("problem_config"),
                    default_problem_config=default_problem_config,
                    label=task.get("label"),
                    metadata=task.get("metadata"),
                )
            )
            continue

        raise TypeError("Each task must be a ProblemTask or dict.")

    if not normalized:
        raise ValueError("Tasks must not be empty.")
    return normalized


def build_problem_instance(
    task: ProblemTask,
    device="auto",
    dtype=None,
    seed=0,
):
    """
    根据一个 `ProblemTask` 真正构造出 torch 版问题实例。

    这部分是通用逻辑，因此放在 `problem_utils.py`，
    而不是塞进某一个具体 EA 的 env 文件里。
    """
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    rng = NumpyTorchRNG(seed=seed, device=device, dtype=dtype)
    return get_problem_instance(
        task.problem_name,
        *task.problem_config,
        device=device,
        dtype=dtype,
        rng=rng,
    )


def discover_problem_names(single_objective_only: bool = False, warn_on_failure: bool = True):
    """自动发现当前工程里可实例化的问题名。

    这里直接扫描 `NeuroEA_GEA_torch.Problems` 包下的类，而不是手写问题列表。
    这样后续只要在 Problems 目录里新增问题类，这里就能自动看到。
    """
    discovered = set()
    failures = []
    module_prefix = Problems.__name__ + "."

    for _, module_full_name, _ in pkgutil.walk_packages(Problems.__path__, module_prefix):
        if single_objective_only and ".Single_objective_optimization." not in module_full_name:
            continue

        try:
            module = importlib.import_module(module_full_name)
        except Exception as exc:
            failures.append((module_full_name, exc))
            continue

        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls is Problem or not issubclass(cls, Problem):
                continue
            if cls.__module__ != module.__name__:
                continue
            discovered.add(name)

    if failures and warn_on_failure:
        preview = ", ".join(
            f"{module_name} ({type(exc).__name__}: {exc})"
            for module_name, exc in failures[:5]
        )
        if len(failures) > 5:
            preview += f", ... 共 {len(failures)} 个失败模块"
        warnings.warn(
            "Some problem modules failed to import during discovery: "
            f"{preview}",
            RuntimeWarning,
        )

    return sorted(discovered)


def discover_single_objective_problem_names(warn_on_failure: bool = True):
    """返回当前 `Single_objective_optimization` 下全部可用问题名。"""
    return discover_problem_names(single_objective_only=True, warn_on_failure=warn_on_failure)


def select_task(tasks: Iterable[ProblemTask], episode_index: int, seed: int, mode="cycle", options=None, prototype=False):
    """
    选择当前 episode 要训练/评估哪个任务。

    支持三种来源：
    - 用户在 `reset(options=...)` 里强制指定
    - 按 `cycle` 轮换
    - 按 `random` 随机抽样
    """
    tasks = list(tasks)
    if not tasks:
        raise ValueError("Tasks must not be empty.")

    if options and "task" in options:
        task = options["task"]
        if isinstance(task, ProblemTask):
            return task, -1
        if isinstance(task, dict):
            return normalize_tasks(tasks=[task])[0], -1
        raise TypeError("`options['task']` must be a ProblemTask or dict.")

    if options and "task_index" in options:
        index = int(options["task_index"])
        return tasks[index], index

    if options and "task_name" in options:
        task_name = str(options["task_name"]).lower()
        for index, task in enumerate(tasks):
            if task.problem_name.lower() == task_name or (task.label and task.label.lower() == task_name):
                return task, index
        raise ValueError(f"Unable to find task with name `{options['task_name']}`.")

    if prototype:
        return tasks[0], 0

    if mode == "cycle":
        index = int(episode_index) % len(tasks)
        return tasks[index], index
    if mode == "random":
        rng = np.random.RandomState(int(seed))
        index = int(rng.randint(len(tasks)))
        return tasks[index], index

    raise ValueError(f"Unsupported task selection mode: {mode}.")


def build_task_context(task: ProblemTask, task_index: int, tasks: list[ProblemTask]):
    """
    构造固定长度任务上下文向量。

    向量由两部分组成：
    - 当前任务在任务列表中的 one-hot 编码
    - 标准化后的 `(N, M, D, max_fe)`
    """
    one_hot = np.zeros(len(tasks), dtype=np.float32)
    if 0 <= task_index < len(tasks):
        one_hot[task_index] = 1.0

    configs = np.asarray([list(item.problem_config) for item in tasks], dtype=np.float32)
    config = np.asarray(task.problem_config, dtype=np.float32)
    scales = np.maximum(np.max(configs, axis=0), 1.0)
    normalized_config = config / scales
    return np.concatenate([one_hot, normalized_config], axis=0).astype(np.float32)
