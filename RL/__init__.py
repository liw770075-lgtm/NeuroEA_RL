"""RL 主包。

这里统一导出环境、reward、agent 和 trainer 的常用入口，
方便外部脚本直接 `from RL import ...` 使用。
"""

from RL.shared.env import (
    AdaptiveWindowScaledLogGapReward,
    ImprovementReward,
    KnownOptimumLogGapReward,
    LogGapReward,
    ProblemTask,
    RelativeImprovementReward,
    StepNeuroEAEnv,
    StepwiseEAEnv,
    build_problem_instance,
    discover_problem_names,
    discover_single_objective_problem_names,
    create_stepneuroea_env,
    normalize_tasks,
)
from RL.shared.method import BaseAgent, ReplayBuffer, SACAgent
from RL.shared.trainer import SACTrainer

__all__ = [
    "ProblemTask",
    "AdaptiveWindowScaledLogGapReward",
    "ImprovementReward",
    "KnownOptimumLogGapReward",
    "LogGapReward",
    "RelativeImprovementReward",
    "StepwiseEAEnv",
    "StepNeuroEAEnv",
    "create_stepneuroea_env",
    "build_problem_instance",
    "discover_problem_names",
    "discover_single_objective_problem_names",
    "normalize_tasks",
    "BaseAgent",
    "ReplayBuffer",
    "SACAgent",
    "SACTrainer",
]
