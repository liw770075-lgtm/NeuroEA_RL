"""Compatibility entry for the action-history static environment."""

from RL.Static.action_history_env import ActionHistoryStaticConfigEnv

StaticStepNeuroEAConfigEnv = ActionHistoryStaticConfigEnv

__all__ = ["ActionHistoryStaticConfigEnv", "StaticStepNeuroEAConfigEnv"]

