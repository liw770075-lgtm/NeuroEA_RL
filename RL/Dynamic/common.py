"""Shared components for stable dynamic NeuroEA experiments."""

from __future__ import annotations

import re

import numpy as np

from RL.shared.env.ela_observation import ELAObservationBuilder
from RL.shared.env.spaces import EnvBase


NORMALIZATION_VERSION = "stable_summary_v1"


def parse_problem_names(text):
    if not text:
        return []
    names = []
    for item in (part.strip() for part in str(text).split(",")):
        if not item:
            continue
        match = re.fullmatch(r"(.+_F)(\d+)-(?:(?:.+_F)?)(\d+)", item, re.IGNORECASE)
        if match:
            prefix, start_text, end_text = match.groups()
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            names.extend(f"{prefix}{index}" for index in range(start, end + step, step))
        else:
            names.append(item)
    return names


def problem_suffix(problem_name):
    match = re.search(r"F(\d+)$", problem_name, re.IGNORECASE)
    return f"F{match.group(1)}" if match else problem_name


def signed_log1p(values):
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values))


class StableELAObservationBuilder(ELAObservationBuilder):
    """ELA observation with deterministic, problem-scale-resistant summary features."""

    def __init__(self, *args, summary_clip=5.0, objective_log_scale=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_clip = float(summary_clip)
        self.objective_log_scale = float(objective_log_scale)

    def __call__(self, state, history_states=None):
        observation = super().__call__(state, history_states=history_states)
        if "summary" in observation:
            observation["summary"] = self._normalize_summary(
                observation["summary"], state, observation.get("control")
            )
        if "ela" in observation:
            observation["ela"] = np.clip(
                np.nan_to_num(observation["ela"], nan=0.0, posinf=0.0, neginf=0.0),
                -self.summary_clip,
                self.summary_clip,
            ).astype(np.float32)
        return observation

    def _normalize_summary(self, summary, state, control):
        values = np.asarray(summary, dtype=np.float64).reshape(-1).copy()
        task_config = np.asarray(state.get("task_config", []), dtype=np.float64).reshape(-1)
        population_size = float(task_config[0]) if task_config.size >= 1 else 1.0
        max_fe = float(task_config[-1]) if task_config.size >= 4 else max(float(values[1]), 1.0)
        population_size = max(population_size, 1.0)
        max_fe = max(max_fe, 1.0)
        max_generations = max(np.ceil(max_fe / population_size), 1.0)

        if values.size > 0:
            values[0] /= max_generations
        if values.size > 1:
            values[1] /= max_fe
        if values.size > 2:
            end = min(values.size, 7)
            values[2:end] = signed_log1p(values[2:end]) / self.objective_log_scale
        if values.size > 7:
            control_dim = max(np.asarray(control).size if control is not None else 1, 1)
            values[7] /= np.sqrt(float(control_dim))

        return np.clip(
            np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
            -self.summary_clip,
            self.summary_clip,
        ).astype(np.float32)


class FirstActionReuseEnv(EnvBase):
    """Treat one fixed parameter vector as a macro action for a complete EA run."""

    metadata = {"render_modes": []}

    def __init__(self, base_env):
        super().__init__()
        self.base_env = base_env
        self.tasks = base_env.tasks
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        self._done = False
        self._observation = None
        self._info = None

    def reset(self, *, seed=None, options=None):
        self._done = False
        self._observation, self._info = self.base_env.reset(seed=seed, options=options)
        info = dict(self._info)
        info.update(flow="first_action_reuse", reused_action=False, reuse_generations=0)
        return self._observation, info

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode is finished; call reset() before step().")
        fixed_action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        fixed_action = np.clip(
            np.nan_to_num(fixed_action, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0
        )
        total_reward = 0.0
        rewards = []
        generations = 0
        while True:
            observation, reward, terminated, truncated, info = self.base_env.step(fixed_action)
            rewards.append(float(reward))
            total_reward += float(reward)
            generations += 1
            if terminated or truncated:
                break
        self._done = True
        self._observation, self._info = observation, info
        info = dict(info)
        info.update(
            flow="first_action_reuse",
            reused_action=True,
            reuse_generations=generations,
            generation_rewards=rewards,
            normalized_update=fixed_action,
        )
        return observation, total_reward, bool(terminated), bool(truncated), info

    def render(self):
        return self.base_env.render()

