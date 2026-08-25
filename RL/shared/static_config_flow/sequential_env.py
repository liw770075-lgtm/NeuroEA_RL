"""Sequential static parameter-configuration environment.

This follows the stronger formulation used by the old static configuration
code: the agent configures one NeuroEA parameter per step. After all parameters
are chosen, StepNeuroEA runs once to completion with the fixed vector.
"""

from __future__ import annotations

import numpy as np

from RL.shared.env.problem_utils import (
    build_problem_instance,
    build_task_context,
    normalize_problem_config,
    normalize_tasks,
    resolve_device,
    resolve_dtype,
    select_task,
)
from RL.shared.env.spaces import EnvBase, spaces
from RL.shared.env.stepneuroea_env import build_stepneuroea_algorithm, resolve_stepneuroea_initial_update
from RL.shared.env.stepwise_ea_env import _extract_best_value, _safe_gap, _to_numpy
from NeuroEA_GEA_torch.main_test import DEFAULT_PROBLEM_PARA


class SequentialStaticStepNeuroEAConfigEnv(EnvBase):
    """Configure one normalized parameter per step, then run NeuroEA once."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        problem_name="SOP_F1",
        problem_config=None,
        tasks=None,
        task_mode="cycle",
        initialization="example",
        device="auto",
        dtype=None,
        seed=0,
        include_task_context=False,
        ela_feature_scale=1.0,
        ela_objective_index=0,
        reward_clip=None,
    ):
        super().__init__()
        default_problem_config = normalize_problem_config(
            problem_config,
            default_problem_config=DEFAULT_PROBLEM_PARA,
        )
        self.tasks = normalize_tasks(
            tasks=tasks,
            problem_name=problem_name,
            problem_config=default_problem_config,
            default_problem_config=DEFAULT_PROBLEM_PARA,
        )
        self.task_mode = task_mode
        self.initialization = initialization
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.base_seed = int(seed)
        self.include_task_context = bool(include_task_context)
        self.ela_feature_scale = None if ela_feature_scale is None else float(ela_feature_scale)
        self.ela_objective_index = int(ela_objective_index)
        self.reward_clip = None if reward_clip is None else float(reward_clip)

        self.algorithm = None
        self.problem = None
        self.current_task = None
        self.current_task_index = None
        self._state = None
        self._static_features = None
        self._initial_gap = None
        self._episode_index = 0
        self._param_index = 0
        self._params = None
        self._done = False

        prototype_state, prototype_features = self._build_prototype()
        self.lower_bounds = _to_numpy(prototype_state["lower_bounds"], dtype=np.float32).reshape(-1)
        self.upper_bounds = _to_numpy(prototype_state["upper_bounds"], dtype=np.float32).reshape(-1)
        self.num_params = int(self.lower_bounds.shape[0])
        self.default_params = self._normalize_params(
            _to_numpy(prototype_state["block_parameters"], dtype=np.float32).reshape(-1)
        )

        observation_dim = int(prototype_features.shape[0] + self.num_params)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        episode_seed = self.base_seed + self._episode_index if seed is None else int(seed)
        episode_index = self._episode_index
        self._episode_index += 1

        self.current_task, self.current_task_index = select_task(
            self.tasks,
            episode_index=episode_index,
            seed=episode_seed,
            mode=self.task_mode,
            options=options,
        )
        self.algorithm, self.problem, self._state = self._initialize_run(episode_seed, self.current_task)
        self._state = self._augment_state(self._state)
        self._static_features = self._build_static_features(self._state)
        self._initial_gap = _safe_gap(_extract_best_value(self._state), optimum=0.0)
        self._param_index = 0
        self._params = np.zeros(self.num_params, dtype=np.float32)
        self._done = False

        return self._build_observation(), self._build_info(self._state, reward=0.0, initial_update=True)

    def step(self, action):
        if self.algorithm is None or self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("The static sequential episode has already finished. Call reset().")

        action_value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self._params[self._param_index] = float(np.clip(action_value, -1.0, 1.0))
        self._param_index += 1

        if self._param_index < self.num_params:
            return self._build_observation(), 0.0, False, False, self._build_info(self._state, reward=0.0)

        real_params = self._denormalize_params(self._params)
        self.algorithm.apply_update(real_params)
        while not self.algorithm.is_done():
            self.algorithm.run_generation()

        final_state = self._augment_state(self.algorithm.get_state())
        final_gap = _safe_gap(_extract_best_value(final_state), optimum=0.0)
        reward = float(np.log(self._initial_gap) - np.log(final_gap))
        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self._state = final_state
        self._done = True
        info = self._build_info(final_state, reward=reward, update=real_params, normalized_update=self._params.copy())
        return self._build_observation(), reward, True, False, info

    def render(self):
        if self._state is None:
            print("Environment not initialized.")
            return
        print(
            f"task={self.current_task.problem_name if self.current_task else None} "
            f"param_index={self._param_index}/{self.num_params} "
            f"best={_extract_best_value(self._state):.6e}"
        )

    def _build_prototype(self):
        task, task_index = select_task(
            self.tasks,
            episode_index=0,
            seed=self.base_seed,
            mode=self.task_mode,
            options=None,
            prototype=True,
        )
        self.current_task = task
        self.current_task_index = task_index
        algorithm, problem, state = self._initialize_run(self.base_seed, task)
        state = self._augment_state(state)
        return state, self._build_static_features(state)

    def _initialize_run(self, seed, task):
        algorithm = build_stepneuroea_algorithm(task=task, device=self.device, dtype=self.dtype, seed=seed)
        problem = build_problem_instance(task=task, device=self.device, dtype=self.dtype, seed=seed)
        initial_update = resolve_stepneuroea_initial_update(self.initialization, algorithm, problem)
        state = algorithm.reset(problem, update=initial_update)
        return algorithm, problem, state

    def _augment_state(self, state):
        augmented = dict(state)
        if self.current_task is not None:
            augmented.update(
                {
                    "task_name": self.current_task.problem_name,
                    "task_config": np.asarray(self.current_task.problem_config, dtype=np.float32),
                    "task_index": int(-1 if self.current_task_index is None else self.current_task_index),
                }
            )
            if self.include_task_context:
                augmented["task_context"] = build_task_context(
                    self.current_task,
                    -1 if self.current_task_index is None else self.current_task_index,
                    self.tasks,
                )
        return augmented

    def _build_static_features(self, state):
        parts = [self._compute_static_ela(state)]
        if self.include_task_context and state.get("task_context") is not None:
            parts.append(_to_numpy(state.get("task_context"), dtype=np.float32).reshape(-1))
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def _build_observation(self):
        index_one_hot = np.zeros(self.num_params, dtype=np.float32)
        if self._param_index < self.num_params:
            index_one_hot[self._param_index] = 1.0
        return np.concatenate([self._static_features, index_one_hot], axis=0).astype(np.float32, copy=False)

    def _compute_static_ela(self, state):
        population_dec = _to_numpy(state.get("population_dec"), dtype=np.float64)
        population_obj = _to_numpy(state.get("population_obj"), dtype=np.float64)
        if population_dec is None or population_obj is None:
            raise ValueError("Static ELA extraction requires population_dec and population_obj.")

        x = np.asarray(population_dec, dtype=np.float64)
        y = self._select_objective(population_obj, expected_rows=x.shape[0])
        if x.ndim != 2 or y is None or y.shape[0] != x.shape[0]:
            raise ValueError(
                f"Invalid static ELA population shapes: X={x.shape}, "
                f"Y={None if y is None else y.shape}."
            )

        from RL.shared.ELA.ELA import ELA_FEATURE_DIM, compute_ela_features

        features = np.asarray(compute_ela_features(x, y), dtype=np.float32).reshape(-1)
        if features.shape != (ELA_FEATURE_DIM,):
            raise ValueError(
                f"Expected {ELA_FEATURE_DIM} ELA features, received {features.shape[0]}."
            )
        if self.ela_feature_scale is not None and self.ela_feature_scale > 0:
            features = features / self.ela_feature_scale
        return features.astype(np.float32, copy=False)

    def _select_objective(self, population_obj, expected_rows):
        y = np.asarray(population_obj, dtype=np.float64)
        if y.ndim == 1:
            return y.reshape(-1)
        if y.ndim == 2 and y.shape[0] == expected_rows:
            index = min(max(self.ela_objective_index, 0), y.shape[1] - 1)
            return y[:, index].reshape(-1)
        return None

    def _normalize_params(self, params):
        denom = np.where(np.abs(self.upper_bounds - self.lower_bounds) < 1e-12, 1.0, self.upper_bounds - self.lower_bounds)
        return (2.0 * (params - self.lower_bounds) / denom - 1.0).astype(np.float32)

    def _denormalize_params(self, params):
        params = np.asarray(params, dtype=np.float32).reshape(-1)
        params = np.clip(params, -1.0, 1.0)
        return self.lower_bounds + (params + 1.0) * 0.5 * (self.upper_bounds - self.lower_bounds)

    def _build_info(self, state, reward=0.0, update=None, normalized_update=None, initial_update=False):
        return {
            "generation": int(state.get("generation", 0)),
            "fe": int(state.get("fe", 0)),
            "best_fitness": _extract_best_value(state),
            "reward": float(reward),
            "update": update,
            "normalized_update": normalized_update,
            "initial_update": initial_update,
            "task": self.current_task,
            "task_index": self.current_task_index,
            "state": state,
            "default_params": self.default_params.copy(),
            "configured_params": None if self._params is None else self._params.copy(),
        }
