"""Observation builder with ELA landscape features.

This module keeps ELA integration separate from the default
PopulationObservationBuilder so training scripts can switch it on without
changing the core step-wise EA environment.
"""

from __future__ import annotations

import numpy as np

from RL.shared.env.stepwise_ea_env import PopulationObservationBuilder, _to_numpy


class ELAObservationBuilder(PopulationObservationBuilder):
    """Append dynamic ELA features computed from the current population.

    The ELA vector is computed from ``population_dec`` and ``population_obj`` at
    each environment step, so it describes the local landscape currently seen by
    the algorithm. By default, multi-objective populations use the first
    objective as the scalar landscape value expected by ``compute_ela_features``.
    """

    def __init__(
        self,
        include_population: bool = True,
        normalize_control: bool = True,
        include_task_context: bool = True,
        feature_dim: int = 9,
        feature_scale: float | None = 100.0,
        objective_index: int = 0,
    ):
        super().__init__(
            include_population=include_population,
            normalize_control=normalize_control,
        )
        self.feature_dim = int(feature_dim)
        self.feature_scale = None if feature_scale is None else float(feature_scale)
        self.objective_index = int(objective_index)
        self.include_task_context = bool(include_task_context)
        self._ela_import_error = None

    def __call__(self, state, history_states=None):
        observation = super().__call__(state, history_states=history_states)
        if not self.include_task_context:
            observation.pop("task", None)
        observation["ela"] = self._build_ela_vector(state)
        return observation

    def _build_ela_vector(self, state):
        population_dec = _to_numpy(state.get("population_dec"), dtype=np.float64)
        population_obj = _to_numpy(state.get("population_obj"), dtype=np.float64)

        if population_dec is None or population_obj is None:
            return self._fallback()

        x = np.asarray(population_dec, dtype=np.float64)
        y = self._select_objective(population_obj, expected_rows=x.shape[0])
        if x.ndim != 2 or y is None or y.shape[0] != x.shape[0]:
            return self._fallback()

        try:
            from RL.shared.ELA.ELA import compute_ela_features

            features = compute_ela_features(x, y)
        except Exception as exc:
            self._ela_import_error = exc
            return self._fallback()

        features = np.asarray(features, dtype=np.float32).reshape(-1)
        if features.shape[0] < self.feature_dim:
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[: features.shape[0]] = features
            features = padded
        else:
            features = features[: self.feature_dim]

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.feature_scale is not None and self.feature_scale > 0:
            features = features / self.feature_scale
        return features.astype(np.float32, copy=False)

    def _select_objective(self, population_obj, expected_rows):
        y = np.asarray(population_obj, dtype=np.float64)
        if y.ndim == 1:
            return y.reshape(-1)
        if y.ndim == 2 and y.shape[0] == expected_rows:
            index = min(max(self.objective_index, 0), y.shape[1] - 1)
            return y[:, index].reshape(-1)
        return None

    def _fallback(self):
        return np.zeros(self.feature_dim, dtype=np.float32)
