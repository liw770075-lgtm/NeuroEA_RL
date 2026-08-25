"""Paper-aligned exploratory landscape analysis (ELA) features.

The feature names and order follow Table 4 of DesignX and are computed with
the standard implementations provided by pflacco. Objective values are
min-max normalized before extraction, matching the public DesignX pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable
import warnings

import numpy as np
import torch
from pflacco.classical_ela_features import (
    calculate_ela_distribution,
    calculate_ela_meta,
    calculate_information_content,
    calculate_nbc,
)


ELA_FEATURE_NAMES = (
    "ela_meta.lin_simple.intercept",
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.lin_w_interact.adj_r2",
    "ic.m0",
    "ic.h_max",
    "ic.eps_ratio",
    "nbc.nn_nb.mean_ratio",
    "nbc.dist_ratio.coeff_var",
    "ela_distr.number_of_peaks",
)
ELA_FEATURE_DIM = len(ELA_FEATURE_NAMES)
DEFAULT_ELA_SEED = 42
ELA_IMPLEMENTATION_VERSION = "designx_table4_pflacco_1.2.2_v1"


def _validate_samples(X, Y):
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"X must be a two-dimensional array; received shape {x.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and Y must contain the same number of samples; "
            f"received {x.shape[0]} and {y.shape[0]}."
        )
    if x.shape[0] < 21:
        raise ValueError(
            "At least 21 samples are required by the default "
            "information-content neighborhood."
        )
    if x.shape[1] == 0:
        raise ValueError("X must contain at least one decision variable.")
    interaction_predictors = x.shape[1] + x.shape[1] * (x.shape[1] - 1) // 2
    if x.shape[0] <= interaction_predictors + 1:
        warnings.warn(
            "The linear interaction model is underdetermined: "
            f"{x.shape[0]} samples for {interaction_predictors} predictors. "
            "Its adjusted R-squared is not statistically well-defined; "
            "increase the ELA sample count or document this limitation.",
            RuntimeWarning,
            stacklevel=3,
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("X and Y must contain only finite values.")
    y_span = float(np.ptp(y))
    if y_span <= np.finfo(np.float64).eps:
        raise ValueError("ELA features are undefined when all objective values are equal.")
    return x, (y - float(np.min(y))) / y_span


def _sanitize_feature(value):
    if value is None:
        return 0.0
    scalar = float(np.asarray(value).reshape(()))
    if np.isnan(scalar):
        return 0.0
    if np.isinf(scalar):
        return 1.0
    return scalar


def compute_ela_feature_dict(X, Y, seed=DEFAULT_ELA_SEED):
    """Return the nine paper features as an insertion-ordered dictionary."""

    x, y = _validate_samples(X, Y)
    numpy_state = np.random.get_state()
    if seed is not None:
        np.random.seed(int(seed))
    try:
        feature_sets = (
            calculate_ela_meta(x, y),
            calculate_information_content(x, y, seed=seed),
            calculate_nbc(x, y, dist_tie_breaker="sample", minimize=True),
            calculate_ela_distribution(x, y),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Standard pflacco ELA extraction failed for X shape {x.shape}."
        ) from exc
    finally:
        np.random.set_state(numpy_state)

    available = {}
    for feature_set in feature_sets:
        available.update(feature_set)
    missing = [name for name in ELA_FEATURE_NAMES if name not in available]
    if missing:
        raise RuntimeError(f"pflacco did not return required ELA features: {missing}.")
    return {name: _sanitize_feature(available[name]) for name in ELA_FEATURE_NAMES}


def compute_ela_features(X, Y, seed=DEFAULT_ELA_SEED):
    """Return the paper-aligned ELA vector in ELA_FEATURE_NAMES order."""

    return np.asarray(
        list(compute_ela_feature_dict(X, Y, seed=seed).values()),
        dtype=np.float64,
    )


def get_ela_from_neuroea(
    problem_name,
    D=10,
    ela_sample=100,
    seed=DEFAULT_ELA_SEED,
):
    """Sample a torch NeuroEA problem and return its paper-aligned ELA vector."""

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from NeuroEA_GEA_torch.Problems import get_problem_instance
    from NeuroEA_GEA_torch.utils.rng import NumpyTorchRNG

    dimension = int(D)
    sample_size = int(ela_sample) * dimension
    if dimension <= 0 or sample_size < 21:
        raise ValueError("D must be positive and ela_sample * D must be at least 21.")
    rng = NumpyTorchRNG(seed=int(seed), device="cpu", dtype=torch.float64)
    problem = get_problem_instance(
        problem_name,
        N=sample_size,
        M=1,
        D=dimension,
        max_fe=max(10_000, sample_size),
        device="cpu",
        dtype=torch.float64,
        rng=rng,
    )
    population = problem.initialization()
    x = population.dec.detach().cpu().numpy()
    y = population.obj.detach().cpu().numpy().reshape(-1)
    return compute_ela_features(x, y, seed=seed)


def get_ela_from_neuroea_batch(
    problem_names: Iterable[str],
    D=10,
    ela_sample=100,
    seed=DEFAULT_ELA_SEED,
):
    """Return one ELA vector per problem; failures are reported, not hidden."""

    vectors = [
        get_ela_from_neuroea(name, D=D, ela_sample=ela_sample, seed=seed)
        for name in problem_names
    ]
    if not vectors:
        return np.empty((0, ELA_FEATURE_DIM), dtype=np.float64)
    return np.vstack(vectors)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-names", default="BBOB_F1,BBOB_F2")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument(
        "--sample-factor",
        type=int,
        default=100,
        help="Number of sampled solutions per decision variable.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_ELA_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    names = [item.strip() for item in args.problem_names.split(",") if item.strip()]
    vectors = get_ela_from_neuroea_batch(
        names,
        D=args.dimension,
        ela_sample=args.sample_factor,
        seed=args.seed,
    )
    payload = {
        "feature_names": list(ELA_FEATURE_NAMES),
        "problems": {
            name: vector.tolist()
            for name, vector in zip(names, vectors)
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
