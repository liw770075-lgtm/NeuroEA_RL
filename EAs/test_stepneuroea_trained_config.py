"""Test a fixed NeuroEA configuration learned by GA or CMA-ES."""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import argparse
import csv
import json

import numpy as np
import torch

from RL.shared.env.problem_utils import ProblemTask
from RL.shared.env.stepneuroea_env import StepNeuroEAEnv
from RL.shared.env.stepwise_ea_env import LogGapReward, PopulationObservationBuilder

try:
    from NeuroEA_GEA_torch.utils.compat import tensor_to_numpy
except ModuleNotFoundError:
    def tensor_to_numpy(value, dtype=None):
        if torch.is_tensor(value):
            array = np.asarray(value.detach().cpu().tolist())
        else:
            array = np.asarray(value)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array


def parse_args():
    parser = argparse.ArgumentParser(description="Test fixed NeuroEA parameters learned by GA or CMA-ES.")
    parser.add_argument(
        "--artifact",
        required=True,
        help="Training directory, best_normalized_action.npy, best_parameters.npy, or best_config.json.",
    )
    parser.add_argument("--problem-name", required=True, help="Test problem, for example BBOB_F1.")
    parser.add_argument("--output-dir", required=True, help="Directory used to save test_results.csv and test_config.json.")
    parser.add_argument("--runs", type=int, default=10, help="Number of independent test runs.")
    parser.add_argument("--dimension", type=int, default=10, help="Decision dimension D.")
    parser.add_argument("--population-size", type=int, default=100, help="NeuroEA population size N.")
    parser.add_argument("--max-fe", type=int, default=10000, help="Maximum function evaluations per run.")
    parser.add_argument("--seed", type=int, default=10000, help="Seed of the first test run.")
    parser.add_argument("--device", default="cpu", help="Environment device: cpu / cuda / auto.")
    parser.add_argument("--dtype", default="float64", help="Environment dtype.")
    parser.add_argument(
        "--artifact-type",
        choices=["auto", "normalized", "parameters"],
        default="auto",
        help="Interpret a .npy file as normalized actions or physical NeuroEA parameters.",
    )
    return parser.parse_args()


def make_env(args):
    task = ProblemTask(
        args.problem_name,
        (args.population_size, 2, args.dimension, args.max_fe),
    )
    return StepNeuroEAEnv(
        tasks=[task],
        task_mode="cycle",
        initialization="example",
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        observation_builder=PopulationObservationBuilder(include_population=False),
        reward_builder=LogGapReward(),
    )


def parameter_vector_to_normalized(parameter_vector, lower_bounds, upper_bounds):
    parameter_vector = np.asarray(parameter_vector, dtype=np.float64).reshape(-1)
    lower_bounds = np.asarray(lower_bounds, dtype=np.float64).reshape(-1)
    upper_bounds = np.asarray(upper_bounds, dtype=np.float64).reshape(-1)
    denominator = np.where(np.abs(upper_bounds - lower_bounds) < 1e-12, 1.0, upper_bounds - lower_bounds)
    return np.clip(2.0 * (parameter_vector - lower_bounds) / denominator - 1.0, -1.0, 1.0)


def resolve_artifact(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact does not exist: {path}")
    if path.is_file():
        return path

    candidates = [
        path / "best_normalized_action.npy",
        path / "best_parameters.npy",
        path / "best_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No completed training artifact found in {path}. Expected one of: "
        "best_normalized_action.npy, best_parameters.npy, best_config.json."
    )


def load_action(artifact_path, artifact_type, lower_bounds, upper_bounds):
    artifact_path = resolve_artifact(artifact_path)
    suffix = artifact_path.suffix.lower()

    if suffix == ".json":
        with artifact_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if "normalized_action" in data:
            action = np.asarray(data["normalized_action"], dtype=np.float64)
            resolved_type = "normalized"
        elif "parameters" in data:
            action = parameter_vector_to_normalized(data["parameters"], lower_bounds, upper_bounds)
            resolved_type = "parameters"
        else:
            raise ValueError(f"JSON artifact has neither normalized_action nor parameters: {artifact_path}")
    elif suffix == ".npy":
        values = np.asarray(np.load(artifact_path), dtype=np.float64).reshape(-1)
        resolved_type = artifact_type
        if resolved_type == "auto":
            filename = artifact_path.name.lower()
            if "normalized" in filename or "action" in filename:
                resolved_type = "normalized"
            elif "parameter" in filename:
                resolved_type = "parameters"
            else:
                raise ValueError("Cannot infer .npy type; pass --artifact-type normalized or parameters.")
        action = values if resolved_type == "normalized" else parameter_vector_to_normalized(values, lower_bounds, upper_bounds)
    else:
        raise ValueError(f"Unsupported artifact format: {artifact_path.suffix}")

    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.size != lower_bounds.size:
        raise ValueError(f"Artifact has {action.size} values, but this NeuroEA environment expects {lower_bounds.size}.")
    if not np.all(np.isfinite(action)):
        raise ValueError("Artifact contains NaN or infinite values.")
    return np.clip(action, -1.0, 1.0), artifact_path, resolved_type


def get_action_bounds(env, seed):
    _, info = env.reset(seed=seed, options={"task_index": 0})
    state = info["state"]
    lower_bounds = tensor_to_numpy(state["lower_bounds"], dtype=np.float64).reshape(-1)
    upper_bounds = tensor_to_numpy(state["upper_bounds"], dtype=np.float64).reshape(-1)
    return lower_bounds, upper_bounds


def run_test(env, action, runs, base_seed):
    rows = []
    for run_index in range(1, runs + 1):
        run_seed = base_seed + run_index - 1
        _, info = env.reset(seed=run_seed, options={"task_index": 0})
        rows.append((run_index, int(info["generation"]), float(info["best_fitness"])))

        while True:
            _, _, terminated, truncated, info = env.step(action)
            rows.append((run_index, int(info["generation"]), float(info["best_fitness"])))
            if terminated or truncated:
                break

        print(
            f"run={run_index:02d} seed={run_seed} generations={info['generation']} "
            f"fe={info['fe']} best={info['best_fitness']:.6e}"
        )
    return rows


def summarize_final_fitness(rows):
    final_by_run = {}
    for run, generation, best_fitness in rows:
        previous = final_by_run.get(run)
        if previous is None or generation > previous[0]:
            final_by_run[run] = (generation, best_fitness)

    final_values = np.asarray(
        [final_by_run[run][1] for run in sorted(final_by_run)],
        dtype=np.float64,
    )
    return {
        "runs": int(final_values.size),
        "mean": float(np.mean(final_values)),
        "std": float(np.std(final_values)),
    }


def save_results(output_dir, rows, config, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "test_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "generation", "best_fitness"])
        writer.writerows(rows)

    with (output_dir / "test_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    with (output_dir / "test_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return csv_path


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args)
    lower_bounds, upper_bounds = get_action_bounds(env, args.seed)
    action, artifact_path, artifact_type = load_action(
        args.artifact,
        args.artifact_type,
        lower_bounds,
        upper_bounds,
    )
    rows = run_test(env, action, args.runs, args.seed)
    summary = summarize_final_fitness(rows)
    csv_path = save_results(
        args.output_dir,
        rows,
        {
            "artifact": str(artifact_path.resolve()),
            "artifact_type": artifact_type,
            "problem_name": args.problem_name,
            "runs": args.runs,
            "dimension": args.dimension,
            "population_size": args.population_size,
            "max_fe": args.max_fe,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "normalized_action": action.tolist(),
        },
        summary,
    )
    print(f"Saved {len(rows)} rows to {csv_path}")
    print(f"Final fitness: mean={summary['mean']:.5e}, std={summary['std']:.3e}")


if __name__ == "__main__":
    main()
