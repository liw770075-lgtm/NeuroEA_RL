"""Run NeuroEA with uniformly random parameter vectors."""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import argparse
import csv
import re

import numpy as np

from RL.shared.env.problem_utils import ProblemTask
from RL.shared.env.stepneuroea_env import StepNeuroEAEnv
from RL.shared.env.stepwise_ea_env import LogGapReward

try:
    from NeuroEA_GEA_torch.utils.compat import tensor_to_numpy
except ModuleNotFoundError:
    def tensor_to_numpy(value, dtype=None):
        if hasattr(value, "detach"):
            array = np.asarray(value.detach().cpu().tolist())
        else:
            array = np.asarray(value)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array


def parse_args():
    parser = argparse.ArgumentParser(description="Random NeuroEA parameter baseline.")
    parser.add_argument("--problem-name", default="BBOB_F1")
    parser.add_argument(
        "--problem-names",
        default=None,
        help=(
            "Comma-separated problems or an inclusive range, for example "
            "BBOB_F3,BBOB_F5 or BBOB_F3-BBOB_F10."
        ),
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--max-fe", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--mode", choices=["fixed", "dynamic"], default="fixed")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Batch output directory. Each problem is saved to a separate CSV.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Batch filename prefix; defaults to RandomDynamic or RandomFixed.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Batch summary CSV path; defaults to <output-dir>/summary.csv.",
    )
    parser.add_argument("--parameters-output", default=None)
    return parser.parse_args()


def parse_problem_names(text):
    if not text:
        return []
    names = []
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        match = re.fullmatch(r"(.+_F)(\d+)-(?:(?:.+_F)?)(\d+)", item, flags=re.IGNORECASE)
        if match:
            prefix, start_text, end_text = match.groups()
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            names.extend(f"{prefix}{index}" for index in range(start, end + step, step))
        else:
            names.append(item)
    return names


def problem_suffix(problem_name):
    match = re.search(r"F(\d+)$", problem_name, flags=re.IGNORECASE)
    return f"F{match.group(1)}" if match else problem_name


def to_numpy(value):
    if hasattr(value, "detach"):
        value = tensor_to_numpy(value)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def map_parameters(action, lower, upper):
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return lower + (action + 1.0) * 0.5 * (upper - lower)


def record_parameters(rows, run, generation, action, parameters):
    for index, (normalized_value, parameter_value) in enumerate(zip(action, parameters)):
        rows.append(
            {
                "run": run,
                "generation": generation,
                "parameter_index": index,
                "normalized_value": float(normalized_value),
                "parameter_value": float(parameter_value),
            }
        )


def run_problem(args, problem_name, output, parameters_output):
    task = ProblemTask(
        problem_name,
        (args.population_size, 1, args.dimension, args.max_fe),
    )
    env = StepNeuroEAEnv(
        tasks=[task],
        task_mode="cycle",
        initialization="example",
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        reward_builder=LogGapReward(),
    )

    convergence_rows = []
    parameter_rows = []
    final_fitness = []

    for run_index in range(args.rounds):
        run = run_index + 1
        run_seed = args.seed + run_index
        rng = np.random.default_rng(run_seed)
        _, info = env.reset(seed=run_seed, options={"task_index": 0})
        state = info["state"]
        lower = to_numpy(state["lower_bounds"])
        upper = to_numpy(state["upper_bounds"])
        action_dim = lower.size
        fixed_action = rng.uniform(-1.0, 1.0, action_dim).astype(np.float32)

        convergence_rows.append(
            {
                "run": run,
                "generation": int(info["generation"]),
                "best_fitness": float(info["best_fitness"]),
            }
        )

        while True:
            if args.mode == "fixed":
                action = fixed_action
            else:
                action = rng.uniform(-1.0, 1.0, action_dim).astype(np.float32)

            parameters = map_parameters(action, lower, upper)
            _, _, terminated, truncated, info = env.step(action)
            generation = int(info["generation"])
            convergence_rows.append(
                {
                    "run": run,
                    "generation": generation,
                    "best_fitness": float(info["best_fitness"]),
                }
            )
            record_parameters(parameter_rows, run, generation, action, parameters)
            if terminated or truncated:
                break

        final_value = float(info["best_fitness"])
        final_fitness.append(final_value)
        print(
            f"run={run:02d} seed={run_seed} mode={args.mode} "
            f"generations={generation} best_fitness={final_value:.6e}"
        )

    output = Path(output)
    parameters_output = Path(parameters_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parameters_output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "generation", "best_fitness"])
        writer.writeheader()
        writer.writerows(convergence_rows)

    with parameters_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "generation",
                "parameter_index",
                "normalized_value",
                "parameter_value",
            ],
        )
        writer.writeheader()
        writer.writerows(parameter_rows)

    values = np.asarray(final_fitness, dtype=np.float64)
    print(
        f"fitness: mean={values.mean():.4e}, std={values.std():.2e}, "
        f"min={values.min():.6e}"
    )
    print(f"convergence saved: {output}")
    print(f"parameters saved: {parameters_output}")
    return {
        "problem_name": problem_name,
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "result": f"{values.mean():.4e}({values.std():.2e})",
        "output": str(output),
        "parameters_output": str(parameters_output),
    }


def main():
    args = parse_args()
    problem_names = parse_problem_names(args.problem_names) if args.problem_names else [args.problem_name]
    if not problem_names:
        raise ValueError("No problems were specified.")

    batch_mode = len(problem_names) > 1 or args.problem_names is not None
    if batch_mode and args.output:
        raise ValueError("Use --output-dir instead of --output when testing multiple problems.")
    if batch_mode and args.parameters_output:
        raise ValueError(
            "Do not use --parameters-output in batch mode; parameter filenames are generated automatically."
        )

    summaries = []
    for problem_name in problem_names:
        if batch_mode:
            output_dir = Path(args.output_dir or f"GA/test_result/Random/{args.mode.title()}")
            prefix = args.output_prefix or f"Random{args.mode.title()}"
            output = output_dir / f"{prefix}_{problem_suffix(problem_name)}.csv"
            parameters_output = output.with_name(f"{output.stem}_parameters.csv")
        else:
            output = Path(args.output) if args.output else Path(
                f"result/random_{args.mode}_{problem_name}_convergence.csv"
            )
            parameters_output = (
                Path(args.parameters_output)
                if args.parameters_output
                else output.with_name(f"{output.stem}_parameters.csv")
            )
        summaries.append(run_problem(args, problem_name, output, parameters_output))

    if batch_mode:
        summary_output = Path(args.summary_output) if args.summary_output else Path(
            args.output_dir or f"GA/test_result/Random/{args.mode.title()}"
        ) / "summary.csv"
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        with summary_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["problem_name", "mean", "std", "min", "result", "output"],
            )
            writer.writeheader()
            for summary in summaries:
                writer.writerow({key: summary[key] for key in writer.fieldnames})

        print("\nBatch summary:")
        for summary in summaries:
            print(f"{summary['problem_name']}: {summary['result']}")
        print(f"summary saved: {summary_output}")


if __name__ == "__main__":
    main()
