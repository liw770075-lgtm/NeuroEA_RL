import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from NeuroEA_GEA_torch.Algorithms.GA.AdaptiveGA import AdaptiveGA
from NeuroEA_GEA_torch.Algorithms.GA.GA import GA
from NeuroEA_GEA_torch.Problems import get_problem_instance
from NeuroEA_GEA_torch.utils.rng import NumpyTorchRNG


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark GA and AdaptiveGA on a single-objective problem.")
    parser.add_argument("--problem", default="BBOB_F1")
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--objectives", type=int, default=1)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--max-fe", type=int, default=10000)
    parser.add_argument("--run-times", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=141)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--pro-c", type=float, default=1.0)
    parser.add_argument("--dis-c", type=float, default=20.0)
    parser.add_argument("--pro-m", type=float, default=1.0)
    parser.add_argument("--dis-m", type=float, default=20.0)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["ga", "adaptive_solve", "adaptive_advance"],
        choices=["ga", "adaptive_solve", "adaptive_advance", "adaptive_step"],
    )
    return parser.parse_args()


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")
    return torch.device(device_name)


def resolve_dtype(dtype_name):
    return getattr(torch, dtype_name)


def maybe_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_problem(args, seed, device, dtype):
    rng = NumpyTorchRNG(seed=seed, device=device, dtype=dtype)
    return get_problem_instance(
        args.problem,
        N=args.population_size,
        M=args.objectives,
        D=args.dimension,
        max_fe=args.max_fe,
        device=device,
        dtype=dtype,
        rng=rng,
    )


def build_algorithm(name, args):
    parameter = [args.pro_c, args.dis_c, args.pro_m, args.dis_m]
    if name == "ga":
        return GA(parameter=parameter)
    if name in {"adaptive_solve", "adaptive_advance", "adaptive_step"}:
        return AdaptiveGA(parameter=parameter)
    raise ValueError(f"Unsupported algorithm name: {name}")


def run_algorithm(name, args, seed, device, dtype):
    problem = build_problem(args, seed, device, dtype)
    if problem.M != 1:
        raise ValueError("This benchmark script currently supports single-objective problems only.")

    algorithm = build_algorithm(name, args)

    maybe_sync(device)
    start = time.perf_counter()
    if name in {"adaptive_advance", "adaptive_step"}:
        algorithm.initialize(problem)
        while algorithm.run_generation():
            pass
    else:
        algorithm.solve(problem)
    maybe_sync(device)
    wall_time = time.perf_counter() - start

    best_value = float(problem.cal_metric("Min_value", algorithm.result))
    return {
        "best": best_value,
        "wall_time": wall_time,
        "metric_runtime": float(algorithm.metric["runtime"]),
        "fe": int(problem.fe),
        "saved_points": len(algorithm.result),
    }


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    print("Benchmark configuration")
    print(
        f"problem={args.problem} N={args.population_size} M={args.objectives} "
        f"D={args.dimension} max_fe={args.max_fe} run_times={args.run_times} warmup_runs={args.warmup_runs}"
    )
    print(
        f"device={device} dtype={args.dtype} seed={args.seed} "
        f"operator=(proC={args.pro_c}, disC={args.dis_c}, proM={args.pro_m}, disM={args.dis_m})"
    )
    print(f"algorithms={', '.join(args.algorithms)}")

    all_results = {name: [] for name in args.algorithms}

    if args.warmup_runs > 0:
        print("\nWarmup")
        for name in args.algorithms:
            for warmup_idx in range(args.warmup_runs):
                warmup_seed = args.seed + 100000 + warmup_idx
                run_algorithm(name, args, warmup_seed, device, dtype)
            print(f"{name:>14}: completed {args.warmup_runs} warmup run(s)")

    for run_idx in range(args.run_times):
        run_seed = args.seed + run_idx
        print(f"\nRun {run_idx + 1}/{args.run_times} seed={run_seed}")
        for name in args.algorithms:
            result = run_algorithm(name, args, run_seed, device, dtype)
            run_seed = run_seed + 1 
            all_results[name].append(result)
            print(
                f"{name:>14}: best={result['best']:.16e} "
                f"wall={result['wall_time']:.4f}s metric={result['metric_runtime']:.4f}s "
                f"fe={result['fe']}"
            )

    print("\nSummary")
    for name in args.algorithms:
        best_stats = summarize([item["best"] for item in all_results[name]])
        wall_stats = summarize([item["wall_time"] for item in all_results[name]])
        print(
            f"{name:>14}: "
            f"best_mean={best_stats['mean']:.16e} best_median={best_stats['median']:.16e} "
            f"best_std={best_stats['std']:.3e} wall_mean={wall_stats['mean']:.4f}s "
            f"wall_median={wall_stats['median']:.4f}s"
        )


if __name__ == "__main__":
    main()


#     conda run -n pyRL python -m NeuroEA_GEA_torch.Algorithms.GA.benchmark_ga \
#   --problem BBOB_F1 \
#   --population-size 100 \
#   --dimension 10 \
#   --max-fe 10000 \
#   --run-times 30 \
#   --algorithms ga adaptive_step



# conda run -n pyRL python -m NeuroEA_GEA_torch.Algorithms.GA.benchmark_ga \
#   --problem BBOB_F1 \
#   --population-size 100 \
#   --dimension 10 \
#   --max-fe 10000 \
#   --run-times 30 \
#   --warmup-runs 1 \
#   --seed 0 \
#   --device cpu \
#   --dtype float64
