import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from NeuroEA_GEA_torch.Algorithms.NEUROEA.BLOCK import Block
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Crossover import Block_Crossover
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Exchange import Block_Exchange
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Mutation import Block_Mutation
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Population import Block_Population
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Selection import Block_Selection
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Tournament import Block_Tournament
from NeuroEA_GEA_torch.Algorithms.NEUROEA.NeuroEA import NeuroEA
from NeuroEA_GEA_torch.Algorithms.NEUROEA.StepNeuroEA import StepNeuroEA
from NeuroEA_GEA_torch.Problems import get_problem_instance
from NeuroEA_GEA_torch.main_test import EXAMPLE_PARAMETERS
from NeuroEA_GEA_torch.utils.rng import NumpyTorchRNG


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark NeuroEA and StepNeuroEA execution modes.")
    parser.add_argument("--problem", default="BBOB_F1")
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--objectives", type=int, default=1)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--max-fe", type=int, default=10000)
    parser.add_argument("--run-times", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument(
        "--parameter-source",
        default="example",
        choices=["example", "random"],
        help="example uses main_test.EXAMPLE_PARAMETERS; random uses each run's initialized block parameters.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "neuroEA",
            "stepneuroEA_solve",
            "stepneuroEA_manualSolve",
            "stepneuroEA_manualSolve+update",
        ],
        choices=[
            "neuroEA",
            "stepneuroEA_solve",
            "stepneuroEA_manualSolve",
            "stepneuroEA_manualSolve+update",
        ],
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
    problem = get_problem_instance(
        args.problem,
        N=args.population_size,
        M=args.objectives,
        D=args.dimension,
        max_fe=args.max_fe,
        device=device,
        dtype=dtype,
        rng=rng,
    )
    return problem, rng


def build_blocks_graph(args, rng, device, dtype):
    n = args.population_size
    blocks = [
        Block_Population(rng=rng, device=device, dtype=dtype),
        Block_Tournament(n_parents=2 * n, upper_k=10, rng=rng, device=device, dtype=dtype),
        Block_Tournament(n_parents=2 * n, upper_k=10, rng=rng, device=device, dtype=dtype),
        Block_Tournament(n_parents=2 * n, upper_k=10, rng=rng, device=device, dtype=dtype),
        Block_Exchange(n_parents=3, rng=rng, device=device, dtype=dtype),
        Block_Exchange(n_parents=3, rng=rng, device=device, dtype=dtype),
        Block_Exchange(n_parents=3, rng=rng, device=device, dtype=dtype),
        Block_Exchange(n_parents=3, rng=rng, device=device, dtype=dtype),
        Block_Crossover(n_parents=2, n_sets=3, rng=rng, device=device, dtype=dtype),
        Block_Mutation(n_sets=3, rng=rng, device=device, dtype=dtype),
        Block_Selection(n_solutions=n, rng=rng, device=device, dtype=dtype),
    ]

    num_blocks = len(blocks)
    graph = np.zeros((num_blocks, num_blocks))
    graph[0, [1, 2, 3, 10]] = 1.0
    graph[1:4, 4:8] = 0.25
    graph[4:8, 8] = 1.0
    graph[8, 9] = 1.0
    graph[9, 10] = 1.0
    graph[10, 0] = 1.0
    Block.Validity(blocks, graph)
    return blocks, graph


def resolve_parameter_vector(args, blocks):
    if args.parameter_source == "example":
        vector = np.asarray(EXAMPLE_PARAMETERS, dtype=np.float64)
    else:
        vector = Block.parameters(blocks).detach().cpu().numpy()

    expected = int(Block.parameters(blocks).numel())
    if vector.size != expected:
        raise ValueError(f"Expected {expected} block parameters, got {vector.size}.")
    return vector


def build_algorithm(args, seed, device, dtype, algorithm_kind):
    problem, rng = build_problem(args, seed, device, dtype)
    if problem.M != 1:
        raise ValueError("This benchmark script currently supports single-objective problems only.")

    blocks, graph = build_blocks_graph(args, rng, device, dtype)
    parameters = resolve_parameter_vector(args, blocks)
    if algorithm_kind == "step":
        algorithm = StepNeuroEA(blocks=blocks, graph=graph)
    else:
        algorithm = NeuroEA(blocks=blocks, graph=graph)
    return problem, algorithm, parameters


def run_algorithm(name, args, seed, device, dtype):
    algorithm_kind = "classic" if name == "neuroEA" else "step"
    problem, algorithm, parameters = build_algorithm(args, seed, device, dtype, algorithm_kind)

    maybe_sync(device)
    start = time.perf_counter()
    if name == "neuroEA":
        Block.ParameterSet(algorithm.parameter[0], parameters)
        algorithm.solve(problem)
    elif name == "stepneuroEA_solve":
        algorithm.set_block_parameters(parameters)
        algorithm.solve(problem)
    elif name == "stepneuroEA_manualSolve":
        algorithm.initialize(problem, update=parameters)
        while algorithm.run_generation():
            pass
    elif name == "stepneuroEA_manualSolve+update":
        algorithm.initialize(problem, update=parameters)
        while algorithm.run_generation(parameters):
            pass
    else:
        raise ValueError(f"Unsupported algorithm name: {name}")
    maybe_sync(device)
    wall_time = time.perf_counter() - start

    best_value = float(problem.cal_metric("Min_value", algorithm.result))
    generations = max(0, len(algorithm.result) - 1)
    return {
        "best": best_value,
        "wall_time": wall_time,
        "metric_runtime": float(algorithm.metric["runtime"]),
        "fe": int(problem.fe),
        "generations": int(generations),
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
        f"device={device} dtype={args.dtype} seed={args.seed} parameter_source={args.parameter_source}"
    )
    print(f"algorithms={', '.join(args.algorithms)}")

    all_results = {name: [] for name in args.algorithms}

    if args.warmup_runs > 0:
        print("\nWarmup")
        for name in args.algorithms:
            for warmup_idx in range(args.warmup_runs):
                warmup_seed = args.seed + 100000 + warmup_idx
                run_algorithm(name, args, warmup_seed, device, dtype)
            print(f"{name:>22}: completed {args.warmup_runs} warmup run(s)")

    for run_idx in range(args.run_times):
        run_seed = args.seed + run_idx 
        print(f"\nRun {run_idx + 1}/{args.run_times} seed={run_seed}")
        for name in args.algorithms:
            # run_seed = np.random.randint(0,200)
            result = run_algorithm(name, args, run_seed, device, dtype)
            all_results[name].append(result)
            print(
                f"{name:>22}: best={result['best']:.16e} wall={result['wall_time']:.4f}s "
                f"metric={result['metric_runtime']:.4f}s fe={result['fe']} gen={result['generations']}"
            )

    print("\nSummary")
    for name in args.algorithms:
        best_stats = summarize([item["best"] for item in all_results[name]])
        wall_stats = summarize([item["wall_time"] for item in all_results[name]])
        print(
            f"{name:>22}: best_mean={best_stats['mean']:.16e} "
            f"best_median={best_stats['median']:.16e} best_std={best_stats['std']:.3e} "
            f"wall_mean={wall_stats['mean']:.4f}s wall_median={wall_stats['median']:.4f}s"
        )


if __name__ == "__main__":
    main()
