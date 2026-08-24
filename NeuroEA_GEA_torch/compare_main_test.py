import argparse
import time

import numpy as np
import torch

from NeuroEA_GEA.main_test import ParameterTraining as NumpyParameterTraining
from NeuroEA_GEA_torch.main_test import EXAMPLE_PARAMETERS, ParameterTraining as TorchParameterTraining

def build_example_case():
    return {"example": np.asarray(EXAMPLE_PARAMETERS, dtype=np.float64)}


def run_numpy(problem, params, seed, run_times):
    np.random.seed(seed)
    start = time.perf_counter()
    fitness = NumpyParameterTraining(problem, run_times=run_times, save_root="results").testing(params.tolist())
    return fitness, time.perf_counter() - start


def run_torch(problem, params, seed, device, dtype, run_times, parallel=False):
    start = time.perf_counter()
    fitness = TorchParameterTraining(
        problem,
        run_times=run_times,
        save_root="results",
        device=device,
        dtype=dtype,
        seed=seed,
        parallel=parallel,
    ).testing(params)
    return fitness, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="BBOB_F1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-times", type=int, default=30)
    args = parser.parse_args()

    problem = args.problem
    seed = args.seed
    run_times = args.run_times
    cases = build_example_case()

    print(f"compare problem={problem} seed={seed} run_times={run_times}")
    for name, params in cases.items():
        numpy_fit, numpy_time = run_numpy(problem, params, seed, run_times)
        torch_cpu_fit, torch_cpu_time = run_torch(
            problem,
            params,
            seed,
            device="cpu",
            dtype="float64",
            run_times=run_times,
            parallel=True,
        )

        print(
            f"{name}: numpy={numpy_fit:.16e} ({numpy_time:.3f}s), "
            f"torch_cpu_parallel={torch_cpu_fit:.16e} ({torch_cpu_time:.3f}s), "
            f"abs_diff={abs(numpy_fit - torch_cpu_fit):.3e}"
        )

        if torch.cuda.is_available():
            try:
                torch_gpu_fit, torch_gpu_time = run_torch(
                    problem,
                    params,
                    seed,
                    device="cuda",
                    dtype="float32",
                    run_times=run_times,
                    parallel=False,
                )
                print(
                    f"{name}: torch_cuda={torch_gpu_fit:.16e} "
                    f"({torch_gpu_time:.3f}s), abs_diff_vs_numpy={abs(numpy_fit - torch_gpu_fit):.3e}"
                )
            except Exception as exc:
                print(f"{name}: torch_cuda failed: {exc}")
        else:
            print(f"{name}: torch_cuda skipped (CUDA unavailable on this machine)")


if __name__ == "__main__":
    main()
    # conda run -n pyRL python -m NeuroEA_GEA_torch.compare_main_test --run-times 30 --seed 3
