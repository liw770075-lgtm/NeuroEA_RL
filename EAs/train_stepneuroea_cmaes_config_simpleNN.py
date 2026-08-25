"""Optimize fixed StepNeuroEA block parameters with CMA-ES.

This script mirrors ``train_stepneuroea_ga_config_simpleNN.py`` but replaces
the outer genetic algorithm with a compact bound-handled CMA-ES optimizer.
Each candidate is a normalized parameter vector in [-1, 1]^d. During one
StepNeuroEA episode, that vector is applied at every generation, so CMA-ES
learns a static NeuroEA configuration.
"""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    current_file = Path(__file__).resolve()
    project_root = next(
        (
            parent
            for parent in current_file.parents
            if (parent / "RL").is_dir() and (parent / "NeuroEA_GEA_torch").is_dir()
        ),
        current_file.parents[1],
    )
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import argparse
import csv
from dataclasses import dataclass
import json
import time

import numpy as np
import torch

from RL.shared.env.ela_observation import ELAObservationBuilder
from RL.shared.env.problem_utils import ProblemTask
from RL.shared.env.stepneuroea_env import StepNeuroEAEnv
from RL.shared.env.stepwise_ea_env import LogGapReward, PopulationObservationBuilder
from RL.shared.examples.train_stepneuroea_sac_multitask_simpleNN import parse_csv_strings, save_json

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


@dataclass
class CandidateResult:
    fitness: float
    reward_mean: float
    best_mean: float
    best_std: float
    per_task: list


@dataclass
class GenerationRecord:
    generation: int
    elapsed_sec: float
    best_fitness: float
    global_best_fitness: float
    mean_fitness: float
    median_fitness: float
    std_fitness: float
    worst_fitness: float
    best_reward: float
    mean_reward: float
    best_index: int
    sigma: float


class SimpleCMAES:
    """Minimal CMA-ES implementation with candidate clipping to [-1, 1]."""

    def __init__(self, mean, sigma, population_size, seed=0, bounds=(-1.0, 1.0)):
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1).copy()
        self.dim = int(self.mean.shape[0])
        self.sigma = float(sigma)
        self.population_size = int(population_size)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.rng = np.random.default_rng(seed)

        self.mu = max(1, self.population_size // 2)
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mueff = float(1.0 / np.sum(self.weights ** 2))

        n = float(self.dim)
        self.cc = float((4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n))
        self.cs = float((self.mueff + 2.0) / (n + self.mueff + 5.0))
        self.c1 = float(2.0 / ((n + 1.3) ** 2 + self.mueff))
        self.cmu = float(
            min(
                1.0 - self.c1,
                2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff),
            )
        )
        self.damps = float(1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0) + self.cs)

        self.pc = np.zeros(self.dim, dtype=np.float64)
        self.ps = np.zeros(self.dim, dtype=np.float64)
        self.cov = np.eye(self.dim, dtype=np.float64)
        self.basis = np.eye(self.dim, dtype=np.float64)
        self.eigenvalues = np.ones(self.dim, dtype=np.float64)
        self.invsqrt_cov = np.eye(self.dim, dtype=np.float64)
        self.chi_n = float(np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n)))
        self.generation = 0
        self._last_unclipped = None

    def ask(self):
        z = self.rng.standard_normal((self.population_size, self.dim))
        y = z @ (self.basis * np.sqrt(self.eigenvalues)).T
        candidates = self.mean + self.sigma * y
        self._last_unclipped = candidates
        return np.clip(candidates, self.bounds[0], self.bounds[1])

    def tell(self, candidates, fitness_values):
        candidates = np.asarray(candidates, dtype=np.float64)
        fitness_values = np.asarray(fitness_values, dtype=np.float64).reshape(-1)
        order = np.argsort(fitness_values)
        selected = candidates[order[: self.mu]]

        old_mean = self.mean.copy()
        self.mean = np.sum(selected * self.weights[:, None], axis=0)
        y_w = (self.mean - old_mean) / max(self.sigma, 1e-30)

        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * (
            self.invsqrt_cov @ y_w
        )
        ps_norm = float(np.linalg.norm(self.ps))
        n = float(self.dim)
        hsig_threshold = (1.4 + 2.0 / (n + 1.0)) * self.chi_n
        hsig_denom = np.sqrt(max(1e-30, 1.0 - (1.0 - self.cs) ** (2.0 * (self.generation + 1))))
        hsig = float(ps_norm / hsig_denom < hsig_threshold)

        self.pc = (1.0 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * y_w
        y_k = (selected - old_mean) / max(self.sigma, 1e-30)
        rank_mu = np.zeros_like(self.cov)
        for weight, y in zip(self.weights, y_k):
            rank_mu += weight * np.outer(y, y)

        correction = (1.0 - hsig) * self.c1 * self.cc * (2.0 - self.cc) * self.cov
        self.cov = (1.0 - self.c1 - self.cmu) * self.cov + self.c1 * np.outer(self.pc, self.pc) + self.cmu * rank_mu + correction
        self.cov = 0.5 * (self.cov + self.cov.T)
        self.sigma *= float(np.exp((self.cs / self.damps) * (ps_norm / self.chi_n - 1.0)))
        self.mean = np.clip(self.mean, self.bounds[0], self.bounds[1])
        self._update_eigensystem()
        self.generation += 1

    def _update_eigensystem(self):
        eigenvalues, basis = np.linalg.eigh(self.cov)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        self.eigenvalues = eigenvalues
        self.basis = basis
        self.invsqrt_cov = basis @ np.diag(1.0 / np.sqrt(eigenvalues)) @ basis.T


def parse_args():
    parser = argparse.ArgumentParser(description="CMA-ES optimizer for fixed StepNeuroEA parameters.")
    parser.add_argument("--generations", type=int, default=1000, help="Number of outer CMA-ES generations.")
    parser.add_argument("--cma-population-size", type=int, default=None, help="CMA-ES lambda. Defaults to 4 + floor(3 log(d)).")
    parser.add_argument("--sigma", type=float, default=0.35, help="Initial CMA-ES step size in normalized space.")
    parser.add_argument("--eval-rounds", type=int, default=1, help="Evaluation rounds per task and candidate.")
    parser.add_argument("--problem-names", type=str, default="SOP_F1", help="Comma-separated training problems.")
    parser.add_argument("--dimension", type=int, default=10, help="Decision dimension D.")
    parser.add_argument("--population-size", type=int, default=100, help="StepNeuroEA population size N.")
    parser.add_argument("--neuroea-generations", type=int, default=100, help="StepNeuroEA generations per candidate.")
    parser.add_argument(
        "--max-fe",
        type=int,
        default=None,
        help="Maximum function evaluations per episode. Defaults to population_size * (neuroea_generations + 1).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Environment device: cpu / cuda / auto.")
    parser.add_argument("--dtype", type=str, default="float64", help="Environment dtype.")
    parser.add_argument("--task-mode", type=str, default="cycle", choices=["cycle", "random"], help="Task sampling mode.")
    parser.add_argument("--use-ela-observation", action="store_true", help="Build the same ELA observation as SAC runs.")
    parser.add_argument("--ela-feature-scale", type=float, default=1.0, help="Optional post-extraction ELA divisor.")
    parser.add_argument("--ela-objective-index", type=int, default=0, help="Objective column used for ELA.")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="GA/runs/stepneuroea_cmaes1000_neuroea100_sop_f1_d10",
        help="Output directory.",
    )
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="Save fitness curve PNG.")
    parser.add_argument(
        "--save-population-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save CMA-ES population, fitness, reward, mean, and sigma histories as a compressed NPZ file.",
    )
    return parser.parse_args()


def resolve_max_fe(args):
    if args.max_fe is not None:
        return int(args.max_fe)
    return int(args.population_size * (args.neuroea_generations + 1))


def make_env(args):
    max_fe = resolve_max_fe(args)
    tasks = [
        ProblemTask(problem_name, (args.population_size, 1, args.dimension, max_fe))
        for problem_name in parse_csv_strings(args.problem_names)
    ]
    if args.use_ela_observation:
        observation_builder = ELAObservationBuilder(
            include_population=False,
            include_task_context=False,
            feature_scale=args.ela_feature_scale,
            objective_index=args.ela_objective_index,
        )
    else:
        observation_builder = PopulationObservationBuilder(include_population=False)

    return StepNeuroEAEnv(
        tasks=tasks,
        task_mode=args.task_mode,
        initialization="example",
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        observation_builder=observation_builder,
        reward_builder=LogGapReward(),
    )


def normalized_to_parameter_vector(normalized_action, lower_bounds, upper_bounds):
    normalized_action = np.asarray(normalized_action, dtype=np.float64).reshape(-1)
    lower_bounds = np.asarray(lower_bounds, dtype=np.float64).reshape(-1)
    upper_bounds = np.asarray(upper_bounds, dtype=np.float64).reshape(-1)
    return lower_bounds + (np.clip(normalized_action, -1.0, 1.0) + 1.0) * 0.5 * (upper_bounds - lower_bounds)


def parameter_vector_to_normalized(parameter_vector, lower_bounds, upper_bounds):
    parameter_vector = np.asarray(parameter_vector, dtype=np.float64).reshape(-1)
    lower_bounds = np.asarray(lower_bounds, dtype=np.float64).reshape(-1)
    upper_bounds = np.asarray(upper_bounds, dtype=np.float64).reshape(-1)
    denom = np.where(np.abs(upper_bounds - lower_bounds) < 1e-12, 1.0, upper_bounds - lower_bounds)
    return np.clip(2.0 * (parameter_vector - lower_bounds) / denom - 1.0, -1.0, 1.0)


def get_action_metadata(env, seed):
    _, info = env.reset(seed=seed, options={"task_index": 0})
    state = info["state"]
    lower_bounds = tensor_to_numpy(state["lower_bounds"], dtype=np.float64).reshape(-1)
    upper_bounds = tensor_to_numpy(state["upper_bounds"], dtype=np.float64).reshape(-1)
    initial_parameters = tensor_to_numpy(info["initial_update"], dtype=np.float64).reshape(-1)
    initial_action = parameter_vector_to_normalized(initial_parameters, lower_bounds, upper_bounds)
    return lower_bounds, upper_bounds, initial_parameters, initial_action


def evaluate_candidate(env, action, eval_rounds, base_seed):
    per_task = []
    all_rewards = []
    all_best_values = []

    for task_index, task in enumerate(env.tasks):
        rewards = []
        best_values = []
        for eval_round in range(eval_rounds):
            _, info = env.reset(seed=base_seed + 1000 * task_index + eval_round, options={"task_index": task_index})
            total_reward = 0.0

            while True:
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break

            rewards.append(total_reward)
            best_values.append(float(info["best_fitness"]))

        task_result = {
            "task_name": task.problem_name,
            "task_config": task.problem_config,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "best_mean": float(np.mean(best_values)),
            "best_std": float(np.std(best_values)),
        }
        per_task.append(task_result)
        all_rewards.extend(rewards)
        all_best_values.extend(best_values)

    best_values_array = np.asarray(all_best_values, dtype=np.float64)
    reward_array = np.asarray(all_rewards, dtype=np.float64)
    return CandidateResult(
        fitness=float(np.mean(best_values_array)),
        reward_mean=float(np.mean(reward_array)),
        best_mean=float(np.mean(best_values_array)),
        best_std=float(np.std(best_values_array)),
        per_task=per_task,
    )


def append_candidate_rows(path, generation, results, fitness_values, reward_values, generation_best_index, global_best_index):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(
                [
                    "generation",
                    "candidate_index",
                    "fitness",
                    "reward_mean",
                    "best_mean",
                    "best_std",
                    "is_generation_best",
                    "is_global_best_update",
                ]
            )
        for candidate_index, result in enumerate(results):
            writer.writerow(
                [
                    generation,
                    candidate_index,
                    float(fitness_values[candidate_index]),
                    float(reward_values[candidate_index]),
                    result.best_mean,
                    result.best_std,
                    int(candidate_index == generation_best_index),
                    int(candidate_index == global_best_index),
                ]
            )


def append_task_rows(path, generation, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(
                [
                    "generation",
                    "candidate_index",
                    "task_name",
                    "task_config",
                    "reward_mean",
                    "reward_std",
                    "best_mean",
                    "best_std",
                ]
            )
        for candidate_index, result in enumerate(results):
            for task_result in result.per_task:
                writer.writerow(
                    [
                        generation,
                        candidate_index,
                        task_result["task_name"],
                        json.dumps(task_result["task_config"], ensure_ascii=False),
                        task_result["reward_mean"],
                        task_result["reward_std"],
                        task_result["best_mean"],
                        task_result["best_std"],
                    ]
                )


def save_generation_curve(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "generation",
                "elapsed_sec",
                "best_fitness",
                "global_best_fitness",
                "mean_fitness",
                "median_fitness",
                "std_fitness",
                "worst_fitness",
                "best_reward",
                "mean_reward",
                "best_index",
                "sigma",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.generation,
                    record.elapsed_sec,
                    record.best_fitness,
                    record.global_best_fitness,
                    record.mean_fitness,
                    record.median_fitness,
                    record.std_fitness,
                    record.worst_fitness,
                    record.best_reward,
                    record.mean_reward,
                    record.best_index,
                    record.sigma,
                ]
            )


def maybe_plot_fitness_curve(log_dir, records):
    if not records:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    log_dir = Path(log_dir)
    generations = np.asarray([record.generation for record in records], dtype=np.int64)
    best_values = np.asarray([record.best_fitness for record in records], dtype=np.float64)
    global_best_values = np.asarray([record.global_best_fitness for record in records], dtype=np.float64)
    mean_values = np.asarray([record.mean_fitness for record in records], dtype=np.float64)
    median_values = np.asarray([record.median_fitness for record in records], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(generations, best_values, alpha=0.45, label="generation best fitness")
    axis.plot(generations, global_best_values, linewidth=2.0, label="global best fitness")
    axis.plot(generations, mean_values, alpha=0.55, label="population mean fitness")
    axis.plot(generations, median_values, alpha=0.55, label="population median fitness")
    axis.set_xlabel("CMA-ES generation")
    axis.set_ylabel("Final fitness")
    axis.set_yscale("symlog", linthresh=1e-12)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output_path = log_dir / "fitness_curve.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    lower_bounds, upper_bounds, initial_parameters, initial_action = get_action_metadata(env, args.seed)
    action_dim = int(initial_action.shape[0])
    max_fe = resolve_max_fe(args)
    cma_population_size = (
        int(args.cma_population_size)
        if args.cma_population_size is not None
        else int(4 + np.floor(3 * np.log(action_dim)))
    )

    optimizer = SimpleCMAES(
        mean=initial_action,
        sigma=args.sigma,
        population_size=cma_population_size,
        seed=args.seed,
        bounds=(-1.0, 1.0),
    )
    before_eval = evaluate_candidate(env, initial_action, args.eval_rounds, base_seed=10_000)

    records = []
    best_action = initial_action.copy()
    best_result = None
    best_generation = None
    candidate_csv_path = log_dir / "candidate_history.csv"
    task_csv_path = log_dir / "task_history.csv"
    population_history = []
    fitness_history = []
    reward_history = []
    mean_history = []
    sigma_history = []
    generation_best_action_history = []
    global_best_action_history = []
    start_time = time.time()

    for generation in range(1, args.generations + 1):
        population = optimizer.ask()
        if generation == 1:
            population[0] = initial_action
            if cma_population_size > 1:
                population[1] = np.zeros(action_dim, dtype=np.float64)

        if args.save_population_history:
            population_history.append(population.copy())
            mean_history.append(optimizer.mean.copy())
            sigma_history.append(float(optimizer.sigma))

        results = [
            evaluate_candidate(
                env,
                population[index],
                args.eval_rounds,
                base_seed=20_000 + generation * 10_000 + index * 100,
            )
            for index in range(cma_population_size)
        ]
        fitness_values = np.asarray([result.fitness for result in results], dtype=np.float64)
        reward_values = np.asarray([result.reward_mean for result in results], dtype=np.float64)
        generation_best_index = int(np.argmin(fitness_values))
        generation_best_result = results[generation_best_index]
        global_best_update_index = -1

        if best_result is None or generation_best_result.fitness < best_result.fitness:
            best_result = generation_best_result
            best_action = population[generation_best_index].copy()
            best_generation = generation
            global_best_update_index = generation_best_index

        record = GenerationRecord(
            generation=generation,
            elapsed_sec=float(time.time() - start_time),
            best_fitness=float(fitness_values[generation_best_index]),
            global_best_fitness=float(best_result.fitness),
            mean_fitness=float(np.mean(fitness_values)),
            median_fitness=float(np.median(fitness_values)),
            std_fitness=float(np.std(fitness_values)),
            worst_fitness=float(np.max(fitness_values)),
            best_reward=float(reward_values[generation_best_index]),
            mean_reward=float(np.mean(reward_values)),
            best_index=generation_best_index,
            sigma=float(optimizer.sigma),
        )
        records.append(record)
        fitness_history.append(fitness_values.copy())
        reward_history.append(reward_values.copy())
        generation_best_action_history.append(population[generation_best_index].copy())
        global_best_action_history.append(best_action.copy())
        append_candidate_rows(
            candidate_csv_path,
            generation,
            results,
            fitness_values,
            reward_values,
            generation_best_index,
            global_best_update_index,
        )
        append_task_rows(task_csv_path, generation, results)
        print(
            f"generation={generation:04d} "
            f"best={record.best_fitness:.6e} "
            f"global_best={record.global_best_fitness:.6e} "
            f"mean={record.mean_fitness:.6e} "
            f"sigma={record.sigma:.3e} "
            f"reward={record.best_reward:.6e}"
        )

        optimizer.tell(population, fitness_values)

    if best_result is None:
        best_result = before_eval
        best_generation = 0

    after_eval = evaluate_candidate(env, best_action, args.eval_rounds, base_seed=10_000)
    best_parameters = normalized_to_parameter_vector(best_action, lower_bounds, upper_bounds)

    np.save(log_dir / "best_normalized_action.npy", best_action)
    np.save(log_dir / "best_parameters.npy", best_parameters)
    save_generation_curve(log_dir / "cmaes_curve.csv", records)
    plot_path = maybe_plot_fitness_curve(log_dir, records) if args.plot else None

    if args.save_population_history:
        np.savez_compressed(
            log_dir / "training_history.npz",
            population_history=np.asarray(population_history, dtype=np.float64),
            fitness_history=np.asarray(fitness_history, dtype=np.float64),
            reward_history=np.asarray(reward_history, dtype=np.float64),
            mean_history=np.asarray(mean_history, dtype=np.float64),
            sigma_history=np.asarray(sigma_history, dtype=np.float64),
            generation_best_action_history=np.asarray(generation_best_action_history, dtype=np.float64),
            global_best_action_history=np.asarray(global_best_action_history, dtype=np.float64),
        )

    save_json(log_dir / "before_eval.json", before_eval.__dict__)
    save_json(log_dir / "after_eval.json", after_eval.__dict__)
    save_json(
        log_dir / "best_config.json",
        {
            "normalized_action": best_action.tolist(),
            "parameters": best_parameters.tolist(),
            "lower_bounds": lower_bounds.tolist(),
            "upper_bounds": upper_bounds.tolist(),
            "initial_example_parameters": initial_parameters.tolist(),
        },
    )
    save_json(
        log_dir / "run_config.json",
        {
            "args": vars(args),
            "resolved_max_fe": max_fe,
            "cma_population_size": cma_population_size,
            "neuroea_generation_note": (
                "If --max-fe is omitted, max_fe = population_size * (neuroea_generations + 1); "
                "the +1 accounts for initial population evaluation."
            ),
        },
    )

    summary = {
        "generations": len(records),
        "cma_population_size": cma_population_size,
        "initial_sigma": args.sigma,
        "final_sigma": float(optimizer.sigma),
        "neuroea_population_size": args.population_size,
        "neuroea_generations": args.neuroea_generations,
        "max_fe": max_fe,
        "action_dim": action_dim,
        "best_mean": after_eval.best_mean,
        "best_std": after_eval.best_std,
        "reward_mean": after_eval.reward_mean,
        "best_generation": best_generation,
        "global_best_fitness": best_result.fitness,
        "best_parameters_npy": str(log_dir / "best_parameters.npy"),
        "best_config_json": str(log_dir / "best_config.json"),
        "cmaes_curve_csv": str(log_dir / "cmaes_curve.csv"),
        "candidate_history_csv": str(candidate_csv_path),
        "task_history_csv": str(task_csv_path),
        "training_history_npz": str(log_dir / "training_history.npz") if args.save_population_history else None,
        "fitness_curve_png": None if plot_path is None else str(plot_path),
        "before_eval": before_eval.__dict__,
        "after_eval": after_eval.__dict__,
    }
    save_json(log_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
