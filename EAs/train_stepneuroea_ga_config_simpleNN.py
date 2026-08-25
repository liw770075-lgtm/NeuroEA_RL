"""Optimize fixed StepNeuroEA block parameters with a simple GA.

This script mirrors the environment setup used by
``train_stepneuroea_sac_ela_simpleNN.py`` but replaces SAC with an outer
genetic algorithm. Each GA individual is a normalized parameter vector in the
same action space used by the RL agent. During one StepNeuroEA episode, that
vector is applied at every generation, so the GA learns a static NeuroEA
configuration.
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
from RL.shared.examples.train_stepneuroea_sac_multitask_simpleNN import (
    parse_csv_strings,
    save_json,
)

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
    evaluation_rows: list


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


def parse_args():
    parser = argparse.ArgumentParser(description="GA optimizer for fixed StepNeuroEA parameters.")
    parser.add_argument("--generations", type=int, default=5000, help="Number of outer GA generations.")
    parser.add_argument("--ga-population-size", type=int, default=32, help="Number of parameter vectors in GA.")
    parser.add_argument("--elite-fraction", type=float, default=0.15, help="Fraction of GA population copied unchanged.")
    parser.add_argument("--mutation-sigma", type=float, default=0.15, help="Gaussian mutation std in normalized space.")
    parser.add_argument("--mutation-prob", type=float, default=0.20, help="Per-gene mutation probability.")
    parser.add_argument("--crossover-rate", type=float, default=0.80, help="Uniform crossover probability.")
    parser.add_argument("--tournament-size", type=int, default=3, help="Tournament size for parent selection.")
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
        default="GA/runs/stepneuroea_ga5000_neuroea100_sop_f1_d10",
        help="Output directory.",
    )
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="Save fitness curve PNG.")
    parser.add_argument(
        "--save-population-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save full GA population, fitness, and reward histories as a compressed NPZ file.",
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
    action_dim = int(initial_parameters.shape[0])
    return lower_bounds, upper_bounds, initial_parameters, action_dim


def evaluate_candidate(env, action, eval_rounds, base_seed, generation=None, candidate_index=None, config_eval_id=None):
    per_task = []
    all_rewards = []
    all_best_values = []
    evaluation_rows = []

    for task_index, task in enumerate(env.tasks):
        rewards = []
        best_values = []
        for eval_round in range(eval_rounds):
            episode_seed = int(base_seed + 1000 * task_index + eval_round)
            _, info = env.reset(seed=episode_seed, options={"task_index": task_index})
            total_reward = 0.0

            while True:
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break

            rewards.append(total_reward)
            final_best_fitness = float(info["best_fitness"])
            best_values.append(final_best_fitness)
            evaluation_rows.append(
                {
                    "configuration_evaluation": None if config_eval_id is None else int(config_eval_id),
                    "generation": None if generation is None else int(generation),
                    "candidate_index": None if candidate_index is None else int(candidate_index),
                    "task_index": int(task_index),
                    "task_name": task.problem_name,
                    "task_config": task.problem_config,
                    "eval_round": int(eval_round),
                    "seed": episode_seed,
                    "total_reward": float(total_reward),
                    "final_best_fitness": final_best_fitness,
                }
            )

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
        evaluation_rows=evaluation_rows,
    )


def tournament_select(rng, population, fitness_values, tournament_size):
    population_size = population.shape[0]
    indices = rng.integers(0, population_size, size=max(1, tournament_size))
    winner_index = indices[int(np.argmin(fitness_values[indices]))]
    return population[winner_index].copy()


def make_next_population(rng, population, fitness_values, args):
    population_size, action_dim = population.shape
    elite_count = max(1, int(round(population_size * args.elite_fraction)))
    elite_indices = np.argsort(fitness_values)[:elite_count]
    next_population = [population[index].copy() for index in elite_indices]

    while len(next_population) < population_size:
        parent_a = tournament_select(rng, population, fitness_values, args.tournament_size)
        parent_b = tournament_select(rng, population, fitness_values, args.tournament_size)

        if rng.random() < args.crossover_rate:
            mask = rng.random(action_dim) < 0.5
            child = np.where(mask, parent_a, parent_b)
        else:
            child = parent_a

        mutation_mask = rng.random(action_dim) < args.mutation_prob
        if np.any(mutation_mask):
            child = child.copy()
            child[mutation_mask] += rng.normal(0.0, args.mutation_sigma, size=int(np.sum(mutation_mask)))

        next_population.append(np.clip(child, -1.0, 1.0))

    return np.asarray(next_population, dtype=np.float64)


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
                ]
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


def append_evaluation_rows(path, results, population, lower_bounds, upper_bounds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(
                [
                    "configuration_evaluation",
                    "generation",
                    "candidate_index",
                    "task_index",
                    "task_name",
                    "task_config",
                    "eval_round",
                    "seed",
                    "final_best_fitness",
                    "total_reward",
                    "normalized_action",
                    "parameter_vector",
                ]
            )
        for candidate_index, result in enumerate(results):
            normalized_action = np.asarray(population[candidate_index], dtype=np.float64).reshape(-1)
            parameter_vector = normalized_to_parameter_vector(normalized_action, lower_bounds, upper_bounds)
            normalized_json = json.dumps(normalized_action.tolist(), ensure_ascii=False)
            parameter_json = json.dumps(parameter_vector.tolist(), ensure_ascii=False)
            for row in result.evaluation_rows:
                writer.writerow(
                    [
                        row["configuration_evaluation"],
                        row["generation"],
                        row["candidate_index"],
                        row["task_index"],
                        row["task_name"],
                        json.dumps(row["task_config"], ensure_ascii=False),
                        row["eval_round"],
                        row["seed"],
                        row["final_best_fitness"],
                        row["total_reward"],
                        normalized_json,
                        parameter_json,
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
    axis.set_xlabel("GA generation")
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
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    lower_bounds, upper_bounds, initial_parameters, action_dim = get_action_metadata(env, args.seed)
    max_fe = resolve_max_fe(args)

    population = rng.uniform(-1.0, 1.0, size=(args.ga_population_size, action_dim))

    records = []
    best_action = None
    best_result = None
    best_generation = None
    candidate_csv_path = log_dir / "candidate_history.csv"
    task_csv_path = log_dir / "task_history.csv"
    evaluation_csv_path = log_dir / "evaluation_history.csv"
    population_history = []
    fitness_history = []
    reward_history = []
    generation_best_action_history = []
    global_best_action_history = []
    start_time = time.time()

    for generation in range(1, args.generations + 1):
        if args.save_population_history:
            population_history.append(population.copy())

        results = [
            evaluate_candidate(
                env,
                population[index],
                args.eval_rounds,
                base_seed=20_000 + generation * 10_000 + index * 100,
                generation=generation,
                candidate_index=index,
                config_eval_id=(generation - 1) * args.ga_population_size + index + 1,
            )
            for index in range(args.ga_population_size)
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
        append_evaluation_rows(evaluation_csv_path, results, population, lower_bounds, upper_bounds)
        print(
            f"generation={generation:04d} "
            f"best={record.best_fitness:.6e} "
            f"global_best={record.global_best_fitness:.6e} "
            f"mean={record.mean_fitness:.6e} "
            f"reward={record.best_reward:.6e}"
        )

        population = make_next_population(rng, population, fitness_values, args)

    if best_result is None or best_action is None:
        raise RuntimeError("No GA candidate was evaluated. Increase --generations above 0.")

    after_eval = evaluate_candidate(env, best_action, args.eval_rounds, base_seed=10_000)
    best_parameters = normalized_to_parameter_vector(best_action, lower_bounds, upper_bounds)

    np.save(log_dir / "best_normalized_action.npy", best_action)
    np.save(log_dir / "best_parameters.npy", best_parameters)
    save_generation_curve(log_dir / "ga_curve.csv", records)
    plot_path = maybe_plot_fitness_curve(log_dir, records) if args.plot else None

    if args.save_population_history:
        np.savez_compressed(
            log_dir / "training_history.npz",
            population_history=np.asarray(population_history, dtype=np.float64),
            fitness_history=np.asarray(fitness_history, dtype=np.float64),
            reward_history=np.asarray(reward_history, dtype=np.float64),
            generation_best_action_history=np.asarray(generation_best_action_history, dtype=np.float64),
            global_best_action_history=np.asarray(global_best_action_history, dtype=np.float64),
        )

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
            "neuroea_generation_note": (
                "If --max-fe is omitted, max_fe = population_size * (neuroea_generations + 1); "
                "the +1 accounts for initial population evaluation."
            ),
        },
    )

    summary = {
        "generations": len(records),
        "ga_population_size": args.ga_population_size,
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
        "ga_curve_csv": str(log_dir / "ga_curve.csv"),
        "candidate_history_csv": str(candidate_csv_path),
        "task_history_csv": str(task_csv_path),
        "evaluation_history_csv": str(evaluation_csv_path),
        "training_history_npz": str(log_dir / "training_history.npz") if args.save_population_history else None,
        "fitness_curve_png": None if plot_path is None else str(plot_path),
        "after_eval": after_eval.__dict__,
    }
    save_json(log_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
