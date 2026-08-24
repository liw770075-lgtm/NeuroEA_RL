"""Test paper-state static SAC checkpoints and save convergence CSV files."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import re
import sys

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np
import torch

from RL.shared.static_config_flow.sequential_env import (
    SequentialStaticStepNeuroEAConfigEnv,
)
from RL.shared.env.problem_utils import ProblemTask
from RL.shared.env.stepwise_ea_env import _extract_best_value
from RL.shared.method.observation import flatten_observation, infer_observation_dim
from RL.shared.method.sac_agent import SACAgent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--checkpoint-template")
    source.add_argument("--model-root")
    parser.add_argument("--checkpoint-name", default="best_fitness.pt")
    parser.add_argument("--problem-name", default="SOP_F1")
    parser.add_argument("--problem-names", default=None)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--max-fe", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="result/Static_PaperState")
    parser.add_argument("--output-prefix", default="StaticPaperState")
    parser.add_argument("--summary-output", default=None)
    return parser.parse_args()


def parse_names(text):
    if not text:
        return []
    names = []
    for item in (part.strip() for part in str(text).split(",")):
        match = re.fullmatch(r"(.+_F)(\d+)-(?:(?:.+_F)?)(\d+)", item, re.IGNORECASE)
        if match:
            prefix, first, last = match.groups()
            first, last = int(first), int(last)
            step = 1 if last >= first else -1
            names.extend(f"{prefix}{index}" for index in range(first, last + step, step))
        elif item:
            names.append(item)
    return names


def suffix(problem_name):
    match = re.search(r"F(\d+)$", problem_name, re.IGNORECASE)
    return f"F{match.group(1)}" if match else problem_name


def resolve_checkpoint(args, problem_name):
    values = {"problem_name": problem_name, "problem_suffix": suffix(problem_name)}
    if args.checkpoint:
        return Path(args.checkpoint)
    if args.checkpoint_template:
        return Path(args.checkpoint_template.format(**values))
    root = Path(args.model_root)
    candidates = [
        root / problem_name / "checkpoints" / args.checkpoint_name,
        root / "checkpoints" / args.checkpoint_name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def load_config(checkpoint_path):
    path = checkpoint_path.parent.parent / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run_config.json: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    expected_state_design = "ela_parameter_index_one_hot_v1"
    if config.get("state_design") != expected_state_design:
        raise ValueError(
            "Checkpoint was not trained with the paper state s_i=[e0,h_i]. "
            f"Expected {expected_state_design!r}, "
            f"got {config.get('state_design')!r}."
        )
    return config


def infer_hidden_dims(agent_state):
    weights = [
        value for key, value in agent_state["actor"].items()
        if key.endswith("weight") and value.ndim == 2
    ]
    return tuple(int(weight.shape[0]) for weight in weights[:-1])


def select_parameters(env, agent, observation):
    normalized = []
    for index in range(env.num_params):
        action = agent.act(flatten_observation(observation), deterministic=True)
        value = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        normalized.append(value)
        if index + 1 < env.num_params:
            observation, _, terminated, truncated, _ = env.step([value])
            if terminated or truncated:
                raise RuntimeError("Configuration ended before all parameters were selected.")
    return env._denormalize_params(np.asarray(normalized, dtype=np.float32))


def optimizer_history(env, real_params, run):
    env.algorithm.apply_update(real_params)
    state = env.algorithm.get_state()
    rows = [{"run": run, "generation": int(state.get("generation", 0)), "best_fitness": float(_extract_best_value(state))}]
    while not env.algorithm.is_done():
        env.algorithm.run_generation()
        state = env.algorithm.get_state()
        rows.append({"run": run, "generation": int(state.get("generation", 0)), "best_fitness": float(_extract_best_value(state))})
    return rows


def test_problem(args, problem_name, checkpoint_path, output_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path.resolve()}")
    config = load_config(checkpoint_path)
    train_args = config.get("args", {})
    task = ProblemTask(problem_name, (args.population_size, 2, args.dimension, args.max_fe))
    if bool(train_args.get("include_task_context", False)):
        raise ValueError(
            "The checkpoint includes task context and therefore does not use "
            "the strict paper state s_i=[e0,h_i]."
        )
    env = SequentialStaticStepNeuroEAConfigEnv(
        tasks=[task],
        initialization="example",
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        include_task_context=False,
        ela_feature_scale=float(train_args.get("ela_feature_scale", 100.0)),
        ela_objective_index=int(train_args.get("ela_objective_index", 0)),
    )
    observation, _ = env.reset(seed=args.seed, options={"task_index": 0})
    observation_dim = infer_observation_dim(observation)
    expected_observation_dim = 9 + env.num_params
    if observation_dim != expected_observation_dim:
        raise RuntimeError(
            "Unexpected paper-state dimension: "
            f"expected={expected_observation_dim}, actual={observation_dim}."
        )
    action_dim = int(np.prod(env.action_space.shape))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("agent", checkpoint)
    saved_dims = (int(state["observation_dim"]), int(state["action_dim"]))
    if saved_dims != (observation_dim, action_dim):
        raise ValueError(
            f"Environment/checkpoint dimension mismatch: env={(observation_dim, action_dim)}, "
            f"checkpoint={saved_dims}."
        )
    agent = SACAgent(
        observation_dim,
        action_dim,
        device=device,
        hidden_dims=infer_hidden_dims(state),
        actor_mean_clip=float(train_args.get("actor_output_clip", 5.0)),
    )
    agent.load_state_dict(state)

    all_rows, final_values = [], []
    for run_index in range(args.rounds):
        run, run_seed = run_index + 1, args.seed + run_index
        observation, _ = env.reset(seed=run_seed, options={"task_index": 0})
        real_params = select_parameters(env, agent, observation)
        rows = optimizer_history(env, real_params, run)
        all_rows.extend(rows)
        final_values.append(rows[-1]["best_fitness"])
        print(f"run={run}/{args.rounds} seed={run_seed} best_fitness={rows[-1]['best_fitness']:.6e}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "generation", "best_fitness"])
        writer.writeheader()
        writer.writerows(all_rows)
    values = np.asarray(final_values, dtype=np.float64)
    return {
        "problem_name": problem_name,
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "result": f"{values.mean():.4e}({values.std():.2e})",
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
    }


def main():
    args = parse_args()
    problems = parse_names(args.problem_names) if args.problem_names else [args.problem_name]
    batch = len(problems) > 1 or args.problem_names is not None
    if batch and args.output:
        raise ValueError("Use --output-dir for batch testing.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    summaries = []
    for problem_name in problems:
        checkpoint = resolve_checkpoint(args, problem_name)
        output = Path(args.output) if args.output else Path(args.output_dir) / f"{args.output_prefix}_{suffix(problem_name)}.csv"
        print(f"\nTesting {problem_name}: {checkpoint}")
        summaries.append(test_problem(args, problem_name, checkpoint, output, device))
    summary_path = Path(args.summary_output) if args.summary_output else Path(args.output_dir) / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["problem_name", "mean", "std", "min", "result", "checkpoint", "output"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    for item in summaries:
        print(f"{item['problem_name']}: {item['result']}")


if __name__ == "__main__":
    main()

