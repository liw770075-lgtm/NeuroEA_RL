"""Unified testing for stable dynamic NeuroEA RL checkpoints."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np
import torch

from RL.shared.TD3.static.td3_agent import TD3Agent
from RL.shared.ELA.ELA import ELA_IMPLEMENTATION_VERSION
from RL.Dynamic.common import (
    NORMALIZATION_VERSION,
    StableELAObservationBuilder,
    parse_problem_names,
    problem_suffix,
)
from RL.shared.env.problem_utils import ProblemTask
from RL.shared.env.ela_observation import ELAObservationBuilder
from RL.shared.env.stepneuroea_env import StepNeuroEAEnv
from RL.shared.env.stepwise_ea_env import LogGapReward
from RL.shared.method.observation import flatten_observation, infer_observation_dim
from RL.shared.method.sac_agent import SACAgent


def parse_args(
    forced_algorithm=None,
    forced_mode=None,
    default_model_root=None,
    default_problem_names=None,
):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["sac", "td3"], default=forced_algorithm or "sac")
    parser.add_argument("--mode", choices=["dynamic", "first-action"], default=forced_mode or "dynamic")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-template", default=None)
    parser.add_argument("--model-root", default=default_model_root)
    parser.add_argument("--checkpoint-name", default="best_fitness.pt")
    parser.add_argument("--problem-name", default="SOP_F1")
    parser.add_argument("--problem-names", default=default_problem_names)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--max-fe", type=int, default=10000)
    parser.add_argument("--objectives", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="result/dynamic_stable")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--summary-output", default=None)
    args = parser.parse_args()
    if forced_algorithm is not None:
        args.algorithm = forced_algorithm
    if forced_mode is not None:
        args.mode = forced_mode
    return args


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_checkpoint(args, problem_name, batch_mode):
    values = {
        "problem_name": problem_name,
        "problem_name_lower": problem_name.lower(),
        "problem_suffix": problem_suffix(problem_name),
        "dimension": args.dimension,
    }
    if args.checkpoint_template:
        return Path(args.checkpoint_template.format(**values))
    if args.model_root:
        root = Path(args.model_root)
        candidates = [
            root / problem_name / "checkpoints" / args.checkpoint_name,
            root / "checkpoints" / args.checkpoint_name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]
    if args.checkpoint and not batch_mode:
        return Path(args.checkpoint)
    if args.checkpoint and batch_mode:
        return Path(str(args.checkpoint).format(**values))
    raise ValueError("Supply --checkpoint, --checkpoint-template, or --model-root.")


def load_run_config(checkpoint_path):
    path = Path(checkpoint_path).parent.parent / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run_config.json beside checkpoint: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_environment(args, problem_name, run_config):
    train_args = run_config.get("args", {})
    ela_implementation = run_config.get("ela_implementation")
    if ela_implementation != ELA_IMPLEMENTATION_VERSION:
        raise ValueError(
            "Checkpoint uses an incompatible ELA implementation. "
            f"Expected {ELA_IMPLEMENTATION_VERSION!r}, "
            f"got {ela_implementation!r}; retrain the model."
        )
    normalization_version = run_config.get("normalization_version")
    if normalization_version not in (None, NORMALIZATION_VERSION):
        raise ValueError(
            f"Checkpoint normalization is {normalization_version!r}; expected "
            f"{NORMALIZATION_VERSION!r}. Retrain it with RL/dynamic training scripts."
        )
    recorded_problem = run_config.get("problem_name")
    if recorded_problem and recorded_problem.lower() != problem_name.lower():
        raise ValueError(
            f"Checkpoint was trained for {recorded_problem}, not requested {problem_name}."
        )
    task = ProblemTask(
        problem_name,
        (args.population_size, args.objectives, args.dimension, args.max_fe),
    )
    if normalization_version is None:
        # Checkpoints produced by the original dynamic ELA trainer predate the
        # normalization metadata and were trained on the raw ELA observation.
        observation_builder = ELAObservationBuilder(
            include_population=False,
            include_task_context=bool(train_args.get("include_task_context", False)),
            feature_scale=float(train_args.get("ela_feature_scale", 1.0)),
            objective_index=int(train_args.get("ela_objective_index", 0)),
        )
    else:
        observation_builder = StableELAObservationBuilder(
            include_population=False,
            include_task_context=bool(train_args.get("include_task_context", False)),
            feature_scale=float(train_args.get("ela_feature_scale", 1.0)),
            objective_index=int(train_args.get("ela_objective_index", 0)),
            summary_clip=float(train_args.get("summary_clip", 5.0)),
            objective_log_scale=float(train_args.get("objective_log_scale", 10.0)),
        )
    return StepNeuroEAEnv(
        tasks=[task],
        task_mode="cycle",
        initialization="example",
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        observation_builder=observation_builder,
        reward_builder=LogGapReward(),
    )


def build_agent(args, checkpoint, run_config, observation_dim, action_dim, device):
    train_args = run_config.get("args", {})
    agent_state = checkpoint.get("agent", checkpoint)
    saved_dims = (
        int(agent_state.get("observation_dim", observation_dim)),
        int(agent_state.get("action_dim", action_dim)),
    )
    if saved_dims != (observation_dim, action_dim):
        raise ValueError(
            "Environment/checkpoint dimension mismatch: "
            f"env=({observation_dim}, {action_dim}), checkpoint={saved_dims}."
        )
    if "hidden_dims" in train_args:
        hidden_dims = tuple(int(value) for value in train_args["hidden_dims"])
    else:
        weights = [
            value
            for key, value in agent_state["actor"].items()
            if key.endswith("weight") and value.ndim == 2
        ]
        if len(weights) < 2:
            raise ValueError("Unable to infer hidden dimensions from checkpoint actor.")
        hidden_dims = tuple(int(weight.shape[0]) for weight in weights[:-1])
    actor_clip = float(train_args.get("actor_output_clip", 5.0))
    if args.algorithm == "sac":
        agent = SACAgent(
            observation_dim,
            action_dim,
            device=device,
            hidden_dims=hidden_dims,
            actor_mean_clip=actor_clip,
        )
    else:
        agent = TD3Agent(
            observation_dim,
            action_dim,
            device=device,
            hidden_dims=hidden_dims,
            exploration_noise=float(train_args.get("exploration_noise", 0.1)),
            policy_noise=float(train_args.get("policy_noise", 0.2)),
            noise_clip=float(train_args.get("noise_clip", 0.5)),
            policy_delay=int(train_args.get("policy_delay", 2)),
            actor_output_clip=actor_clip,
        )
    agent.load_state_dict(agent_state)
    return agent


def test_problem(args, problem_name, checkpoint_path, output_path, device):
    run_config = load_run_config(checkpoint_path)
    recorded_algorithm = str(run_config.get("algorithm", args.algorithm)).lower()
    recorded_mode = str(run_config.get("mode", args.mode)).lower()
    if recorded_algorithm != args.algorithm or recorded_mode != args.mode:
        raise ValueError(
            f"Requested {args.algorithm}/{args.mode}, checkpoint is "
            f"{recorded_algorithm}/{recorded_mode}."
        )
    env = build_environment(args, problem_name, run_config)
    observation, _ = env.reset(seed=args.seed, options={"task_index": 0})
    observation_dim = infer_observation_dim(observation)
    action_dim = int(np.prod(env.action_space.shape))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent = build_agent(args, checkpoint, run_config, observation_dim, action_dim, device)

    rows = []
    final_fitness = []
    total_rewards = []
    for run_index in range(args.rounds):
        run = run_index + 1
        run_seed = args.seed + run_index
        observation, info = env.reset(seed=run_seed, options={"task_index": 0})
        rows.append({"run": run, "generation": int(info["generation"]), "best_fitness": float(info["best_fitness"])})
        total_reward = 0.0
        fixed_action = None
        while True:
            if args.mode == "first-action":
                if fixed_action is None:
                    fixed_action = agent.act(flatten_observation(observation), deterministic=True)
                action = fixed_action
            else:
                action = agent.act(flatten_observation(observation), deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            rows.append({"run": run, "generation": int(info["generation"]), "best_fitness": float(info["best_fitness"])})
            if terminated or truncated:
                break
        total_rewards.append(total_reward)
        final_fitness.append(float(info["best_fitness"]))
        print(
            f"run={run}/{args.rounds} seed={run_seed} reward={total_reward:.6f} "
            f"best_fitness={float(info['best_fitness']):.6e}",
            flush=True,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "generation", "best_fitness"])
        writer.writeheader()
        writer.writerows(rows)
    fitness = np.asarray(final_fitness, dtype=np.float64)
    rewards = np.asarray(total_rewards, dtype=np.float64)
    return {
        "problem_name": problem_name,
        "algorithm": args.algorithm.upper(),
        "mode": args.mode,
        "mean": float(fitness.mean()),
        "std": float(fitness.std()),
        "min": float(fitness.min()),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "result": f"{fitness.mean():.4e}({fitness.std():.2e})",
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
    }


def main(
    forced_algorithm=None,
    forced_mode=None,
    default_model_root=None,
    default_problem_names=None,
):
    args = parse_args(
        forced_algorithm,
        forced_mode,
        default_model_root,
        default_problem_names,
    )
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive.")
    if args.dimension <= 0 or args.population_size <= 0 or args.max_fe <= 0:
        raise ValueError("--dimension, --population-size, and --max-fe must be positive.")
    problems = parse_problem_names(args.problem_names) if args.problem_names else [args.problem_name]
    batch_mode = len(problems) > 1 or args.problem_names is not None
    if batch_mode and args.output:
        raise ValueError("Use --output-dir in batch mode.")
    device = resolve_device(args.device)
    prefix = args.output_prefix or f"{args.algorithm.upper()}_{args.mode.replace('-', '_')}"
    summaries = []
    for problem_name in problems:
        checkpoint_path = resolve_checkpoint(args, problem_name, batch_mode)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path.resolve()}")
        if batch_mode:
            output_path = Path(args.output_dir) / f"{prefix}_{problem_suffix(problem_name)}.csv"
        else:
            output_path = Path(args.output) if args.output else Path(args.output_dir) / f"{prefix}_{problem_suffix(problem_name)}.csv"
        print(f"\nTesting {problem_name}: {checkpoint_path}")
        summaries.append(test_problem(args, problem_name, checkpoint_path, output_path, device))

    summary_path = Path(args.summary_output) if args.summary_output else Path(args.output_dir) / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["problem_name", "algorithm", "mode", "mean", "std", "min", "reward_mean", "reward_std", "result", "checkpoint", "output"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    for summary in summaries:
        print(f"{summary['problem_name']}: {summary['result']}")
    print(f"summary saved: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
