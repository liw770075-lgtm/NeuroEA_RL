"""Unified stable training for dynamic SAC, first-action SAC, and dynamic TD3."""

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
from RL.shared.ELA.ELA import ELA_FEATURE_NAMES, ELA_IMPLEMENTATION_VERSION
from RL.Dynamic.common import (
    NORMALIZATION_VERSION,
    FirstActionReuseEnv,
    StableELAObservationBuilder,
    parse_problem_names,
)
from RL.shared.env.problem_utils import ProblemTask
from RL.shared.env.stepneuroea_env import StepNeuroEAEnv
from RL.shared.env.stepwise_ea_env import LogGapReward
from RL.shared.examples.train_stepneuroea_sac_multitask_simpleNN import (
    evaluate_multitask,
    maybe_plot_curves,
    save_json,
    save_reward_curve_csv,
)
from RL.shared.method.observation import flatten_observation, infer_observation_dim
from RL.shared.method.replay_buffer import ReplayBuffer
from RL.shared.method.sac_agent import SACAgent
from RL.shared.trainer.sac_trainer import SACTrainer


def parse_hidden_dims(text):
    dims = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("hidden dimensions must be positive integers")
    return dims


def parse_args(forced_algorithm=None, forced_mode=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["sac", "td3"], default=forced_algorithm or "sac")
    parser.add_argument("--mode", choices=["dynamic", "first-action"], default=forced_mode or "dynamic")
    parser.add_argument("--problem-names", default="SOP_F1")
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Episodes per problem (paper default: 5000).",
    )
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--max-fe", type=int, default=10000)
    parser.add_argument("--objectives", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-rounds", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--start-steps", type=int, default=128)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(128, 128))
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--ela-feature-scale", type=float, default=1.0)
    parser.add_argument("--ela-objective-index", type=int, default=0)
    parser.add_argument("--summary-clip", type=float, default=5.0)
    parser.add_argument("--objective-log-scale", type=float, default=10.0)
    parser.add_argument("--actor-output-clip", type=float, default=5.0)
    parser.add_argument("--include-task-context", action="store_true")
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--output-root", default="RL/runs/dynamic/stable")
    args = parser.parse_args()
    if forced_algorithm is not None:
        args.algorithm = forced_algorithm
    if forced_mode is not None:
        args.mode = forced_mode
    if args.algorithm == "td3" and args.mode != "dynamic":
        parser.error("TD3 currently supports --mode dynamic only.")
    return args


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_environment(args, problem_name, seed):
    task = ProblemTask(
        problem_name,
        (args.population_size, args.objectives, args.dimension, args.max_fe),
    )
    base_env = StepNeuroEAEnv(
        tasks=[task],
        task_mode="cycle",
        initialization="example",
        seed=seed,
        device=args.device,
        dtype=args.dtype,
        observation_builder=StableELAObservationBuilder(
            include_population=False,
            include_task_context=args.include_task_context,
            feature_scale=args.ela_feature_scale,
            objective_index=args.ela_objective_index,
            summary_clip=args.summary_clip,
            objective_log_scale=args.objective_log_scale,
        ),
        reward_builder=LogGapReward(),
    )
    return FirstActionReuseEnv(base_env) if args.mode == "first-action" else base_env


def build_agent(args, observation_dim, action_dim, device):
    common = dict(
        observation_dim=observation_dim,
        action_dim=action_dim,
        device=device,
        hidden_dims=args.hidden_dims,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
    )
    if args.algorithm == "sac":
        return SACAgent(**common, actor_mean_clip=args.actor_output_clip)
    return TD3Agent(
        **common,
        exploration_noise=args.exploration_noise,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        actor_output_clip=args.actor_output_clip,
    )


def train_problem(args, problem_name, problem_index):
    run_seed = int(args.seed + problem_index * args.seed_stride)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    log_dir = Path(args.output_root) / problem_name
    env = build_environment(args, problem_name, run_seed)
    initial_observation, _ = env.reset(seed=run_seed, options={"task_index": 0})
    observation_dim = infer_observation_dim(initial_observation)
    action_dim = int(np.prod(env.action_space.shape))
    agent_device = resolve_device(args.device)
    agent = build_agent(args, observation_dim, action_dim, agent_device)
    replay_buffer = ReplayBuffer(observation_dim, action_dim, capacity=100_000)
    trainer = SACTrainer(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        updates_per_step=args.updates_per_step,
        update_after_steps=max(args.start_steps, args.batch_size),
        train_freq_steps=1,
        gradient_steps=args.updates_per_step,
        normalize_rewards=True,
        reward_clip=10.0,
        log_dir=log_dir,
        observation_preprocessor=flatten_observation,
        show_progress=True,
        show_episode_progress=args.mode == "dynamic",
        best_checkpoint_metric="eval_fitness",
    )

    eval_seed = 10_000 + run_seed
    before_eval = evaluate_multitask(env, agent, args.eval_rounds, eval_seed)
    results = trainer.train(
        num_episodes=args.episodes,
        start_seed=run_seed,
        eval_fn=lambda: evaluate_multitask(env, agent, args.eval_rounds, eval_seed),
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    after_eval = evaluate_multitask(env, agent, args.eval_rounds, eval_seed)

    save_json(log_dir / "before_eval.json", before_eval)
    save_json(log_dir / "after_eval.json", after_eval)
    save_reward_curve_csv(log_dir / "reward_curve.csv", results)
    run_config = {
        "algorithm": args.algorithm.upper(),
        "mode": args.mode,
        "problem_name": problem_name,
        "seed": run_seed,
        "normalization_version": NORMALIZATION_VERSION,
        "ela_implementation": ELA_IMPLEMENTATION_VERSION,
        "ela_feature_names": list(ELA_FEATURE_NAMES),
        "args": {**vars(args), "hidden_dims": list(args.hidden_dims)},
    }
    save_json(log_dir / "run_config.json", run_config)
    plot_path = maybe_plot_curves(log_dir, results)

    rewards = np.asarray([item.total_reward for item in results], dtype=np.float64)
    fitness = np.asarray([item.final_best_fitness for item in results], dtype=np.float64)
    policy_results = [item for item in results if item.num_updates > 0]
    policy_fitness = np.asarray(
        [item.final_best_fitness for item in policy_results], dtype=np.float64
    )
    summary = {
        "algorithm": args.algorithm.upper(),
        "mode": args.mode,
        "flow": "first_action_reuse" if args.mode == "first-action" else "dynamic",
        "problem_name": problem_name,
        "seed": run_seed,
        "episodes": len(results),
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "normalization_version": NORMALIZATION_VERSION,
        "ela_implementation": ELA_IMPLEMENTATION_VERSION,
        "ela_feature_names": list(ELA_FEATURE_NAMES),
        "reward_first_10_mean": float(rewards[:10].mean()),
        "reward_last_10_mean": float(rewards[-10:].mean()),
        "best_first_10_mean": float(fitness[:10].mean()),
        "best_last_10_mean": float(fitness[-10:].mean()),
        "policy_train_best_fitness": float(policy_fitness.min()) if policy_fitness.size else None,
        "before_eval": before_eval,
        "after_eval": after_eval,
        "best_checkpoint": str(log_dir / "checkpoints" / "best_fitness.pt"),
        "reward_curve_png": None if plot_path is None else str(plot_path),
    }
    save_json(log_dir / "summary.json", summary)
    return summary


def save_aggregate(args, summaries):
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    fields = [
        "problem_name", "seed", "algorithm", "mode", "episodes",
        "policy_train_best_fitness", "best_last_10_mean", "reward_last_10_mean",
        "best_checkpoint",
    ]
    with (root / "aggregate_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: summary.get(key) for key in fields} for summary in summaries)
    save_json(root / "aggregate_results.json", {"args": vars(args), "summaries": summaries})


def main(forced_algorithm=None, forced_mode=None):
    args = parse_args(forced_algorithm, forced_mode)
    problems = parse_problem_names(args.problem_names)
    if not problems:
        raise ValueError("No problems were specified.")
    summaries = []
    for index, problem_name in enumerate(problems):
        print(f"\n===== {args.algorithm.upper()} {args.mode}: {problem_name} ({index + 1}/{len(problems)}) =====")
        summary = train_problem(args, problem_name, index)
        summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    save_aggregate(args, summaries)


if __name__ == "__main__":
    main()

