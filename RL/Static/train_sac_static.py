"""Train sequential static SAC with previous actions included in the state."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
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
from RL.shared.TD3.static.td3_agent import TD3Agent
from RL.shared.env.problem_utils import ProblemTask
from RL.shared.method.observation import flatten_observation, infer_observation_dim
from RL.shared.method.replay_buffer import ReplayBuffer
from RL.shared.method.sac_agent import SACAgent
from RL.shared.static_config_flow.sequential_trainer import (
    SequentialRewardShapingConfig,
    SequentialStaticSACTrainer,
)
from RL.shared.static_config_flow.train_static_ela_sac import (
    evaluate_multitask,
    maybe_plot_curves,
    save_json,
    save_reward_curve_csv,
)


def parse_names(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def add_arguments(parser, output_argument="log-dir"):
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--problem-names", default="SOP_F1")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--max-fe", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--task-mode", choices=["cycle", "random"], default="cycle")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-rounds", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=960)
    parser.add_argument("--start-steps", type=int, default=128)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--ela-feature-scale", type=float, default=100.0)
    parser.add_argument("--ela-objective-index", type=int, default=0)
    parser.add_argument("--reward-clip", type=float, default=None)
    parser.add_argument("--actor-output-clip", type=float, default=5.0)
    parser.add_argument("--default-penalty-weight", type=float, default=0.0)
    parser.add_argument("--default-close-radius", type=float, default=0.6)
    parser.add_argument("--default-close-signed-weight", type=float, default=0.0)
    parser.add_argument("--final-only-after-episode", type=int, default=None)
    parser.add_argument("--include-task-context", action="store_true")
    if output_argument == "log-dir":
        parser.add_argument(
            "--log-dir",
            default="RL/runs/Static/action_history_sac_sop_f1",
        )
    else:
        parser.add_argument(
            "--output-root",
            default="RL/runs/Static/action_history_sac_sop_f1_f10",
        )
        parser.add_argument("--seed-stride", type=int, default=1000)
    return parser


def parse_args():
    return add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()


def train(args, tasks, log_dir, seed, algorithm="sac"):
    algorithm = str(algorithm).lower()
    if algorithm not in {"sac", "td3"}:
        raise ValueError("algorithm must be 'sac' or 'td3'.")
    if args.include_task_context:
        raise ValueError(
            "--include-task-context is incompatible with the paper state "
            "s_i=[e0,h_i]."
        )
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = SequentialStaticStepNeuroEAConfigEnv(
        tasks=tasks,
        task_mode=args.task_mode,
        initialization="example",
        seed=seed,
        device=args.device,
        dtype=args.dtype,
        include_task_context=args.include_task_context,
        ela_feature_scale=args.ela_feature_scale,
        ela_objective_index=args.ela_objective_index,
        reward_clip=args.reward_clip,
    )
    initial_observation, _ = env.reset(seed=seed, options={"task_index": 0})
    observation_dim = infer_observation_dim(initial_observation)
    expected_observation_dim = 9 + env.num_params
    if observation_dim != expected_observation_dim:
        raise RuntimeError(
            "Unexpected paper-state dimension: "
            f"expected={expected_observation_dim}, actual={observation_dim}."
        )
    action_dim = int(np.prod(env.action_space.shape))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if algorithm == "sac":
        agent = SACAgent(
            observation_dim,
            action_dim,
            device=device,
            hidden_dims=(128, 128),
            actor_mean_clip=args.actor_output_clip,
        )
    else:
        agent = TD3Agent(
            observation_dim,
            action_dim,
            device=device,
            hidden_dims=args.hidden_dims,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            tau=args.tau,
            exploration_noise=args.exploration_noise,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_delay=args.policy_delay,
            actor_output_clip=args.actor_output_clip,
        )
    replay = ReplayBuffer(observation_dim, action_dim, capacity=100_000)
    shaping = SequentialRewardShapingConfig(
        default_penalty_weight=args.default_penalty_weight,
        default_close_radius=args.default_close_radius,
        default_close_signed_weight=args.default_close_signed_weight,
        final_only_after_episode=args.final_only_after_episode,
    )
    trainer = SequentialStaticSACTrainer(
        env=env,
        agent=agent,
        replay_buffer=replay,
        shaping_config=shaping,
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
        show_episode_progress=True,
        best_checkpoint_metric="eval_fitness",
    )
    eval_seed = 10_000 + seed
    before_eval = evaluate_multitask(env, agent, args.eval_rounds, eval_seed)
    results = trainer.train(
        num_episodes=args.episodes,
        start_seed=seed,
        eval_fn=lambda: evaluate_multitask(env, agent, args.eval_rounds, eval_seed),
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    after_eval = evaluate_multitask(env, agent, args.eval_rounds, eval_seed)
    log_dir = Path(log_dir)
    save_json(log_dir / "before_eval.json", before_eval)
    save_json(log_dir / "after_eval.json", after_eval)
    save_reward_curve_csv(log_dir / "reward_curve.csv", results)
    save_json(
        log_dir / "run_config.json",
        {
            "algorithm": algorithm.upper(),
            "state_design": "ela_parameter_index_one_hot_v1",
            "args": vars(args),
            "seed": seed,
            "problem_names": [task.problem_name for task in tasks],
        },
    )
    plot_path = maybe_plot_curves(log_dir, results)
    fitness = np.asarray([item.final_best_fitness for item in results], dtype=np.float64)
    rewards = np.asarray([item.total_reward for item in results], dtype=np.float64)
    summary = {
        "algorithm": algorithm.upper(),
        "flow": "static_sequential_parameter_index",
        "state_design": "ela + parameter_index_one_hot",
        "episodes": len(results),
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "num_params": env.num_params,
        "reward_first_10_mean": float(rewards[:10].mean()),
        "reward_last_10_mean": float(rewards[-10:].mean()),
        "best_first_10_mean": float(fitness[:10].mean()),
        "best_last_10_mean": float(fitness[-10:].mean()),
        "train_best_fitness": float(fitness.min()),
        "before_eval": before_eval,
        "after_eval": after_eval,
        "best_checkpoint": str(log_dir / "checkpoints" / "best_fitness.pt"),
        "reward_curve_png": None if plot_path is None else str(plot_path),
    }
    save_json(log_dir / "summary.json", summary)
    return summary


def main():
    args = parse_args()
    tasks = [
        ProblemTask(name, (args.population_size, 2, args.dimension, args.max_fe))
        for name in parse_names(args.problem_names)
    ]
    summary = train(args, tasks, args.log_dir, args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
