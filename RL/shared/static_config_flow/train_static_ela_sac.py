"""Train SAC for sequential static StepNeuroEA parameter configuration.

The agent configures one parameter per step. After all parameters are chosen,
StepNeuroEA runs to completion with the fixed vector.
"""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import argparse
import csv
import json

import numpy as np
import torch

from RL.shared.env.problem_utils import ProblemTask
from RL.shared.method.observation import flatten_observation, infer_observation_dim
from RL.shared.method.replay_buffer import ReplayBuffer
from RL.shared.method.sac_agent import SACAgent
from RL.shared.static_config_flow.sequential_env import SequentialStaticStepNeuroEAConfigEnv
from RL.shared.static_config_flow.sequential_trainer import (
    SequentialRewardShapingConfig,
    SequentialStaticSACTrainer,
)


def parse_csv_strings(text: str):
    return [item.strip() for item in text.split(",") if item.strip()]


def evaluate_each_task(env, agent, num_eval_rounds=1, base_seed=10_000):
    results = []
    for task_index, task in enumerate(env.tasks):
        rewards = []
        best_values = []
        for eval_round in range(num_eval_rounds):
            observation, info = env.reset(seed=base_seed + eval_round, options={"task_index": task_index})
            total_reward = 0.0
            while True:
                action = agent.act(flatten_observation(observation), deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            rewards.append(total_reward)
            best_values.append(float(info["best_fitness"]))

        results.append(
            {
                "task_name": task.problem_name,
                "task_config": task.problem_config,
                "reward_mean": float(np.mean(rewards)),
                "best_mean": float(np.mean(best_values)),
            }
        )
    return results


def evaluate_multitask(env, agent, num_eval_rounds=1, base_seed=10_000):
    per_task = evaluate_each_task(env, agent, num_eval_rounds=num_eval_rounds, base_seed=base_seed)
    reward_values = np.asarray([item["reward_mean"] for item in per_task], dtype=np.float64)
    best_values = np.asarray([item["best_mean"] for item in per_task], dtype=np.float64)
    return {
        "reward_mean": float(np.mean(reward_values)),
        "reward_std": float(np.std(reward_values)),
        "best_mean": float(np.mean(best_values)),
        "best_std": float(np.std(best_values)),
        "per_task": per_task,
    }


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def save_reward_curve_csv(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rewards = np.asarray([result.total_reward for result in results], dtype=np.float64)
    best_values = np.asarray([result.final_best_fitness for result in results], dtype=np.float64)
    window = min(10, len(results))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "reward", "best_fitness", "reward_ma10", "best_fitness_ma10"])
        for index, (reward, best_value) in enumerate(zip(rewards, best_values), start=1):
            left = max(0, index - window)
            writer.writerow(
                [
                    index,
                    reward,
                    best_value,
                    float(np.mean(rewards[left:index])),
                    float(np.mean(best_values[left:index])),
                ]
            )


def maybe_plot_curves(log_dir, results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    log_dir = Path(log_dir)
    rewards = np.asarray([result.total_reward for result in results], dtype=np.float64)
    best_values = np.asarray([result.final_best_fitness for result in results], dtype=np.float64)
    episodes = np.arange(1, len(results) + 1)
    window = min(10, len(results))

    if window > 1:
        kernel = np.ones(window, dtype=np.float64) / window
        smooth_rewards = np.convolve(rewards, kernel, mode="valid")
        smooth_best = np.convolve(best_values, kernel, mode="valid")
        smooth_x = np.arange(window, len(results) + 1)
    else:
        smooth_rewards = rewards
        smooth_best = best_values
        smooth_x = episodes

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(episodes, rewards, alpha=0.35, label="episode reward")
    axes[0].plot(smooth_x, smooth_rewards, linewidth=2.0, label=f"{window}-episode mean")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(episodes, best_values, alpha=0.35, label="final best fitness")
    axes[1].plot(smooth_x, smooth_best, linewidth=2.0, label=f"{window}-episode mean")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Best Fitness")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    output_path = log_dir / "reward_curve.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential static StepNeuroEA SAC configuration with ELA state.")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes.")
    parser.add_argument("--problem-names", type=str, default="SOP_F1", help="Comma-separated training problems.")
    parser.add_argument("--dimension", type=int, default=10, help="Decision dimension D.")
    parser.add_argument("--population-size", type=int, default=100, help="Population size N.")
    parser.add_argument("--max-fe", type=int, default=10000, help="Maximum function evaluations per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Environment device: cpu / cuda / auto.")
    parser.add_argument("--dtype", type=str, default="float64", help="Environment dtype.")
    parser.add_argument("--task-mode", type=str, default="cycle", choices=["cycle", "random"], help="Task sampling mode.")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate every N episodes.")
    parser.add_argument("--eval-rounds", type=int, default=1, help="Evaluation rounds per task.")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N episodes.")
    parser.add_argument("--batch-size", type=int, default=960, help="SAC batch size.")
    parser.add_argument("--start-steps", type=int, default=128, help="Random exploration episodes before policy actions.")
    parser.add_argument("--updates-per-step", type=int, default=1, help="Gradient updates per stored transition.")
    parser.add_argument("--ela-feature-scale", type=float, default=100.0, help="Divide ELA features by this scale.")
    parser.add_argument("--ela-objective-index", type=int, default=0, help="Objective column used for ELA.")
    parser.add_argument("--reward-clip", type=float, default=None, help="Optional clip for environment reward.")
    parser.add_argument(
        "--default-penalty-weight",
        type=float,
        default=0.0,
        help="Penalty weight for deviating from the default normalized parameter during reward backfill.",
    )
    parser.add_argument(
        "--default-close-radius",
        type=float,
        default=0.6,
        help="Distance threshold used by the optional old-style close-to-default reward shaping.",
    )
    parser.add_argument(
        "--default-close-signed-weight",
        type=float,
        default=0.0,
        help="Old-style signed close-to-default shaping weight. Set 3.0 to mimic the old script more closely.",
    )
    parser.add_argument(
        "--final-only-after-episode",
        type=int,
        default=None,
        help="After this episode index, backfill only final reward without default-parameter shaping.",
    )
    parser.add_argument(
        "--include-task-context",
        action="store_true",
        help="Include task one-hot and normalized problem config in the state.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="RL/runs/static_sequential_stepneuroea_sac_ela_sop_f1_d10",
        help="Output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = [
        ProblemTask(problem_name, (args.population_size, 2, args.dimension, args.max_fe))
        for problem_name in parse_csv_strings(args.problem_names)
    ]

    env = SequentialStaticStepNeuroEAConfigEnv(
        tasks=tasks,
        task_mode=args.task_mode,
        initialization="example",
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        include_task_context=args.include_task_context,
        ela_feature_scale=args.ela_feature_scale,
        ela_objective_index=args.ela_objective_index,
        reward_clip=args.reward_clip,
    )

    initial_observation, _ = env.reset(seed=args.seed, options={"task_index": 0})
    observation_dim = infer_observation_dim(initial_observation)
    action_dim = int(np.prod(env.action_space.shape))
    agent_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )

    agent = SACAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        device=agent_device,
        hidden_dims=(128, 128),
    )
    replay_buffer = ReplayBuffer(observation_dim=observation_dim, action_dim=action_dim, capacity=100_000)
    shaping_config = SequentialRewardShapingConfig(
        default_penalty_weight=args.default_penalty_weight,
        default_close_radius=args.default_close_radius,
        default_close_signed_weight=args.default_close_signed_weight,
        final_only_after_episode=args.final_only_after_episode,
    )
    trainer = SequentialStaticSACTrainer(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        shaping_config=shaping_config,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        updates_per_step=args.updates_per_step,
        update_after_steps=args.batch_size,
        train_freq_steps=1,
        gradient_steps=args.updates_per_step,
        normalize_rewards=True,
        reward_clip=10.0,
        log_dir=args.log_dir,
        observation_preprocessor=flatten_observation,
        show_progress=True,
        show_episode_progress=True,
    )

    eval_seed = 10_000
    before_eval = evaluate_multitask(env, agent, num_eval_rounds=args.eval_rounds, base_seed=eval_seed)
    results = trainer.train(
        num_episodes=args.episodes,
        start_seed=args.seed,
        eval_fn=lambda: evaluate_multitask(env, agent, num_eval_rounds=args.eval_rounds, base_seed=eval_seed),
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    after_eval = evaluate_multitask(env, agent, num_eval_rounds=args.eval_rounds, base_seed=eval_seed)

    save_json(Path(args.log_dir) / "before_eval.json", before_eval)
    save_json(Path(args.log_dir) / "after_eval.json", after_eval)
    save_reward_curve_csv(Path(args.log_dir) / "reward_curve.csv", results)
    save_json(Path(args.log_dir) / "run_config.json", {"args": vars(args)})

    plot_path = maybe_plot_curves(args.log_dir, results)
    final_best_values = np.asarray([result.final_best_fitness for result in results], dtype=np.float64)
    total_rewards = np.asarray([result.total_reward for result in results], dtype=np.float64)
    best_episode_index = int(np.argmin(final_best_values)) if len(final_best_values) > 0 else None
    best_reward_episode_index = int(np.argmax(total_rewards)) if len(total_rewards) > 0 else None
    summary = {
        "episodes": len(results),
        "flow": "static_sequential_config",
        "observation_keys": ["static_features", "parameter_index_one_hot"],
        "observation_dim": int(observation_dim),
        "action_dim": int(action_dim),
        "reward": "log(initial_gap) - log(final_gap)",
        "steps_per_episode": int(env.num_params),
        "default_penalty_weight": float(args.default_penalty_weight),
        "default_close_signed_weight": float(args.default_close_signed_weight),
        "reward_first_10_mean": float(np.mean([result.total_reward for result in results[:10]])) if results else 0.0,
        "reward_last_10_mean": float(np.mean([result.total_reward for result in results[-10:]])) if results else 0.0,
        "best_first_10_mean": float(np.mean([result.final_best_fitness for result in results[:10]])) if results else 0.0,
        "best_last_10_mean": float(np.mean([result.final_best_fitness for result in results[-10:]])) if results else 0.0,
        "train_best_fitness_min": float(final_best_values[best_episode_index]) if best_episode_index is not None else None,
        "train_best_fitness_episode": int(results[best_episode_index].episode) if best_episode_index is not None else None,
        "train_best_reward_max": float(total_rewards[best_reward_episode_index]) if best_reward_episode_index is not None else None,
        "train_best_reward_episode": int(results[best_reward_episode_index].episode) if best_reward_episode_index is not None else None,
        "before_eval": before_eval,
        "after_eval": after_eval,
        "reward_curve_png": None if plot_path is None else str(plot_path),
    }
    save_json(Path(args.log_dir) / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
