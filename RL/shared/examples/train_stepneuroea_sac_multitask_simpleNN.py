"""
恢复版：加入 Transformer 之前的多任务 StepNeuroEA + MLP-SAC 训练脚本。

这份脚本尽量保持之前版本的行为：
- observation: `summary + control + task`
- agent: `SACAgent`
- replay buffer: `ReplayBuffer`
- 默认训练节奏接近之前版本：
  - replay buffer 一旦积累到 `batch_size`
  - 之后每一步都执行 `updates_per_step` 次更新

说明：
- 当前 trainer 已经支持更细的“先采样、后更新”节奏控制
- 但为了保留旧版本效果，这里显式把更新条件设置回旧风格
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
from RL.shared.env.ela_observation import ELAObservationBuilder
from RL.shared.env.stepneuroea_env import StepNeuroEAEnv
from RL.shared.env.stepwise_ea_env import LogGapReward
from RL.shared.method.observation import flatten_observation, infer_observation_dim
from RL.shared.method.replay_buffer import ReplayBuffer
from RL.shared.method.sac_agent import SACAgent
from RL.shared.trainer.sac_trainer import SACTrainer


def parse_csv_strings(text: str):
    return [item.strip() for item in text.split(",") if item.strip()]


def evaluate_each_task(env, agent, num_eval_rounds=1, base_seed=10_000):
    """逐个任务做 deterministic 评估。"""
    results = []
    for task_index, task in enumerate(env.tasks):
        rewards = []
        best_values = []
        for eval_round in range(num_eval_rounds):
            observation, info = env.reset(seed=base_seed + eval_round, options={"task_index": task_index})
            flat_observation = flatten_observation(observation)
            total_reward = 0.0

            while True:
                action = agent.act(flat_observation, deterministic=True)
                next_observation, reward, terminated, truncated, info = env.step(action)
                flat_observation = flatten_observation(next_observation)
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
    """汇总所有任务的平均 reward / best。"""
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
    """把结果落成 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def save_reward_curve_csv(path, results):
    """把 episode reward / best 曲线保存成 CSV。"""
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
    """如果本地安装了 matplotlib，就额外画一张训练曲线图。"""
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
    parser = argparse.ArgumentParser(description="Pre-transformer multi-task StepNeuroEA SAC.")
    parser.add_argument("--episodes", type=int, default=10000, help="训练 episode 数。")
    parser.add_argument("--problem-names", type=str, default="SOP_F1,SOP_F2,SOP_F9", help="逗号分隔的训练问题列表。")
    parser.add_argument("--dimension", type=int, default=10, help="所有任务共享的决策维度 D。")
    parser.add_argument("--population-size", type=int, default=100, help="所有任务共享的种群规模 N。")
    parser.add_argument("--max-fe", type=int, default=10000, help="每个任务的最大评价次数。")
    parser.add_argument("--seed", type=int, default=0, help="随机种子。")
    parser.add_argument("--device", type=str, default="cpu", help="训练设备，例如 cpu / cuda / auto。")
    parser.add_argument("--dtype", type=str, default="float64", help="环境与算法的 dtype。")
    parser.add_argument("--task-mode", type=str, default="cycle", choices=["cycle", "random"], help="多任务切换方式。")
    parser.add_argument("--eval-every", type=int, default=10, help="每隔多少个 episode 做一次评估。")
    parser.add_argument("--eval-rounds", type=int, default=1, help="每个任务评估多少轮。")
    parser.add_argument("--save-every", type=int, default=1000, help="每隔多少个 episode 保存一次 checkpoint。")
    parser.add_argument("--batch-size", type=int, default=64, help="SAC batch size。")
    parser.add_argument("--start-steps", type=int, default=128, help="开始训练前先随机采样多少步。")
    parser.add_argument("--updates-per-step", type=int, default=1, help="旧风格设置下，每步执行多少次更新。")
    parser.add_argument("--log-dir", type=str, default="RL/runs/stepneuroea_sac_multitask_d10_LogReward_2", help="输出目录。")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = [
        ProblemTask(problem_name, (args.population_size, 1, args.dimension, args.max_fe))
        for problem_name in parse_csv_strings(args.problem_names)
    ]

    observation_builder = ELAObservationBuilder(include_population=False)
    env = StepNeuroEAEnv(
        tasks=tasks,
        task_mode=args.task_mode,
        initialization="example",
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        observation_builder=observation_builder,
        reward_builder=LogGapReward(),
    )

    initial_observation, _ = env.reset(seed=args.seed, options={"task_index": 0})
    observation_dim = infer_observation_dim(initial_observation)
    action_dim = int(np.prod(env.action_space.shape))
    agent_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    agent = SACAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        device=agent_device,
        hidden_dims=(128, 128),
    )
    replay_buffer = ReplayBuffer(observation_dim=observation_dim, action_dim=action_dim, capacity=100_000)
    trainer = SACTrainer(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        updates_per_step=args.updates_per_step,
        # 旧版本的关键行为：
        # - 只要 replay buffer 达到 batch_size，就允许开始更新
        # - 之后每一步都更新一次
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
    summary = {
        "episodes": len(results),
        "reward_first_10_mean": float(np.mean([result.total_reward for result in results[:10]])) if results else 0.0,
        "reward_last_10_mean": float(np.mean([result.total_reward for result in results[-10:]])) if results else 0.0,
        "best_first_10_mean": float(np.mean([result.final_best_fitness for result in results[:10]])) if results else 0.0,
        "best_last_10_mean": float(np.mean([result.final_best_fitness for result in results[-10:]])) if results else 0.0,
        "before_eval": before_eval,
        "after_eval": after_eval,
        "reward_curve_png": None if plot_path is None else str(plot_path),
    }
    save_json(Path(args.log_dir) / "summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# 命令行示例：
# conda run -n pyRL python RL/examples/train_stepneuroea_sac_multitask_simpleNN.py \
#   --episodes 100 \
#   --problem-names SOP_F1,SOP_F2,SOP_F9 \
#   --dimension 10 \
#   --population-size 100 \
#   --max-fe 10000
