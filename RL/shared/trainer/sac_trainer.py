"""SAC 训练主循环。

这个 trainer 负责把：
- 环境交互
- replay buffer
- agent.update()
- logging / checkpoint / progress bar

串成一条可直接运行的训练链路。
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from RL.shared.method.observation import flatten_observation

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class SACEpisodeResult:
    """单个 episode 的训练结果快照。"""
    episode: int
    task_name: str | None
    task_index: int | None
    total_reward: float
    num_steps: int
    final_best_fitness: float
    mean_step_reward: float
    mean_normalized_step_reward: float
    num_updates: int = 0
    replay_size: int = 0
    metrics: dict = field(default_factory=dict)


class RunningMeanStd:
    """在线估计均值和方差，用于 reward normalization。"""

    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = float(eps)

    def update(self, value):
        value = float(value)
        delta = value - self.mean
        total = self.count + 1.0
        new_mean = self.mean + delta / total
        m_a = self.var * self.count
        m_b = 0.0
        m2 = m_a + m_b + delta * delta * self.count / total
        self.mean = new_mean
        self.var = max(m2 / total, 1e-8)
        self.count = total

    def normalize(self, value):
        return (float(value) - self.mean) / np.sqrt(self.var + 1e-8)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}


class SACTrainer:
    """最小 SAC trainer，负责采样、回放和评估。"""

    def __init__(
        self,
        env,
        agent,
        replay_buffer,
        batch_size=64,
        start_steps=100,
        updates_per_step=1,
        update_after_steps=None,
        train_freq_steps=1,
        gradient_steps=None,
        normalize_rewards=False,
        reward_clip=None,
        log_dir=None,
        observation_preprocessor=flatten_observation,
        show_progress=True,
        show_episode_progress=True,
        best_checkpoint_metric="eval_reward",
    ):
        # trainer 负责的是“环境交互 + replay buffer + update + logging/save”，
        # 让 agent 本体只关注 SAC 算法更新。
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.batch_size = int(batch_size)
        self.start_steps = int(start_steps)
        self.updates_per_step = int(updates_per_step)
        self.update_after_steps = int(start_steps if update_after_steps is None else update_after_steps)
        self.train_freq_steps = max(int(train_freq_steps), 1)
        self.gradient_steps = int(self.updates_per_step if gradient_steps is None else gradient_steps)
        self.normalize_rewards = bool(normalize_rewards)
        self.reward_clip = None if reward_clip is None else float(reward_clip)
        self.reward_stats_by_task = {} if self.normalize_rewards else None
        self.log_dir = None if log_dir is None else Path(log_dir)
        self.observation_preprocessor = observation_preprocessor
        self.show_progress = bool(show_progress)
        self.show_episode_progress = bool(show_episode_progress)
        if best_checkpoint_metric not in {"eval_reward", "eval_fitness", "train_fitness"}:
            raise ValueError(
                "best_checkpoint_metric must be 'eval_reward', 'eval_fitness', "
                "or 'train_fitness'."
            )
        self.best_checkpoint_metric = best_checkpoint_metric

        self.total_steps = 0
        self.best_eval_metric = None
        self.best_reward_mean = None
        self.best_best_mean = None
        self.best_train_fitness = None
        self._train_log_path = None
        self._eval_log_path = None
        self._summary_path = None

        if self.log_dir is not None:
            self._prepare_log_dir()

    def train(self, num_episodes, start_seed=0, eval_fn=None, eval_every=None, save_every=None):
        # 主训练循环按 episode 推进。
        results = []
        episode_iterable = range(int(num_episodes))
        episode_bar = None
        if self.show_progress and tqdm is not None:
            episode_bar = tqdm(
                episode_iterable,
                desc="Training Episodes",
                unit="episode",
                dynamic_ncols=True,
                file=sys.stdout,
            )
            episode_iterable = episode_bar

        for episode_idx in episode_iterable:
            result = self.run_episode(episode_idx, seed=start_seed + episode_idx)
            results.append(result)
            self._log_train_result(result)

            train_fitness = float(result.final_best_fitness)
            policy_has_acted = self.total_steps > self.start_steps
            if policy_has_acted and np.isfinite(train_fitness) and (
                self.best_train_fitness is None or train_fitness < self.best_train_fitness
            ):
                self.best_train_fitness = train_fitness
                self._save_checkpoint("best_train_fitness.pt")
                if self.best_checkpoint_metric == "train_fitness":
                    self.best_eval_metric = train_fitness
                    self._save_checkpoint("best.pt")

            if episode_bar is not None:
                episode_bar.set_postfix(
                    reward=f"{result.total_reward:.3f}",
                    best=f"{result.final_best_fitness:.3e}",
                    updates=result.num_updates,
                    replay=result.replay_size,
                )

            if save_every is not None and (episode_idx + 1) % int(save_every) == 0:
                self._save_checkpoint(f"episode_{episode_idx + 1}.pt")
                self._save_checkpoint("latest.pt")

            if eval_fn is not None and eval_every is not None and (episode_idx + 1) % int(eval_every) == 0:
                evaluation = eval_fn()
                self._log_eval_result(episode_idx + 1, evaluation)
                reward_mean = float(evaluation.get("reward_mean", float("-inf")))
                best_mean = float(evaluation.get("best_mean", float("inf")))

                if self.best_reward_mean is None or reward_mean > self.best_reward_mean:
                    self.best_reward_mean = reward_mean
                    self._save_checkpoint("best_reward.pt")
                    if self.best_checkpoint_metric == "eval_reward":
                        self.best_eval_metric = reward_mean
                        self._save_checkpoint("best.pt")

                if self.best_best_mean is None or best_mean < self.best_best_mean:
                    self.best_best_mean = best_mean
                    self._save_checkpoint("best_fitness.pt")
                    if self.best_checkpoint_metric == "eval_fitness":
                        self.best_eval_metric = best_mean
                        self._save_checkpoint("best.pt")

        if episode_bar is not None:
            episode_bar.close()

        if self.log_dir is not None:
            self._write_summary(results)
        return results

    def run_episode(self, episode_idx, seed=None):
        # 一个 episode 内，step-wise EA 的每一代都对应一次 `env.step(action)`。
        observation, info = self.env.reset(seed=seed)
        processed_observation = self.observation_preprocessor(observation)
        task_name = self._extract_task_name(info)
        task_index = self._extract_task_index(info)
        reward_task_key = self._reward_task_key(info)
        total_reward = 0.0
        num_steps = 0
        num_updates = 0
        update_metrics = None
        normalized_rewards = []
        raw_rewards = []

        step_bar = self._make_episode_progress_bar(episode_idx, info)

        while True:
            if self.total_steps < self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.act(processed_observation, deterministic=False)

            next_observation, reward, terminated, truncated, info = self.env.step(action)
            processed_next_observation = self.observation_preprocessor(next_observation)
            done = bool(terminated or truncated)

            raw_rewards.append(float(reward))
            buffer_reward = float(reward)
            if self.normalize_rewards:
                # 注意：这里的 normalization 是 trainer 层的再加工，
                # 不会改变日志里记录的原始 episode reward。
                reward_stats = self._get_reward_stats(reward_task_key)
                reward_stats.update(buffer_reward)
                buffer_reward = reward_stats.normalize(buffer_reward)
                if self.reward_clip is not None:
                    buffer_reward = float(np.clip(buffer_reward, -self.reward_clip, self.reward_clip))
            normalized_rewards.append(float(buffer_reward))

            self.replay_buffer.add(processed_observation, action, buffer_reward, processed_next_observation, done)
            processed_observation = processed_next_observation
            total_reward += float(reward)
            num_steps += 1
            self.total_steps += 1

            if self._should_update():
                for _ in range(self.gradient_steps):
                    update_metrics = self.agent.update(self.replay_buffer.sample(self.batch_size))
                    num_updates += 1

            if step_bar is not None:
                step_bar.update(1)
                step_bar.set_postfix(
                    reward=f"{total_reward:.3f}",
                    best=f"{float(info.get('best_fitness', np.nan)):.3e}",
                    updates=num_updates,
                )

            if done:
                break

        if step_bar is not None:
            step_bar.close()

        final_best = float(info.get("best_fitness", np.nan))
        metrics = {} if update_metrics is None else update_metrics.__dict__.copy()
        return SACEpisodeResult(
            episode=int(episode_idx),
            task_name=task_name,
            task_index=task_index,
            total_reward=float(total_reward),
            num_steps=int(num_steps),
            final_best_fitness=final_best,
            mean_step_reward=float(np.mean(raw_rewards)) if raw_rewards else 0.0,
            mean_normalized_step_reward=float(np.mean(normalized_rewards)) if normalized_rewards else 0.0,
            num_updates=int(num_updates),
            replay_size=len(self.replay_buffer),
            metrics=metrics,
        )

    def evaluate(self, num_episodes=5, start_seed=10_000):
        # 评估时使用 deterministic action，避免把训练探索噪声混进去。
        rewards = []
        best_values = []
        for episode_idx in range(int(num_episodes)):
            observation, info = self.env.reset(seed=start_seed + episode_idx)
            processed_observation = self.observation_preprocessor(observation)
            total_reward = 0.0

            while True:
                action = self.agent.act(processed_observation, deterministic=True)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                processed_observation = self.observation_preprocessor(next_observation)
                total_reward += float(reward)
                if terminated or truncated:
                    break

            rewards.append(total_reward)
            best_values.append(float(info.get("best_fitness", np.nan)))

        return {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "best_mean": float(np.mean(best_values)),
            "best_std": float(np.std(best_values)),
        }

    def _prepare_log_dir(self):
        # 把所有实验产物固定落到一个目录下，方便后续对比实验。
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "checkpoints").mkdir(exist_ok=True)
        self._train_log_path = self.log_dir / "train_history.csv"
        self._eval_log_path = self.log_dir / "eval_history.csv"
        self._summary_path = self.log_dir / "summary.json"

        with self._train_log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "episode",
                    "task_name",
                    "task_index",
                    "total_reward",
                    "num_steps",
                    "final_best_fitness",
                    "mean_step_reward",
                    "mean_normalized_step_reward",
                    "num_updates",
                    "replay_size",
                    "actor_loss",
                    "critic_loss",
                    "alpha_loss",
                    "alpha",
                    "q_mean",
                    "log_prob",
                    "action_saturation",
                ]
            )

        with self._eval_log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode", "reward_mean", "reward_std", "best_mean", "best_std"])

    def _log_train_result(self, result):
        if self._train_log_path is None:
            return
        metrics = result.metrics
        with self._train_log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    result.episode,
                    result.task_name,
                    result.task_index,
                    result.total_reward,
                    result.num_steps,
                    result.final_best_fitness,
                    result.mean_step_reward,
                    result.mean_normalized_step_reward,
                    result.num_updates,
                    result.replay_size,
                    metrics.get("actor_loss"),
                    metrics.get("critic_loss"),
                    metrics.get("alpha_loss"),
                    metrics.get("alpha"),
                    metrics.get("q_mean"),
                    metrics.get("log_prob"),
                    metrics.get("action_saturation"),
                ]
            )

    def _log_eval_result(self, episode, evaluation):
        if self._eval_log_path is None:
            return
        with self._eval_log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    int(episode),
                    evaluation.get("reward_mean"),
                    evaluation.get("reward_std"),
                    evaluation.get("best_mean"),
                    evaluation.get("best_std"),
                ]
            )

    def _save_checkpoint(self, filename):
        # checkpoint 里除了 agent 参数，也会保存 reward normalization 的统计量。
        if self.log_dir is None:
            return
        checkpoint = {
            "total_steps": self.total_steps,
            "best_eval_metric": self.best_eval_metric,
            "best_reward_mean": self.best_reward_mean,
            "best_best_mean": self.best_best_mean,
            "best_train_fitness": self.best_train_fitness,
            "best_checkpoint_metric": self.best_checkpoint_metric,
            "reward_stats": self._single_reward_stats_state(),
            "reward_stats_by_task": self._reward_stats_by_task_state(),
            "agent": self.agent.state_dict(),
        }
        path = self.log_dir / "checkpoints" / filename
        import torch

        torch.save(checkpoint, path)

    def _write_summary(self, results):
        if self._summary_path is None or not results:
            return

        rewards = np.asarray([result.total_reward for result in results], dtype=np.float64)
        best_values = np.asarray([result.final_best_fitness for result in results], dtype=np.float64)
        summary = {
            "episodes": len(results),
            "total_steps": self.total_steps,
            "reward_first_10_mean": float(np.mean(rewards[: min(10, len(rewards))])),
            "reward_last_10_mean": float(np.mean(rewards[-min(10, len(rewards)) :])),
            "best_first_10_mean": float(np.mean(best_values[: min(10, len(best_values))])),
            "best_last_10_mean": float(np.mean(best_values[-min(10, len(best_values)) :])),
            "best_eval_metric": self.best_eval_metric,
            "best_reward_mean": self.best_reward_mean,
            "best_best_mean": self.best_best_mean,
            "best_train_fitness": self.best_train_fitness,
            "best_checkpoint_metric": self.best_checkpoint_metric,
        }
        if self.reward_stats_by_task is not None:
            summary["reward_stats"] = self._single_reward_stats_state()
            summary["reward_stats_by_task"] = self._reward_stats_by_task_state()

        with self._summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    def _extract_task_name(self, info):
        task = info.get("task")
        if task is not None and hasattr(task, "problem_name"):
            return str(task.problem_name)

        state = info.get("state", {})
        if state.get("task_name") is not None:
            return str(state["task_name"])
        return None

    def _extract_task_index(self, info):
        if info.get("task_index") is not None:
            return int(info["task_index"])

        state = info.get("state", {})
        if state.get("task_index") is not None:
            return int(state["task_index"])
        return None

    def _reward_task_key(self, info):
        task_name = self._extract_task_name(info)
        if task_name is not None:
            return task_name

        task_index = self._extract_task_index(info)
        if task_index is not None:
            return f"task_{task_index}"
        return "default"

    def _get_reward_stats(self, task_key):
        if self.reward_stats_by_task is None:
            return None
        if task_key not in self.reward_stats_by_task:
            self.reward_stats_by_task[task_key] = RunningMeanStd()
        return self.reward_stats_by_task[task_key]

    def _reward_stats_by_task_state(self):
        if self.reward_stats_by_task is None:
            return None
        return {
            str(task_key): stats.state_dict()
            for task_key, stats in self.reward_stats_by_task.items()
        }

    def _single_reward_stats_state(self):
        stats_by_task = self._reward_stats_by_task_state()
        if not stats_by_task:
            return None
        if len(stats_by_task) == 1:
            return next(iter(stats_by_task.values()))
        return None

    def _should_update(self):
        if len(self.replay_buffer) < self.batch_size:
            return False
        if self.total_steps < self.update_after_steps:
            return False
        return ((self.total_steps - self.update_after_steps) % self.train_freq_steps) == 0

    def _infer_episode_total_steps(self, info):
        """
        估计一个 episode 的总 step 数，用于进度条。

        对当前 StepNeuroEA 问题配置 `(N, M, D, max_fe)`，
        一般初始化会先消耗 `N` 次评估，后面每代再大致消耗 `N` 次。
        因此 episode step 数可近似估计为：
        `(max_fe - N) / N`
        """
        task = info.get("task")
        if task is not None and hasattr(task, "problem_config"):
            config = tuple(task.problem_config)
        else:
            state = info.get("state", {})
            config = tuple(state.get("task_config", ()))

        if len(config) >= 4:
            population_size = int(config[0])
            max_fe = int(config[3])
            if population_size > 0 and max_fe > population_size:
                return max(int(math.ceil((max_fe - population_size) / float(population_size))), 1)
        return None

    def _make_episode_progress_bar(self, episode_idx, info):
        if not self.show_episode_progress or tqdm is None:
            return None
        total = self._infer_episode_total_steps(info)
        desc = f"Episode {int(episode_idx) + 1}"
        return tqdm(
            total=total,
            desc=desc,
            unit="step",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
