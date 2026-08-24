"""Trainer for sequential static parameter configuration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from RL.shared.trainer.sac_trainer import SACEpisodeResult, SACTrainer


@dataclass
class SequentialRewardShapingConfig:
    """Reward backfill settings inspired by the old static-config code."""

    default_penalty_weight: float = 0.0
    default_close_radius: float = 0.6
    default_close_signed_weight: float = 0.0
    final_only_after_episode: int | None = None


class SequentialStaticSACTrainer(SACTrainer):
    """Collect a whole parameter sequence, then backfill final rewards."""

    def __init__(self, *args, shaping_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.shaping_config = shaping_config or SequentialRewardShapingConfig()

    def run_episode(self, episode_idx, seed=None):
        observation, info = self.env.reset(seed=seed)
        processed_observation = self.observation_preprocessor(observation)
        task_name = self._extract_task_name(info)
        task_index = self._extract_task_index(info)
        reward_task_key = self._reward_task_key(info)
        transitions = []
        total_reward = 0.0
        update_metrics = None

        step_bar = self._make_episode_progress_bar(episode_idx, info)

        while True:
            if self.total_steps < self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.act(processed_observation, deterministic=False)

            next_observation, reward, terminated, truncated, info = self.env.step(action)
            processed_next_observation = self.observation_preprocessor(next_observation)
            done = bool(terminated or truncated)
            transitions.append((processed_observation, action, float(reward), processed_next_observation, done))

            processed_observation = processed_next_observation
            total_reward += float(reward)
            self.total_steps += 1

            if step_bar is not None:
                step_bar.update(1)
                step_bar.set_postfix(
                    reward=f"{total_reward:.3f}",
                    best=f"{float(info.get('best_fitness', np.nan)):.3e}",
                    replay=len(self.replay_buffer),
                )

            if done:
                break

        final_reward = float(transitions[-1][2]) if transitions else 0.0
        default_params = np.asarray(info.get("default_params", np.empty(0)), dtype=np.float32).reshape(-1)
        shaped_rewards = self._shape_episode_rewards(
            episode_idx=episode_idx,
            final_reward=final_reward,
            transitions=transitions,
            default_params=default_params,
        )

        normalized_rewards = []
        num_updates = 0
        for index, (transition, shaped_reward) in enumerate(zip(transitions, shaped_rewards)):
            current_observation, action, _, next_observation, done = transition
            buffer_reward = float(shaped_reward)
            if self.normalize_rewards:
                reward_stats = self._get_reward_stats(reward_task_key)
                reward_stats.update(buffer_reward)
                buffer_reward = reward_stats.normalize(buffer_reward)
                if self.reward_clip is not None:
                    buffer_reward = float(np.clip(buffer_reward, -self.reward_clip, self.reward_clip))
            normalized_rewards.append(float(buffer_reward))
            self.replay_buffer.add(current_observation, action, buffer_reward, next_observation, done)

            if self._should_update():
                for _ in range(self.gradient_steps):
                    update_metrics = self.agent.update(self.replay_buffer.sample(self.batch_size))
                    num_updates += 1

        if step_bar is not None:
            step_bar.close()

        final_best = float(info.get("best_fitness", np.nan))
        metrics = {} if update_metrics is None else update_metrics.__dict__.copy()
        metrics["final_reward"] = final_reward
        metrics["mean_shaped_reward"] = float(np.mean(shaped_rewards)) if shaped_rewards else 0.0

        return SACEpisodeResult(
            episode=int(episode_idx),
            task_name=task_name,
            task_index=task_index,
            total_reward=float(final_reward),
            num_steps=int(len(transitions)),
            final_best_fitness=final_best,
            mean_step_reward=float(np.mean(shaped_rewards)) if shaped_rewards else 0.0,
            mean_normalized_step_reward=float(np.mean(normalized_rewards)) if normalized_rewards else 0.0,
            num_updates=int(num_updates),
            replay_size=len(self.replay_buffer),
            metrics=metrics,
        )

    def _shape_episode_rewards(self, episode_idx, final_reward, transitions, default_params):
        config = self.shaping_config
        if config.final_only_after_episode is not None and int(episode_idx) > int(config.final_only_after_episode):
            return [float(final_reward)] * len(transitions)

        rewards = []
        for index, (_, action, _, _, _) in enumerate(transitions):
            reward = float(final_reward)
            if index < default_params.shape[0]:
                action_scalar = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
                distance = abs(action_scalar - float(default_params[index]))
                if config.default_close_signed_weight > 0 and distance < config.default_close_radius:
                    reward -= config.default_close_signed_weight * (action_scalar - float(default_params[index]))
                elif config.default_penalty_weight > 0:
                    reward -= config.default_penalty_weight * distance
            rewards.append(float(reward))
        return rewards
