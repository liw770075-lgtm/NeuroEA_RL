"""Standard TD3 agent for continuous NeuroEA parameter configuration."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RL.shared.method.base_agent import BaseAgent
from RL.shared.model.mlp import MLP
from RL.shared.model.sac import TwinQNetwork


class DeterministicPolicy(nn.Module):
    """Deterministic policy whose output is bounded to [-1, 1]."""

    def __init__(self, observation_dim, action_dim, hidden_dims=(256, 256), output_clip=None):
        super().__init__()
        self.output_clip = None if output_clip is None else float(output_clip)
        self.network = MLP(observation_dim, hidden_dims, action_dim)

    def forward(self, observation):
        output = self.network(observation)
        if self.output_clip is not None:
            output = torch.clamp(output, -self.output_clip, self.output_clip)
        return torch.tanh(output)


@dataclass
class TD3UpdateMetrics:
    actor_loss: float | None
    critic_loss: float
    q_mean: float
    policy_updated: bool
    action_saturation: float


class TD3Agent(BaseAgent):
    """Twin Delayed DDPG with target policy smoothing."""

    def __init__(
        self,
        observation_dim,
        action_dim,
        device="cpu",
        hidden_dims=(256, 256),
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        exploration_noise=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        actor_output_clip=None,
    ):
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.exploration_noise = float(exploration_noise)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = max(int(policy_delay), 1)
        self.actor_output_clip = None if actor_output_clip is None else float(actor_output_clip)
        self.update_step = 0

        self.actor = DeterministicPolicy(
            self.observation_dim,
            self.action_dim,
            hidden_dims=hidden_dims,
            output_clip=self.actor_output_clip,
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.critic = TwinQNetwork(
            self.observation_dim, self.action_dim, hidden_dims=hidden_dims
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        for network in (self.target_actor, self.target_critic):
            for parameter in network.parameters():
                parameter.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def act(self, observation, deterministic=False):
        observation_tensor = self._to_tensor(observation).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(observation_tensor).squeeze(0).cpu().numpy()
        if not deterministic and self.exploration_noise > 0.0:
            action = action + np.random.normal(
                loc=0.0, scale=self.exploration_noise, size=self.action_dim
            )
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, batch):
        observations = self._to_tensor(batch["observations"])
        actions = self._to_tensor(batch["actions"])
        rewards = self._to_tensor(batch["rewards"])
        next_observations = self._to_tensor(batch["next_observations"])
        dones = self._to_tensor(batch["dones"])
        self.update_step += 1

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(next_observations) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.target_critic(next_observations, next_actions)
            target_q = torch.min(target_q1, target_q2)
            td_target = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(observations, actions)
        critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = None
        policy_updated = self.update_step % self.policy_delay == 0
        if policy_updated:
            policy_actions = self.actor(observations)
            actor_loss = -self.critic.q1(
                torch.cat([observations, policy_actions], dim=-1)
            ).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())

            self._soft_update(self.actor, self.target_actor, self.tau)
            self._soft_update(self.critic, self.target_critic, self.tau)

        return TD3UpdateMetrics(
            actor_loss=actor_loss_value,
            critic_loss=float(critic_loss.item()),
            q_mean=float(torch.min(current_q1, current_q2).mean().item()),
            policy_updated=policy_updated,
            action_saturation=float((self.actor(observations).abs() > 0.99).float().mean().item()),
        )

    def state_dict(self):
        return {
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "tau": self.tau,
            "exploration_noise": self.exploration_noise,
            "policy_noise": self.policy_noise,
            "noise_clip": self.noise_clip,
            "policy_delay": self.policy_delay,
            "actor_output_clip": self.actor_output_clip,
            "update_step": self.update_step,
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.target_actor.load_state_dict(state_dict["target_actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.update_step = int(state_dict.get("update_step", 0))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def _to_tensor(self, value):
        if torch.is_tensor(value):
            return value.to(self.device, dtype=torch.float32)
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    @staticmethod
    def _soft_update(source, target, tau):
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.data.mul_(1.0 - tau).add_(tau * source_parameter.data)
