"""simpleNN 版本的 SAC agent。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from RL.shared.method.base_agent import BaseAgent
from RL.shared.model.sac import SquashedGaussianPolicy, TwinQNetwork


@dataclass
class SACUpdateMetrics:
    """一次 SAC 更新后常用的诊断指标。"""
    actor_loss: float
    critic_loss: float
    alpha_loss: float
    alpha: float
    q_mean: float
    log_prob: float
    action_saturation: float


class SACAgent(BaseAgent):
    """最小可训练 SAC，动作范围固定在 [-1, 1]。"""

    def __init__(
        self,
        observation_dim,
        action_dim,
        device="cpu",
        hidden_dims=(256, 256),
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        init_alpha=0.2,
        target_entropy=None,
        actor_mean_clip=None,
    ):
        # SAC 的 actor / critic 都是标准的 MLP 版本。
        # 当前项目先优先保证“和 step-wise EA 环境能稳定打通”，
        # 所以这里先不引入更复杂的编码器结构。
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)

        self.actor_mean_clip = None if actor_mean_clip is None else float(actor_mean_clip)
        self.actor = SquashedGaussianPolicy(
            self.observation_dim,
            self.action_dim,
            hidden_dims=hidden_dims,
            mean_clip=self.actor_mean_clip,
        ).to(self.device)
        self.critic = TwinQNetwork(self.observation_dim, self.action_dim, hidden_dims=hidden_dims).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(init_alpha), device=self.device, dtype=torch.float32, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -float(self.action_dim) if target_entropy is None else float(target_entropy)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, observation, deterministic=False):
        # 训练时用随机采样动作，评估时通常用均值动作。
        observation_tensor = self._to_tensor(observation).unsqueeze(0)
        with torch.no_grad():
            action, _, mean_action = self.actor.sample(observation_tensor, deterministic=deterministic)
            chosen = mean_action if deterministic else action
        return chosen.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, batch):
        # SAC 的一次更新分三步：
        # 1. 更新 critic
        # 2. 更新 actor
        # 3. 更新 temperature(alpha)
        observations = self._to_tensor(batch["observations"])
        actions = self._to_tensor(batch["actions"])
        rewards = self._to_tensor(batch["rewards"])
        next_observations = self._to_tensor(batch["next_observations"])
        dones = self._to_tensor(batch["dones"])

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_observations)
            target_q1, target_q2 = self.target_critic(next_observations, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            td_target = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(observations, actions)
        critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        sampled_actions, log_prob, _ = self.actor.sample(observations)
        q1_pi, q2_pi = self.critic(observations, sampled_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.critic, self.target_critic, self.tau)
        return SACUpdateMetrics(
            actor_loss=float(actor_loss.item()),
            critic_loss=float(critic_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha=float(self.alpha.item()),
            q_mean=float(q_pi.mean().item()),
            log_prob=float(log_prob.mean().item()),
            action_saturation=float((sampled_actions.abs() > 0.99).float().mean().item()),
        )

    def state_dict(self):
        # 单独暴露 agent 的保存接口，方便 trainer 做 checkpoint。
        return {
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "tau": self.tau,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "target_entropy": self.target_entropy,
            "actor_mean_clip": self.actor_mean_clip,
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        self.target_entropy = float(state_dict["target_entropy"])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def _to_tensor(self, value):
        """把 replay buffer 取出的 numpy batch 统一搬到训练设备上。"""
        if torch.is_tensor(value):
            return value.to(self.device, dtype=torch.float32)
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    @staticmethod
    def _soft_update(source, target, tau):
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.data.mul_(1.0 - tau).add_(tau * source_parameter.data)
