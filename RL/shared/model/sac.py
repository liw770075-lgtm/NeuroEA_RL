"""simpleNN 版 SAC 的 actor / critic 网络。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from RL.shared.model.mlp import MLP


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class SquashedGaussianPolicy(nn.Module):
    """SAC 使用的 tanh-squashed Gaussian policy，动作范围固定在 [-1, 1]。"""

    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256), mean_clip=None):
        super().__init__()
        self.action_dim = int(action_dim)
        self.mean_clip = None if mean_clip is None else float(mean_clip)
        self.backbone = MLP(obs_dim, hidden_dims, 2 * action_dim)

    def forward(self, observation):
        output = self.backbone(observation)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        if self.mean_clip is not None:
            mean = torch.clamp(mean, -self.mean_clip, self.mean_clip)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    def sample(self, observation, deterministic=False):
        mean, log_std = self(observation)
        if deterministic:
            pre_tanh = mean
        else:
            std = log_std.exp()
            noise = torch.randn_like(mean)
            pre_tanh = mean + std * noise

        action = torch.tanh(pre_tanh)

        if deterministic:
            log_prob = None
        else:
            normal_log_prob = -0.5 * (
                ((pre_tanh - mean) / (log_std.exp() + 1e-8)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
            )
            normal_log_prob = normal_log_prob.sum(dim=-1, keepdim=True)
            correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            log_prob = normal_log_prob - correction

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


class TwinQNetwork(nn.Module):
    """SAC 的双 Q 网络。"""

    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        input_dim = int(obs_dim) + int(action_dim)
        self.q1 = MLP(input_dim, hidden_dims, 1)
        self.q2 = MLP(input_dim, hidden_dims, 1)

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        return self.q1(x), self.q2(x)
