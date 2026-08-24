"""最基础的 MLP 组件。

simpleNN-SAC 和 Transformer-SAC 的 head 都会复用这里的 MLP。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """可配置的多层感知机骨干网络。"""

    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU, output_activation=None):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            is_last = idx == len(dims) - 2
            if not is_last:
                layers.append(activation())
            elif output_activation is not None:
                layers.append(output_activation())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播。"""
        return self.network(x)
