"""Observation 处理工具。

这里同时服务两条训练路径：
- 把 observation flatten 成向量，给 simpleNN-SAC 使用
- 保留结构化 observation，并在 sample 时做 batch collate，给 EvoFormer-SAC 使用
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def flatten_observation(observation: Any) -> np.ndarray:
    """
    把环境 observation 压平成一维向量。

    当前默认策略是：
    - 如果是 dict，就按 key 排序后依次拼接
    - 如果是 tensor / ndarray，就直接 flatten

    这样 SAC 这类 MLP 方法可以直接消费 observation。
    """
    if isinstance(observation, dict):
        parts = [flatten_observation(observation[key]) for key in sorted(observation.keys())]
        if not parts:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    if torch.is_tensor(observation):
        observation = observation.detach().cpu().numpy()

    array = np.asarray(observation, dtype=np.float32)
    return array.reshape(-1)


def infer_observation_dim(observation: Any) -> int:
    """推断 flatten 后的一维 observation 长度。"""
    return int(flatten_observation(observation).shape[0])


def identity_observation(observation: Any) -> Any:
    """
    不对 observation 做额外处理。

    这个入口主要给“结构化 observation”路径使用，例如：
    - Transformer 直接消费 X_seq / Y_seq / mask_*
    - agent 自己内部再做 collate / encode
    """
    return observation


def clone_observation(observation: Any) -> Any:
    """
    深拷贝 observation。

    replay buffer 里存结构化 observation 时，不能直接保存环境返回的引用；
    否则后面环境继续推进时，历史内容可能被意外共享或覆盖。
    """
    if isinstance(observation, dict):
        return {key: clone_observation(value) for key, value in observation.items()}
    if torch.is_tensor(observation):
        return observation.detach().cpu().clone()
    if isinstance(observation, np.ndarray):
        return observation.copy()
    if isinstance(observation, (list, tuple)):
        cloned = [clone_observation(item) for item in observation]
        return type(observation)(cloned)
    return observation


def collate_transformer_observations(batch):
    """
    把一批“按代轨迹 observation”整理成可直接送入三层 Transformer 的 batch。

    单条 observation 允许：
    - T 不同：不同 episode 已经历代数不同
    - N / D / M 不同：不同任务规模不同

    这里统一 pad 到 batch 内最大形状，并给出 mask。
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate an empty observation batch.")

    t_max = max(int(np.asarray(item["X_seq"]).shape[0]) for item in batch)
    n_max = max(int(np.asarray(item["X"]).shape[0]) for item in batch)
    d_max = max(int(np.asarray(item["X"]).shape[1]) for item in batch)
    m_max = max(int(np.asarray(item["Y"]).shape[1]) for item in batch)

    x_seq = torch.zeros((len(batch), t_max, n_max, d_max), dtype=torch.float32)
    y_seq = torch.zeros((len(batch), t_max, n_max, m_max), dtype=torch.float32)
    mask_t = torch.zeros((len(batch), t_max), dtype=torch.bool)
    mask_n_seq = torch.zeros((len(batch), t_max, n_max), dtype=torch.bool)
    mask_d_seq = torch.zeros((len(batch), t_max, d_max), dtype=torch.bool)
    mask_m_seq = torch.zeros((len(batch), t_max, m_max), dtype=torch.bool)

    x_cur = torch.zeros((len(batch), n_max, d_max), dtype=torch.float32)
    y_cur = torch.zeros((len(batch), n_max, m_max), dtype=torch.float32)
    mask_n = torch.zeros((len(batch), n_max), dtype=torch.bool)
    mask_d = torch.zeros((len(batch), d_max), dtype=torch.bool)
    mask_m = torch.zeros((len(batch), m_max), dtype=torch.bool)

    summary_list = []
    control_list = []
    task_list = []

    summary_dim = max(int(np.asarray(item.get("summary", np.empty(0))).reshape(-1).shape[0]) for item in batch)
    control_dim = max(int(np.asarray(item.get("control", np.empty(0))).reshape(-1).shape[0]) for item in batch)
    task_dim = max(int(np.asarray(item.get("task", np.empty(0))).reshape(-1).shape[0]) for item in batch)

    for index, item in enumerate(batch):
        current_x = np.asarray(item["X"], dtype=np.float32)
        current_y = np.asarray(item["Y"], dtype=np.float32)
        current_mask_n = np.asarray(item["mask_N"], dtype=bool)
        current_mask_d = np.asarray(item["mask_D"], dtype=bool)
        current_mask_m = np.asarray(item["mask_M"], dtype=bool)

        t_size = int(np.asarray(item["X_seq"]).shape[0])
        seq_n_size = int(np.asarray(item["X_seq"]).shape[1])
        seq_d_size = int(np.asarray(item["X_seq"]).shape[2])
        seq_m_size = int(np.asarray(item["Y_seq"]).shape[2])
        n_size, d_size = current_x.shape
        _, m_size = current_y.shape

        x_seq[index, :t_size, :seq_n_size, :seq_d_size] = torch.as_tensor(item["X_seq"], dtype=torch.float32)
        y_seq[index, :t_size, :seq_n_size, :seq_m_size] = torch.as_tensor(item["Y_seq"], dtype=torch.float32)
        mask_t[index, :t_size] = torch.as_tensor(item["mask_T"], dtype=torch.bool)
        mask_n_seq[index, :t_size, :seq_n_size] = torch.as_tensor(item["mask_N_seq"], dtype=torch.bool)
        mask_d_seq[index, :t_size, :seq_d_size] = torch.as_tensor(item["mask_D_seq"], dtype=torch.bool)
        mask_m_seq[index, :t_size, :seq_m_size] = torch.as_tensor(item["mask_M_seq"], dtype=torch.bool)

        x_cur[index, :n_size, :d_size] = torch.as_tensor(current_x, dtype=torch.float32)
        y_cur[index, :n_size, :m_size] = torch.as_tensor(current_y, dtype=torch.float32)
        mask_n[index, :n_size] = torch.as_tensor(current_mask_n, dtype=torch.bool)
        mask_d[index, :d_size] = torch.as_tensor(current_mask_d, dtype=torch.bool)
        mask_m[index, :m_size] = torch.as_tensor(current_mask_m, dtype=torch.bool)

        if summary_dim > 0:
            summary = np.asarray(item.get("summary", np.empty(0)), dtype=np.float32).reshape(-1)
            holder = np.zeros((summary_dim,), dtype=np.float32)
            holder[: summary.shape[0]] = summary
            summary_list.append(holder)

        if control_dim > 0:
            control = np.asarray(item.get("control", np.empty(0)), dtype=np.float32).reshape(-1)
            holder = np.zeros((control_dim,), dtype=np.float32)
            holder[: control.shape[0]] = control
            control_list.append(holder)

        if task_dim > 0:
            task = np.asarray(item.get("task", np.empty(0)), dtype=np.float32).reshape(-1)
            holder = np.zeros((task_dim,), dtype=np.float32)
            holder[: task.shape[0]] = task
            task_list.append(holder)

    output = {
        "X_seq": x_seq,
        "Y_seq": y_seq,
        "mask_T": mask_t,
        "mask_N_seq": mask_n_seq,
        "mask_D_seq": mask_d_seq,
        "mask_M_seq": mask_m_seq,
        "X": x_cur,
        "Y": y_cur,
        "mask_N": mask_n,
        "mask_D": mask_d,
        "mask_M": mask_m,
    }

    if summary_dim > 0:
        output["summary"] = torch.as_tensor(np.stack(summary_list, axis=0), dtype=torch.float32)
    if control_dim > 0:
        output["control"] = torch.as_tensor(np.stack(control_list, axis=0), dtype=torch.float32)
    if task_dim > 0:
        output["task"] = torch.as_tensor(np.stack(task_list, axis=0), dtype=torch.float32)

    return output
