import numpy as np
import torch


def tosz(x):
    mask = x != 0
    xh = torch.zeros_like(x)
    xh = torch.where(mask, torch.log(torch.abs(x)), xh)
    c1 = torch.where(x > 0, torch.full_like(x, 10.0), torch.full_like(x, 5.5))
    c2 = torch.where(x > 0, torch.full_like(x, 7.9), torch.full_like(x, 3.1))
    return torch.sign(x) * torch.exp(xh + 0.049 * (torch.sin(c1 * xh) + torch.sin(c2 * xh)))


def tasy(beta, x):
    if x.shape[1] == 1:
        return x
    dim_seq = torch.arange(x.shape[1], device=x.device, dtype=x.dtype) / (x.shape[1] - 1)
    exponent = 1 + beta * dim_seq.reshape(1, -1) * torch.sqrt(torch.abs(x))
    return torch.sign(x) * (torch.abs(x) ** exponent)


def fpen(x):
    return torch.sum(torch.clamp(torch.abs(x) - 5, min=0) ** 2, dim=1)


def make_orthogonal_matrix(d, seed, device, dtype):
    state = np.random.RandomState(seed)
    base = torch.as_tensor(state.rand(d, d), dtype=dtype)
    q, _ = torch.linalg.qr(base)
    return q.to(device=device, dtype=dtype)
