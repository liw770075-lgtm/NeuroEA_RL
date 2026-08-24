import torch


def get_fitness_single(population):
    """Single-objective fitness with constraint penalty."""
    objs = population.obj.reshape(-1)
    cons = population.con.reshape(-1)
    penalty = torch.max(objs) if objs.numel() > 0 else torch.tensor(0.0, device=objs.device, dtype=objs.dtype)
    return objs + torch.clamp(cons, min=0) * (penalty + 1e6)
