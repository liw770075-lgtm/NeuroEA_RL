import numpy as np
import torch

from NeuroEA_GEA_torch.Problems.POPULATION import Population


def tournament_selection(k, n, *fitness_values, rng=None, device="cpu"):
    """K-tournament selection with lexicographic tie-breaking."""
    if len(fitness_values) == 0:
        raise ValueError("At least one fitness array is required.")

    fitness_arrays = []
    for value in fitness_values:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        fitness_arrays.append(np.asarray(value).reshape(-1, 1))

    fit = np.concatenate(fitness_arrays, axis=1)
    unique_fit, inverse = np.unique(fit, axis=0, return_inverse=True)
    sort_order = np.lexsort(unique_fit.T[::-1])
    rank = np.empty_like(sort_order)
    rank[sort_order] = np.arange(len(sort_order))

    n_candidates = fit.shape[0]
    if rng is not None and hasattr(rng, "randint_numpy"):
        parents = rng.randint_numpy(0, n_candidates, size=(k, n))
    else:
        parents = np.random.randint(0, n_candidates, size=(k, n))

    candidate_rank = rank[inverse[parents]]
    best = np.argmin(candidate_rank, axis=0)
    indices = parents[best, np.arange(n)]
    return torch.as_tensor(indices, device=device, dtype=torch.long)


def operator_ga(problem, parents, parameters=None):
    """Crossover and mutation operators of the PlatEMO-style GA."""
    if parameters is None:
        pro_c, dis_c, pro_m, dis_m = 1.0, 20.0, 1.0, 20.0
    else:
        pro_c, dis_c, pro_m, dis_m = parameters

    if isinstance(parents, Population):
        evaluated = True
        parent_dec = parents.dec
    else:
        evaluated = False
        if torch.is_tensor(parents):
            parent_dec = parents.to(device=problem.device, dtype=problem.dtype)
        else:
            parent_dec = torch.as_tensor(parents, device=problem.device, dtype=problem.dtype)

    parent1 = parent_dec[: parent_dec.shape[0] // 2]
    parent2 = parent_dec[parent_dec.shape[0] // 2 : (parent_dec.shape[0] // 2) * 2]
    offspring = torch.zeros(
        (2 * parent1.shape[0], parent1.shape[1]),
        device=problem.device,
        dtype=problem.dtype,
    )

    encoding = _normalize_encoding(problem.encoding, problem.D)
    type_indices = [np.where(encoding == i)[0] for i in range(1, 6)]

    real_or_integer = np.concatenate([type_indices[0], type_indices[1]]) if type_indices[0].size or type_indices[1].size else np.array([], dtype=int)
    if real_or_integer.size > 0:
        lower = problem.lower.index_select(0, torch.as_tensor(real_or_integer, device=problem.device, dtype=torch.long))
        upper = problem.upper.index_select(0, torch.as_tensor(real_or_integer, device=problem.device, dtype=torch.long))
        real_offspring = ga_real(
            parent1.index_select(1, torch.as_tensor(real_or_integer, device=problem.device, dtype=torch.long)),
            parent2.index_select(1, torch.as_tensor(real_or_integer, device=problem.device, dtype=torch.long)),
            lower,
            upper,
            float(pro_c),
            float(dis_c),
            float(pro_m) * len(real_or_integer) / parent1.shape[1],
            float(dis_m),
            problem.rng,
        )
        if type_indices[1].size > 0:
            integer_mask = torch.as_tensor(np.isin(real_or_integer, type_indices[1]), device=problem.device, dtype=torch.bool)
            real_offspring[:, integer_mask] = torch.round(real_offspring[:, integer_mask])
        offspring[:, torch.as_tensor(real_or_integer, device=problem.device, dtype=torch.long)] = real_offspring

    if type_indices[2].size > 0:
        idx = torch.as_tensor(type_indices[2], device=problem.device, dtype=torch.long)
        offspring[:, idx] = ga_label(
            parent1.index_select(1, idx),
            parent2.index_select(1, idx),
            problem.lower.index_select(0, idx),
            problem.upper.index_select(0, idx),
            float(pro_c),
            float(pro_m) * len(type_indices[2]) / parent1.shape[1],
            problem.rng,
        )

    if type_indices[3].size > 0:
        idx = torch.as_tensor(type_indices[3], device=problem.device, dtype=torch.long)
        offspring[:, idx] = ga_binary(
            parent1.index_select(1, idx),
            parent2.index_select(1, idx),
            float(pro_c),
            float(pro_m) * len(type_indices[3]) / parent1.shape[1],
            problem.rng,
        )

    if type_indices[4].size > 0:
        idx = torch.as_tensor(type_indices[4], device=problem.device, dtype=torch.long)
        offspring[:, idx] = ga_permutation(
            parent1.index_select(1, idx),
            parent2.index_select(1, idx),
            float(pro_c),
            problem.rng,
        )

    if evaluated:
        return problem.evaluation(offspring)
    return offspring


def ga_real(parent1, parent2, lower, upper, pro_c, dis_c, pro_m, dis_m, rng):
    n, d = parent1.shape
    device = parent1.device
    dtype = parent1.dtype

    beta = torch.zeros((n, d), device=device, dtype=dtype)
    mu = rng.rand(n, d).to(device=device, dtype=dtype)
    left = mu <= 0.5
    right = ~left
    beta[left] = torch.pow(2.0 * mu[left], 1.0 / (dis_c + 1.0))
    beta[right] = torch.pow(2.0 - 2.0 * mu[right], -1.0 / (dis_c + 1.0))

    sign = torch.where(
        rng.randint(0, 2, size=(n, d)).to(device=device) == 1,
        torch.full((n, d), -1.0, device=device, dtype=dtype),
        torch.full((n, d), 1.0, device=device, dtype=dtype),
    )
    beta = beta * sign
    beta[rng.rand(n, d).to(device=device) < 0.5] = 1.0

    crossover_off = rng.rand(n, 1).to(device=device, dtype=dtype) > pro_c
    beta = beta.masked_fill(crossover_off.expand(-1, d), 1.0)

    avg = (parent1 + parent2) / 2.0
    diff = (parent1 - parent2) / 2.0
    offspring = torch.cat([avg + beta * diff, avg - beta * diff], dim=0)

    lower = lower.unsqueeze(0).expand(2 * n, -1)
    upper = upper.unsqueeze(0).expand(2 * n, -1)
    offspring = torch.maximum(torch.minimum(offspring, upper), lower)

    site = rng.rand(2 * n, d).to(device=device) < (pro_m / max(d, 1))
    mu = rng.rand(2 * n, d).to(device=device, dtype=dtype)
    valid = (upper - lower) > 0

    temp = site & (mu <= 0.5) & valid
    if torch.any(temp):
        delta = 1.0 - (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp])
        offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (
            torch.pow(2.0 * mu[temp] + (1.0 - 2.0 * mu[temp]) * torch.pow(delta, dis_m + 1.0), 1.0 / (dis_m + 1.0))
            - 1.0
        )

    temp = site & (mu > 0.5) & valid
    if torch.any(temp):
        delta = 1.0 - (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp])
        offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (
            1.0
            - torch.pow(
                2.0 * (1.0 - mu[temp]) + 2.0 * (mu[temp] - 0.5) * torch.pow(delta, dis_m + 1.0),
                1.0 / (dis_m + 1.0),
            )
        )

    return torch.maximum(torch.minimum(offspring, upper), lower)


def ga_label(parent1, parent2, lower, upper, pro_c, pro_m, rng):
    n, d = parent1.shape
    device = parent1.device
    dtype = parent1.dtype

    mask = rng.rand(n, d).to(device=device) < 0.5
    row_mask = rng.rand(n, 1).to(device=device) > pro_c
    mask[row_mask.expand(-1, d)] = False

    offspring1 = parent1.clone()
    offspring2 = parent2.clone()
    offspring1[mask] = parent2[mask]
    offspring2[mask] = parent1[mask]
    offspring = torch.cat([offspring1, offspring2], dim=0)

    site = rng.rand(2 * n, d).to(device=device) < (pro_m / max(d, 1))
    rand_values = rng.uniform(
        lower.unsqueeze(0).expand(2 * n, -1).detach().cpu().numpy(),
        upper.unsqueeze(0).expand(2 * n, -1).detach().cpu().numpy(),
        size=(2 * n, d),
    ).to(device=device, dtype=dtype)
    rand_values = torch.round(rand_values)
    offspring[site] = rand_values[site]
    return offspring


def ga_binary(parent1, parent2, pro_c, pro_m, rng):
    n, d = parent1.shape
    device = parent1.device

    mask = rng.rand(n, d).to(device=device) < 0.5
    row_mask = rng.rand(n, 1).to(device=device) > pro_c
    mask[row_mask.expand(-1, d)] = False

    offspring1 = parent1.clone()
    offspring2 = parent2.clone()
    offspring1[mask] = parent2[mask]
    offspring2[mask] = parent1[mask]
    offspring = torch.cat([offspring1, offspring2], dim=0)

    site = rng.rand(2 * n, d).to(device=device) < (pro_m / max(d, 1))
    offspring[site] = 1 - offspring[site]
    return offspring


def ga_permutation(parent1, parent2, pro_c, rng):
    n, d = parent1.shape
    device = parent1.device
    dtype = parent1.dtype

    offspring = torch.cat([parent1, parent2], dim=0).detach().cpu().numpy()
    parent1_np = parent1.detach().cpu().numpy()
    parent2_np = parent2.detach().cpu().numpy()

    cut_points = rng.randint_numpy(0, d, size=(2 * n,))
    for i in range(n):
        if float(rng.rand_numpy((1,))[0]) < pro_c:
            k = int(cut_points[i])
            offspring[i, k + 1 :] = _stable_setdiff(parent2_np[i], parent1_np[i, : k + 1])

            k2 = int(cut_points[i + n])
            offspring[i + n, k2 + 1 :] = _stable_setdiff(parent1_np[i], parent2_np[i, : k2 + 1])

    k_pos = rng.randint_numpy(0, d, size=(2 * n,))
    s_pos = rng.randint_numpy(0, d, size=(2 * n,))
    for i in range(2 * n):
        k = int(k_pos[i])
        s = int(s_pos[i])
        if s < k:
            offspring[i] = offspring[i, np.r_[0:s, k, s:k, k + 1 : d]]
        elif s > k:
            offspring[i] = offspring[i, np.r_[0:k, k + 1 : s, k, s:d]]

    return torch.as_tensor(offspring, device=device, dtype=dtype)


def _normalize_encoding(encoding, dimension):
    if torch.is_tensor(encoding):
        encoding = encoding.detach().cpu().numpy()

    encoding = np.asarray(encoding, dtype=int).reshape(-1)
    if encoding.size == 1:
        encoding = np.full(dimension, int(encoding[0]), dtype=int)
    if encoding.size != dimension:
        raise ValueError("Problem encoding must be a scalar or have length equal to problem.D.")
    return encoding


def _stable_setdiff(values, excluded):
    excluded_set = set(excluded.tolist())
    return np.asarray([value for value in values.tolist() if value not in excluded_set])
