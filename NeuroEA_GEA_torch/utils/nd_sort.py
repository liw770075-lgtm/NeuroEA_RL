import numpy as np
import torch


def nd_sort(pop_obj, pop_con=None, n_sort=None):
    """A small non-dominated sorter. Multi-objective is not the hot path here."""
    if not torch.is_tensor(pop_obj):
        pop_obj = torch.as_tensor(pop_obj)

    n, m = pop_obj.shape
    if n_sort is None:
        n_sort = n

    device = pop_obj.device
    dtype = pop_obj.dtype

    if pop_con is not None:
        if not torch.is_tensor(pop_con):
            pop_con = torch.as_tensor(pop_con, device=device, dtype=dtype)
        else:
            pop_con = pop_con.to(device=device, dtype=dtype)
        con_sum = torch.clamp(pop_con, min=0).sum(dim=1)
    else:
        con_sum = None

    sort_keys = tuple(pop_obj[:, col].detach().cpu().numpy() for col in reversed(range(m)))
    idx_np = np.lexsort(sort_keys)
    idx = torch.as_tensor(idx_np, device=device, dtype=torch.long)
    sorted_obj = pop_obj.index_select(0, idx)

    front_no = torch.full((n,), float("inf"), device=device, dtype=dtype)
    max_f_no = 0

    for i in range(n):
        cur_idx = idx[i]
        cur_obj = sorted_obj[i]
        cur_con = None if con_sum is None else con_sum[cur_idx]

        for front in range(1, max_f_no + 2):
            dominated = False
            for j in range(i):
                prev_idx = idx[j]
                if int(front_no[prev_idx].item()) != front:
                    continue

                if con_sum is not None:
                    prev_con = con_sum[prev_idx]
                    prev_feasible = prev_con <= 0
                    cur_feasible = cur_con <= 0
                    if prev_feasible and not cur_feasible:
                        dominated = True
                        break
                    if not prev_feasible and cur_feasible:
                        continue
                    if not prev_feasible and not cur_feasible:
                        if prev_con < cur_con:
                            dominated = True
                            break
                        continue

                prev_obj = sorted_obj[j]
                if torch.all(prev_obj <= cur_obj) and torch.any(prev_obj < cur_obj):
                    dominated = True
                    break

            if not dominated:
                front_no[cur_idx] = float(front)
                max_f_no = max(max_f_no, front)
                break

        if int(torch.sum(front_no <= max_f_no).item()) >= n_sort:
            break

    return front_no, max_f_no
