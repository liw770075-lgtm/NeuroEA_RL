from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Population:
    dec: torch.Tensor
    obj: torch.Tensor
    con: torch.Tensor
    add: Optional[torch.Tensor] = None

    def __len__(self):
        return int(self.dec.shape[0])

    @property
    def device(self):
        return self.dec.device

    @property
    def dtype(self):
        return self.dec.dtype

    @property
    def n_var(self):
        return int(self.dec.shape[1]) if self.dec.ndim == 2 else 0

    @property
    def n_obj(self):
        return int(self.obj.shape[1]) if self.obj.ndim == 2 else 0

    def take(self, indices):
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        else:
            indices = indices.to(device=self.device, dtype=torch.long)

        add = None if self.add is None else self.add.index_select(0, indices)
        return Population(
            dec=self.dec.index_select(0, indices),
            obj=self.obj.index_select(0, indices),
            con=self.con.index_select(0, indices),
            add=add,
        )

    @classmethod
    def empty(cls, n_var, n_obj=1, device="cpu", dtype=torch.float64):
        return cls(
            dec=torch.empty((0, n_var), device=device, dtype=dtype),
            obj=torch.empty((0, n_obj), device=device, dtype=dtype),
            con=torch.empty((0, 1), device=device, dtype=dtype),
            add=None,
        )

    @classmethod
    def cat(cls, populations):
        valid = [pop for pop in populations if len(pop) > 0]
        if not valid:
            template = populations[0]
            return cls.empty(template.n_var, template.n_obj, template.device, template.dtype)
        return cls(
            dec=torch.cat([pop.dec for pop in valid], dim=0),
            obj=torch.cat([pop.obj for pop in valid], dim=0),
            con=torch.cat([pop.con for pop in valid], dim=0),
            add=None,
        )
