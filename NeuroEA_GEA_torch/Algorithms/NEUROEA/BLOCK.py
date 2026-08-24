from abc import ABC, abstractmethod

import numpy as np
import torch

from NeuroEA_GEA_torch.Problems.POPULATION import Population


class BlockValidityError(ValueError):
    """Validation error raised for invalid block graphs."""

    def __init__(self, identifier, message, invalid_indices=None):
        super().__init__(message)
        self.identifier = identifier
        self.invalid_indices = [] if invalid_indices is None else list(invalid_indices)


class Block(ABC):
    def __init__(self, rng=None, device="cpu", dtype=torch.float64):
        self.rng = rng
        self.device = torch.device(device)
        self.dtype = dtype

        self.parameter = torch.empty(0, device=self.device, dtype=self.dtype)
        self.lower = torch.empty(0, device=self.device, dtype=self.dtype)
        self.upper = torch.empty(0, device=self.device, dtype=self.dtype)
        self.output = None
        self.next_out = 0
        self.train_time = 0

    def _tensor(self, value, dtype=None):
        target_dtype = self.dtype if dtype is None else dtype
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=target_dtype)
        return torch.as_tensor(value, device=self.device, dtype=target_dtype)

    @abstractmethod
    def Main(self, problem, precursors, ratio):
        pass

    def ParameterAssign(self):
        pass

    def set_parameters(self, value):
        value = self._tensor(value)
        if self.lower.numel() > 0:
            value = torch.maximum(torch.minimum(value, self.upper), self.lower)
        self.parameter = value
        self.ParameterAssign()
        self.train_time += 1

    def Gather(self, problem, predecessors, ratios, in_type, multiple):
        valid_indices = [i for i, r in enumerate(ratios) if float(r) > 0]
        temp_outputs = []
        lengths = []

        for i in valid_indices:
            pred = predecessors[i]
            ratio = float(ratios[i])

            if in_type == 1:
                if torch.is_tensor(pred.output):
                    pred.output = problem.evaluation(pred.output)
                out = pred.output if isinstance(pred.output, Population) else Population.empty(
                    problem.D, problem.M, problem.device, problem.dtype
                )
                size_n = len(out)
            else:
                if isinstance(pred.output, Population):
                    out = pred.output.dec
                elif pred.output is None:
                    out = torch.empty((0, problem.D), device=problem.device, dtype=problem.dtype)
                else:
                    out = pred.output

                if out.ndim == 1 and out.numel() > 0:
                    out = out.reshape(-1, problem.D)
                size_n = int(out.shape[0])

            if size_n == 0:
                continue

            num_to_gather = max(1, int(np.floor(size_n * ratio)))
            indices = np.mod(np.arange(pred.next_out, pred.next_out + num_to_gather), size_n)
            idx_tensor = torch.as_tensor(indices, device=problem.device, dtype=torch.long)

            if in_type == 1:
                temp_outputs.append(out.take(idx_tensor))
            else:
                temp_outputs.append(out.index_select(0, idx_tensor))

            lengths.append(len(indices))
            pred.next_out = int((indices[-1] + 1) % size_n)

        if not temp_outputs:
            self.next_out = 0
            if in_type == 1:
                return Population.empty(problem.D, problem.M, problem.device, problem.dtype)
            return torch.empty((0, problem.D), device=problem.device, dtype=problem.dtype)

        max_len = max(lengths)
        final_indices = []
        offsets = np.cumsum([0] + lengths[:-1])
        for j in range(max_len):
            for i, length in enumerate(lengths):
                if j < length:
                    final_indices.append(int(offsets[i] + j))

        if in_type == 1:
            merged = Population.cat(temp_outputs)
            limit = (len(final_indices) // multiple) * multiple
            if limit == 0:
                self.next_out = 0
                return Population.empty(problem.D, problem.M, problem.device, problem.dtype)
            self.next_out = 0
            return merged.take(final_indices[:limit])

        merged = torch.cat(temp_outputs, dim=0)
        limit = (len(final_indices) // multiple) * multiple
        if limit == 0:
            self.next_out = 0
            return torch.empty((0, problem.D), device=problem.device, dtype=problem.dtype)
        final_idx = torch.as_tensor(final_indices[:limit], device=problem.device, dtype=torch.long)
        self.next_out = 0
        return merged.index_select(0, final_idx)

    @classmethod
    def ParameterSet(cls, blocks, value):
        block_list = cls._normalize_blocks(blocks)
        flat_value = torch.as_tensor(value).reshape(-1)
        expected = sum(int(block.parameter.numel()) for block in block_list)
        if int(flat_value.numel()) != expected:
            raise ValueError(f"Expected {expected} block parameters, got {int(flat_value.numel())}.")

        offset = 0
        for block in block_list:
            n_par = int(block.parameter.numel())
            block.set_parameters(flat_value[offset : offset + n_par])
            offset += n_par

    @classmethod
    def parameters(cls, blocks):
        return cls._concat_attribute(blocks, "parameter")

    @classmethod
    def lowers(cls, blocks):
        return cls._concat_attribute(blocks, "lower")

    @classmethod
    def uppers(cls, blocks):
        return cls._concat_attribute(blocks, "upper")

    @classmethod
    def Validity(cls, blocks, graph):
        block_list = cls._normalize_blocks(blocks)
        graph = np.asarray(graph, dtype=float)
        cls._validate_graph_shape(block_list, graph)

        type_names = [block.__class__.__name__.replace("Block_", "") for block in block_list]
        adjacency = graph > 0

        if not type_names or type_names[0] != "Population":
            raise BlockValidityError("BLOCK:NoPopulation", "the first block is not a population.", [])

        if not any(name in {"Crossover", "Exchange", "Mutation"} for name in type_names):
            raise BlockValidityError(
                "BLOCK:NoOperator",
                "the algorithm does not contain variation operator.",
                [],
            )

        no_input = (np.where(adjacency.sum(axis=0) == 0)[0] + 1).tolist()
        if no_input:
            raise BlockValidityError(
                "BLOCK:NoInput",
                f"the block #{' '.join(map(str, no_input))} have no predecessor.",
                no_input,
            )

        no_output = (np.where(adjacency.sum(axis=1) == 0)[0] + 1).tolist()
        if no_output:
            raise BlockValidityError(
                "BLOCK:NoOutput",
                f"the block #{' '.join(map(str, no_output))} have no successor.",
                no_output,
            )

        isolated = cls._isolated_blocks(adjacency)
        if isolated:
            raise BlockValidityError(
                "BLOCK:Isolation",
                f"the block #{' '.join(map(str, isolated))} are isolated.",
                isolated,
            )

        self_loop = (np.where(np.diag(adjacency))[0] + 1).tolist()
        if self_loop:
            raise BlockValidityError(
                "BLOCK:SelfLoop",
                f"the block #{' '.join(map(str, self_loop))} have self-loop.",
                self_loop,
            )

        invalid_cycle = cls._first_invalid_cycle(adjacency, type_names)
        if invalid_cycle:
            raise BlockValidityError(
                "BLOCK:InfLoop",
                f"the cycle #{' '.join(map(str, invalid_cycle))} have no population.",
                invalid_cycle,
            )

    @staticmethod
    def _normalize_blocks(blocks):
        if isinstance(blocks, Block):
            return [blocks]
        return list(blocks)

    @classmethod
    def _concat_attribute(cls, blocks, attribute):
        block_list = cls._normalize_blocks(blocks)
        if not block_list:
            return torch.empty(0)

        device = block_list[0].device
        dtype = block_list[0].dtype
        tensors = [getattr(block, attribute).reshape(-1).to(device=device, dtype=dtype) for block in block_list]
        if len(tensors) == 1:
            return tensors[0].clone()
        return torch.cat(tensors, dim=0)

    @staticmethod
    def _validate_graph_shape(blocks, graph):
        n_blocks = len(blocks)
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1] or graph.shape[0] != n_blocks:
            raise ValueError(f"Graph shape must be ({n_blocks}, {n_blocks}), got {graph.shape}.")

    @staticmethod
    def _isolated_blocks(adjacency):
        if adjacency.size == 0:
            return []

        undirected = adjacency | adjacency.T
        visited = np.zeros(adjacency.shape[0], dtype=bool)
        stack = [0]
        visited[0] = True

        while stack:
            node = stack.pop()
            neighbors = np.where(undirected[node])[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))

        return (np.where(~visited)[0] + 1).tolist()

    @classmethod
    def _first_invalid_cycle(cls, adjacency, type_names):
        cycles = cls._simple_cycles(adjacency)
        for cycle in cycles:
            if "Population" not in [type_names[index] for index in cycle]:
                return [index + 1 for index in cycle]
        return []

    @staticmethod
    def _simple_cycles(adjacency):
        n_nodes = adjacency.shape[0]
        cycles = set()

        for start in range(n_nodes):
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                for nxt in np.where(adjacency[node])[0]:
                    nxt = int(nxt)
                    if nxt == start and len(path) > 1:
                        cycles.add(Block._canonical_cycle(path))
                    elif nxt not in visited:
                        stack.append((nxt, path + [nxt], visited | {nxt}))

        return [list(cycle) for cycle in sorted(cycles)]

    @staticmethod
    def _canonical_cycle(cycle):
        cycle = list(cycle)
        min_pos = min(range(len(cycle)), key=cycle.__getitem__)
        rotated = cycle[min_pos:] + cycle[:min_pos]
        return tuple(rotated)
