import numpy as np

from NeuroEA_GEA_torch.Algorithms.ALGORITHM import Algorithm
from NeuroEA_GEA_torch.Algorithms.NEUROEA.BLOCK import Block
from NeuroEA_GEA_torch.Algorithms.NEUROEA.Block_Population import Block_Population


class NeuroEA(Algorithm):
    """Original NeuroEA that runs the whole optimization in one solve call."""

    def __init__(self, blocks=None, graph=None, **kwargs):
        super().__init__(parameter=[blocks, graph], **kwargs)

    def main(self, problem):
        blocks, graph = self.parameter_set([], [])
        if not blocks or graph.size == 0:
            raise ValueError("NeuroEA requires configured blocks and graph.")

        Block.Validity(blocks, graph)

        num_blocks = len(blocks)
        is_pop = np.array([isinstance(block, Block_Population) for block in blocks])

        initial_pop = problem.initialization()
        for i in np.where(is_pop)[0]:
            blocks[i].initialization(initial_pop)

        while self.not_terminated(blocks[0].output):
            activated = np.zeros(num_blocks, dtype=bool)
            while not np.all(activated[is_pop]):
                for i in range(num_blocks):
                    if activated[i]:
                        continue

                    predecessors_mask = graph[:, i] > 0
                    if np.all(activated[predecessors_mask] | is_pop[predecessors_mask]):
                        pre_blocks = [blocks[j] for j in range(num_blocks) if predecessors_mask[j]]
                        ratios = graph[predecessors_mask, i]
                        blocks[i].Main(problem, pre_blocks, ratios)
                        activated[i] = True
