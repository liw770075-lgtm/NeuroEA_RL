from NeuroEA_GEA_torch.Algorithms.NEUROEA.BLOCK import Block


class Block_Population(Block):
    def Main(self, problem, precursors, ratio):
        self.output = self.Gather(problem, precursors, ratio, 1, 1)

    def initialization(self, population):
        self.output = population
        self.next_out = 0
