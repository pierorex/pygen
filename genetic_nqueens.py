from geneticoptimizer import GeneticOptimizer
from random import random, randint, shuffle
from timeit import timeit
from time import time
from parents_mixins import ParentsRouletteSelectionMixin, \
    ParentsRandomSelectionMixin
from survivors_mixins import SurvivorsTruncatedSelectionMixin, \
    SurvivorsRouletteSelectionMixin


class NqueensOptimizer(GeneticOptimizer):
    def __init__(self, n=8):
        self.n = n

    def individual(self):
        """
        representation: [1,5,4,3,9,7,8,6,0,2]
        meaning queen 0 is at (0,1), queen 1 is at (1,5), ...
        """
        l = range(self.n)
        shuffle(l)
        return l

    def fitness(self, solution):
        # count how many times a queen collides with another. We only count
        # diagonal collisions because our representation takes care of
        # horizontal and vertical collisions
        fit = 0

        for x1, y1 in enumerate(solution):
            for x2, y2 in enumerate(solution):
                # for each pair of queens
                # if vertical_distance == horizontal_distance
                # then they're colliding diagonally
                fit += 1 if abs(x1-x2) == abs(y1-y2) else 0

        # subtract self.n from fit because we are counting the collision
        # between every queen and theirselves
        return fit - self.n

    def mutate(self, mutate_prob, parents):
        for solution in parents:
            if mutate_prob > random():
                pos1, pos2 = 0, 0

                while pos1 == pos2:
                    pos1 = randint(0, len(solution) - 1)
                    pos2 = randint(0, len(solution) - 1)

                solution['individual'][pos1], solution['individual'][pos2] = \
                    solution['individual'][pos2], solution['individual'][pos1]

    def mix(self, parent1, parent2):
        # We take the queens in the first half from parent1, then grab all
        # queens from parent2 that we haven't taken already from parent1
        half = len(parent1) / 2
        child = parent1[:half]
        child.extend([i for i in parent2 if i not in child])
        return child


if __name__ == '__main__':
    no = NqueensOptimizer(20)
    no.__class__ = type('Classifier', (ParentsRandomSelectionMixin,
                                       SurvivorsTruncatedSelectionMixin,
                                       NqueensOptimizer),
                        {})
    no.find_optimal(iterations=300, pop_count=50, target=0.0, mutate_prob=0.2)
