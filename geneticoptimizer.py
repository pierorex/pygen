from random import randint, random
from time import time


class GeneticOptimizer(object):
    """
    Framework to solve optimization problems using a genetic algorithm.
    To use this class you should subclass it and override:

    __init__(self)
        add parameters to your instances that you may need for calculations
    individual(self)
        define a representation for a single solution
    fitness(self, x)
        penalize an individual x, according to some rules. High fitness means
        it is a worse solution
    mutate(self, mutate_prob, parents)
        change an individual in some way to allow it to explore new conditions
    mix(self, parent1, parent2)
        define how two individuals produce a new one

    After that, normal usage would go like this:

    MyOptimizer(n=12, p=2).runGA(iterations=500,
                                 pop_count=100,
                                 target=0.0,
                                 mutate_prob=0.1,
                                 retain=0.2,
                                 diversity_prob=0.05)

    This will output a tuple (rank, sol), where rank is the fitness of the best
    found solution and sol is that solution (an array of chromosomes)
    """
    def __init__(self, goal=100, length=5, i_min=0, i_max=100):
        self.goal = goal
        self.length = length
        self.min = i_min
        self.max = i_max

    def individual(self):
        return [randint(self.min, self.max) for i in xrange(self.length)]

    def population(self, pop_count):
        return [self.individual() for x in xrange(pop_count)]

    def fitness(self, x):
        return abs(self.goal - sum(x))

    def rank(self, x):
        return sum(self.fitness(x) for x in self.pop) / (len(self.pop) * 1.0)

    def mutate(self, mutate_prob, parents):
        for individual in parents:
            if mutate_prob > random():
                pos_to_mutate = randint(0, len(individual)-1)
                individual[pos_to_mutate] = \
                    randint(min(individual), max(individual))

    def promote_diversity(self, ranked, retain_length, parents,
                          diversity_prob):
        for individual in ranked[retain_length:]:
            if diversity_prob > random():
                parents.append(individual)

    def mix(self, parent1, parent2):
        half = len(parent1) / 2
        child = parent1[:half] + parent2[half:]
        return child

    def crossover(self, parents, pop_length):
        parents_length = len(parents)
        desired_length = pop_length - parents_length
        children = []

        while len(children) < desired_length:
            male = randint(0, parents_length-1)
            female = randint(0, parents_length-1)
            if male != female:
                male = parents[male]
                female = parents[female]
                children.append(self.mix(male, female))
                children.append(self.mix(female, male))
        parents.extend(children)

    def evolve(self, ranked, retain, diversity_prob, mutate_prob):
        retain_length = int(len(ranked) * retain)
        parents = ranked[:retain_length]
        self.promote_diversity(ranked, retain_length, parents, diversity_prob)
        self.mutate(mutate_prob, parents)
        self.crossover(parents, len(self.pop))
        return parents

    def runGA(self, iterations, pop_count, target, debug=False, retain=0.2,
              diversity_prob=0.05, mutate_prob=0.01):
        self.pop = self.population(pop_count)
        # if debug:
        #     self.fitness_history = [(self.rank(self.pop), self.pop)]

        for i in xrange(iterations):
            ranked = sorted([(self.fitness(x), x) for x in self.pop])
            ranks = [x[0] for x in ranked]
            solutions = [x[1] for x in ranked]

            # did we arrive at the target solution?
            if target in ranks:
                return ranked[0]

            self.pop = self.evolve(solutions, retain, diversity_prob,
                                   mutate_prob)

            # if debug:
            #     self.fitness_history.append((self.rank(self.pop), self.pop))

        # we missed the target, let's analyze how we scored
        ranked = sorted([(self.fitness(x), x) for x in self.pop])
        return self.population_report() if debug else ranked[0]

    def find_optimal(self, **args):
        repetitions = 0

        while True:
            start_time = time()
            repetitions += 1
            s = self.runGA(**args)
            print s
            print "Elapsed time: %f" % (time() - start_time)
            if s[0] == 0.0:
                print ("Took %d repetitions to get to the optimal " +
                       "solution") % repetitions
                return

    def population_report(self):
        best = (100000000, [])

        for rank, solution in self.fitness_history:
            best = (rank, solution) if rank < best[0] else best
            print rank

        return best


if __name__ == '__main__':
    a = GeneticOptimizer()
    a.find_optimal(iterations=300, pop_count=100, target=0.0, mutate_prob=0.1)
