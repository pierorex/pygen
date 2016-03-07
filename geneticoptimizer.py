from random import randint, random
from time import time
from operator import itemgetter


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
        return [{'individual': self.individual(), 'fitness': None} 
                for x in xrange(pop_count)]

    def fitness(self, x):
        #print x
        return abs(self.goal - sum(x))

    def rank(self, x):
        return sum(self.fitness(x['individual']) 
                   for x in self.pop) / (len(self.pop) * 1.0)

    def mutate(self, mutate_prob, parents):
        for individual in parents:
            if mutate_prob > random():
                pos_to_mutate = randint(0, len(individual)-1)
                individual['individual'][pos_to_mutate] = \
                    randint(self.min, self.max)

    def promote_diversity(self, ranked, retain_length, parents,
                          diversity_prob):
        for individual in ranked[retain_length:]:
            if diversity_prob > random():
                parents.append(individual)

    def mix(self, parent1, parent2):
        half = len(parent1) / 2
        child = parent1[:half] + parent2[half:]
        return child

    def select_random_parents(self, parents):
        return (randint(0, len(parents)-1), randint(0, len(parents)-1))

    def select_roullette_parents(self, parents, ranked):
        raise NotImplementedError

    def select(self, **kwargs):
        return self.select_random_parents(**kwargs)

    def crossover(self, parents, pop_length): 
        desired_length = pop_length - len(parents)
        children = []

        while len(children) < desired_length:
            male, female = self.select(parents=parents)
            if male != female:
                male = parents[male]
                female = parents[female]
                children.append({'fitness':0,
                                 'individual': self.mix(male['individual'],
                                                        female['individual'])})
                children.append({'fitness':0,
                                 'individual': self.mix(female['individual'],
                                                        male['individual'])})
        parents.extend(children)

    def evolve(self, ranked, retain, diversity_prob, mutate_prob):
        retain_length = int(len(ranked) * retain)
        parents = ranked[:retain_length]
        self.promote_diversity(ranked, retain_length, parents, diversity_prob)
        self.mutate(mutate_prob, parents)
        self.crossover(parents, len(self.pop))
        return parents

    def runGA(self, iterations, pop_count, target, best=None, debug=False, retain=0.2,
              diversity_prob=0.05, mutate_prob=0.01, reverse=False):
        if not best:
            best = {'fitness': float('-inf' if reverse else 'inf'), 
                    'individual': []}

        self.pop = self.population(pop_count)
        compare = (lambda x, y: cmp(x,y) if reverse else -cmp(x,y))
        i = 0
        # if debug:
        #     self.fitness_history = [(self.rank(self.pop), self.pop)]

        while i < iterations:
            i += 1
            ranked = sorted([{'fitness': self.fitness(x['individual']), 
                              'individual': x['individual']} 
                             for x in self.pop], reverse=reverse, 
                            key=itemgetter('fitness'))

            # did we arrive at the target solution?
            if target == ranked[0]['fitness']:
                return ranked[0]

            if compare(ranked[0]['fitness'], best['fitness']) > 0:
                print ranked[0]['fitness'], best['fitness']
                best = dict(ranked[0])
                #iterations += 3

            self.pop = self.evolve(ranked, retain, diversity_prob,
                                   mutate_prob)
            #print self.pop

        # we missed the target, let's analyze how we scored
        ranked = sorted([{'fitness': self.fitness(x['individual']), 'individual': x['individual']} 
                         for x in self.pop], reverse=reverse)
        print ranked[0]
        return self.population_report() if debug else ranked[0]

    def find_optimal(self, reverse=False, **kwargs):
        def process_lap(solution, best, start_time):
            print "Best solution: " + str(best['individual'])
            print "len(Best solution): %d " % len(best['individual'])
            print "New solution value:  " + str(solution['fitness'])
            print "Best solution value: " + str(best['fitness'])
            print "Elapsed time: %f" % (time() - start_time)
            print

        repetitions = 0
        best = {'fitness': float('-inf' if reverse else 'inf'), 
                'individual': []}
        compare = (lambda x, y: cmp(x,y) if kwargs.get('reverse') else -cmp(x,y))

        while True:
            start_time = time()
            repetitions += 1
            solution = self.runGA(best=best, **kwargs)

            if compare(solution['fitness'], best['fitness']) == 1:
                print "solutions %f %f" % (solution['fitness'], best['fitness'])
                best = solution
            if compare(solution['fitness'], kwargs['target']) != -1:
                process_lap(solution, best, start_time)
                print ("Took %d repetitions to get to the optimal " +
                       "solution") % repetitions
                return

            process_lap(solution, best, start_time)

    def population_report(self):
        best = (100000000, [])

        for rank, solution in self.fitness_history:
            best = (rank, solution) if rank < best[0] else best
            print rank

        return best

    def precision(self, classifier, dataset, class_length):
        count = 0
        print len(dataset), len(dataset[0])
        for example in dataset:
            #print example
            prediction = self.predict(classifier, example, class_length)
            if example[len(example) - class_length:] == prediction:
                count += 1
        print count
        return float(count) / len(dataset)

    def predict(self, classifier, example, class_length):
        for rule in classifier:
            if self.matches(example, rule):
                return rule[len(rule) - class_length:]


if __name__ == '__main__':
    a = GeneticOptimizer(goal=1000, length=5, i_min=10, i_max=1000)
    a.find_optimal(iterations=10, pop_count=100, target=0.0, mutate_prob=0.1)
