from random import random, randint


class RouletteSelectionMixin(object):
    def roulette_pick(self, arr):
        """Returns the index of the parent chosen by the roulette"""
        rand = random()
        for i in xrange(len(arr)):
            if rand < arr[i]['prob']:
                return i
        return randint(0, len(arr)-1)

    def calculate_probs_roulette(self, arr):
        total_fitness = 0.0
        best_fitness = arr[0]['fitness']

        for x in arr:
            total_fitness += x['fitness']
            if x['fitness'] > best_fitness:
                best_fitness = x['fitness']

        best_fitness += 1  # to avoid making it negative in the next loop
        total_prob = 0.0

        # calculate accumulated probability for each individual depending on
        # its fitness
        for x in arr:
            if self.reverse:  # high fitness is better
                x['prob'] = total_prob + (x['fitness'] / total_fitness)
            else:             # low fitness is better
                x['prob'] = \
                    total_prob + ((best_fitness-x['fitness'])/total_fitness)\
                    if x['fitness'] != 0 else 0.0
            total_prob = x['prob']

        # the last one should have prob == 1.0, computers numeric imprecision
        # forces us to do this manually
        arr[len(arr)-1]['prob'] = 1.0
