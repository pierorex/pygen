from geneticoptimizer import GeneticOptimizer
from random import random, randint, shuffle
from parents_mixins import ParentsRouletteSelectionMixin, \
    ParentsRandomSelectionMixin
from survivors_mixins import SurvivorsTruncatedSelectionMixin, \
    SurvivorsRouletteSelectionMixin


class DialerOptimizer(GeneticOptimizer):
    def __init__(self, service_state):
        # service_state should contain information valid for a particular moment
        #  in time, including agents statuses (probably get the dictionary from
        # redis with all this information), calls logs (1000 calls? maybe more),
        # and anything else that could prove beneficial for simulations
        self.service_state = service_state

    def individual(self):
        """
        representation: 0.50
        meaning: the dialer intensity will be at 50%, which means it will
        generate half the max amount of calls it can make
        """
        return random()

    def fitness(self, solution):
        number_lost_calls = 0
        number_attended_calls = 0
        agents = []
        number_agents = len(agents)
        idle_times = {}

        """TODO: run simulation here to update previous variables
        basically, pick the given state and:

        for every agent:
            if it should be called:
                'call' him by taking some new calls from the calls record,
                the exact number of calls to place depends on the intensity
                of the dialer (A.K.A. the solution given as parameter)

                for each call 'placed' (given) to the agent:
                    analyze whether it was attended, lost or connected
                    was attended: attended++
                    was lost: lost++
                    was connected: connected++
                update agent status according to the calls received, if one of
                them got attended, how many seconds did it last? That will be
                the agent's occupation time

        end of simulation"""

        avg_idle_time = sum(idle_times) / number_agents

        lost_calls_ratio = \
            number_lost_calls / (number_lost_calls + number_attended_calls)
        return -(avg_idle_time + lost_calls_ratio)

    def mutate(self, mutate_prob, parents):
        """
        Add or subtract 0.01 from every solution that is going to be mutated
        the sign of the operation is decided at random with .5 probability
        at each option (.5 for + and .5 for -)
        """
        for i in range(len(parents)):
            if mutate_prob > random.random():
                if random() > 0.5:
                    parents[i] += 0.01 if parents[i] <= 0.99 else 0.0
                else:
                    parents[i] -= 0.01 if parents[i] >= 0.01 else 0.0

    def mix(self, parent1, parent2):
        """
        Uses the 'weighted means' method to generate a child:
            - Compute a random omega between 0 and 1.
            - Use omega to generate the child of two parents by a simple
              formula: child = omega*parent1 + (1-omega)*parent2
        """
        omega = random()
        return omega*parent1 + (1-omega)*parent2


if __name__ == '__main__':
    do = DialerOptimizer()
    do.__class__ = type('Classifier', (ParentsRandomSelectionMixin,
                                       SurvivorsTruncatedSelectionMixin,
                                       DialerOptimizer),
                        {})
    do.find_optimal(iterations=10,
                    pop_count=10,
                    target=100.0,
                    mutate_prob=0.2,
                    retain_percent=0.25,
                    diversity_prob=0.1,
                    reverse=True)
