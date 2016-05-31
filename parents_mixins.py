from random import randint
from roulette_selection import RouletteSelectionMixin


class ParentsRouletteSelectionMixin(RouletteSelectionMixin):
    def parents_select(self, parents):
        self.calculate_probs_roulette(parents)
        # roulette-pick two parents and return them both
        return self.roulette_pick(parents), self.roulette_pick(parents)


class ParentsRandomSelectionMixin(object):
    def parents_select(self, parents):
        return randint(0, len(parents)-1), randint(0, len(parents)-1)
