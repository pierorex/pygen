from random import randint
from roullette_selection import RoulletteSelectionMixin


class ParentsRouletteSelectionMixin(RoulletteSelectionMixin):
    def parents_select(self, parents):
        self.calculate_probs_roullette(parents)
        # roulette-pick two parents and return them both
        return self.roullette_pick(parents), self.roullette_pick(parents)


class ParentsRandomSelectionMixin(object):
    def parents_select(self, parents):
        return randint(0, len(parents)-1), randint(0, len(parents)-1)
