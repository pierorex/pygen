from roullette_selection import RoulletteSelectionMixin


class SurvivorsTruncatedSelectionMixin(object):
    def survivors_select(self, ranked, retain_length):
        return ranked[:retain_length]


class SurvivorsRoulletteSelectionMixin(RoulletteSelectionMixin):
    def survivors_select(self, ranked, retain_length):
        self.calculate_probs_roullette(ranked)
        # roullette-pick 'retain_length' survivors
        survivors = []

        while len(survivors) < retain_length:
            chosen = ranked[self.roullette_pick(ranked)]
            survivors.append(chosen)
            ranked.remove(chosen)
        return survivors
