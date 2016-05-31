from roulette_selection import RouletteSelectionMixin


class SurvivorsTruncatedSelectionMixin(object):
    def survivors_select(self, ranked, retain_length):
        return ranked[:retain_length]


class SurvivorsRouletteSelectionMixin(RouletteSelectionMixin):
    def survivors_select(self, ranked, retain_length):
        self.calculate_probs_roulette(ranked)
        # roulette-pick 'retain_length' survivors
        survivors = []

        while len(survivors) < retain_length:
            chosen = ranked[self.roulette_pick(ranked)]
            survivors.append(chosen)
            ranked.remove(chosen)
        return survivors
