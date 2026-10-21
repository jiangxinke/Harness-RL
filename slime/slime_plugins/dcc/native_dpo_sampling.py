"""Explicit two-domain DPO draw sampling; legacy sampling stays in its trainer."""
from __future__ import annotations

from collections import Counter
import random

LEGACY_PAIR_SAMPLING = "global-deck"
DOMAIN_BALANCED_PAIR_SAMPLING = "domain-balanced-fixed-batch"
DOMAINS = ("cuad", "widesearch")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def balanced_pair_sampling_configuration(*, seed, global_draws):
    require(type(seed) is int, "Integer pair sampling seed required")
    require(type(global_draws) is int and global_draws > 0 and global_draws % 2 == 0,
            "Two-domain fixed batches require a positive even global draw count")
    return {
        "version": "native-two-domain-fixed-pair-draws-v1",
        "mode": DOMAIN_BALANCED_PAIR_SAMPLING,
        "domain_weights": {domain: 0.5 for domain in DOMAINS},
        "global_draws_per_step": global_draws,
        "seed": seed,
        "within_domain": "independent-uniform-shuffled-cycles; carry-tails; never-drop",
        "insufficient_unique_pairs": "cycle-with-explicit-repeated-draws",
        "repeated_draw_gradient": "merge-multiplicity; each-draw-weight-1/global-draws",
        "reference_ownership": "unchanged-dataset-index-mod-world",
    }


class DomainBalancedPairDeck:
    """Half the draws per domain; every domain pair appears once per cycle.

    Continue into a freshly shuffled domain cycle to fill a batch. This can
    repeat a pair within an update, including domains with fewer pairs than
    their draw quota. Repetition is explicit and must use multiplicity-aware
    training, not be represented as additional independent data.
    """
    def __init__(self, pairs, *, seed):
        require(type(seed) is int, "Integer pair sampling seed required")
        self.domains = tuple(pair.chosen.prefix.benchmark for pair in pairs)
        require(set(self.domains) == set(DOMAINS),
                "Balanced DPO requires nonempty CUAD and WideSearch pair domains only")
        self.indices = {domain: [i for i, value in enumerate(self.domains) if value == domain]
                        for domain in DOMAINS}
        self.rngs = {domain: random.Random(seed + offset * 1000003)
                     for offset, domain in enumerate(DOMAINS)}
        self.decks = {domain: [] for domain in DOMAINS}

    def _draw(self, domain, count):
        selected = []
        while len(selected) < count:
            if not self.decks[domain]:
                self.decks[domain] = list(self.indices[domain])
                self.rngs[domain].shuffle(self.decks[domain])
            size = min(count - len(selected), len(self.decks[domain]))
            selected.extend(self.decks[domain][:size])
            del self.decks[domain][:size]
        return selected

    def next(self, requested):
        balanced_pair_sampling_configuration(seed=0, global_draws=requested)
        draws = {domain: self._draw(domain, requested // 2) for domain in DOMAINS}
        return [index for row in zip(*(draws[domain] for domain in DOMAINS)) for index in row]

    def state_dict(self):
        return {
            "version": "native-two-domain-fixed-pair-draw-state-v1",
            "rng_states": {domain: self.rngs[domain].getstate() for domain in DOMAINS},
            "remaining_decks": {domain: list(self.decks[domain]) for domain in DOMAINS},
        }

    def describe(self, selected):
        selected = tuple(selected)
        require(all(type(i) is int and 0 <= i < len(self.domains) for i in selected),
                "Draw index is outside the validated pair dataset")
        counts = Counter(selected)
        return {
            "global_pair_draws": len(selected),
            "global_unique_pairs": len(counts),
            "domain_draw_counts": {domain: sum(self.domains[i] == domain for i in selected)
                                   for domain in DOMAINS},
            "pair_multiplicities": [{"pair_index": i, "draws": count}
                                    for i, count in sorted(counts.items())],
            "global_pairs_semantics": "draws; repeated pairs are not independent samples",
        }
