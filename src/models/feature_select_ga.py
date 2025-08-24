from __future__ import annotations

import random
from typing import Callable, List, Tuple


def genetic_feature_selection(
    n_features: int,
    fitness_fn: Callable[[List[int]], float],
    population_size: int = 50,
    generations: int = 10,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.1,
    elitism: int = 2,
    seed: int = 42,
) -> Tuple[List[int], float]:
    """Lightweight GA over binary masks; offline, no numpy.

    Returns (best_mask, best_score)
    """
    random.seed(seed)

    def random_mask() -> List[int]:
        return [1 if random.random() < 0.5 else 0 for _ in range(n_features)]

    def crossover(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
        if random.random() > crossover_rate:
            return a[:], b[:]
        child1, child2 = [], []
        for i in range(n_features):
            if random.random() < 0.5:
                child1.append(a[i])
                child2.append(b[i])
            else:
                child1.append(b[i])
                child2.append(a[i])
        return child1, child2

    def mutate(m: List[int]) -> List[int]:
        return [1 - bit if random.random() < mutation_rate else bit for bit in m]

    pop = [random_mask() for _ in range(population_size)]
    scores = [fitness_fn(m) for m in pop]

    for _ in range(generations):
        ranked = sorted(zip(pop, scores), key=lambda t: t[1], reverse=True)
        new_pop = [m for m, _ in ranked[:elitism]]
        while len(new_pop) < population_size:
            a = tournament_select(ranked)
            b = tournament_select(ranked)
            c1, c2 = crossover(a, b)
            new_pop.extend([mutate(c1), mutate(c2)])
        pop = new_pop[:population_size]
        scores = [fitness_fn(m) for m in pop]

    best_idx = max(range(len(pop)), key=lambda i: scores[i])
    return pop[best_idx], scores[best_idx]


def tournament_select(ranked: List[tuple], k: int = 3) -> List[int]:
    import random as _r

    contestants = [_r.choice(ranked) for _ in range(k)]
    contestants.sort(key=lambda t: t[1], reverse=True)
    return contestants[0][0]

