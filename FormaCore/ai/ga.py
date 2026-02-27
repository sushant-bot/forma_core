"""
FormaCore AI - ai/ga.py
Genetic Algorithm engine for optimizing PCB routing strategy.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Callable

from ai.genome import Genome
from ai.fitness import FitnessEvaluator


@dataclass
class GAConfig:
    """GA hyperparameters."""
    population_size: int = 15
    generations: int = 20
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.3
    weight_mutation_sigma: float = 2.0
    elitism_count: int = 2
    early_stop_gens: int = 5  # stop if no improvement for N gens


class GeneticAlgorithm:
    """GA optimizer for PCB routing strategy."""

    def __init__(self, evaluator: FitnessEvaluator,
                 config: Optional[GAConfig] = None,
                 on_generation: Optional[Callable] = None):
        """
        Args:
            evaluator:     FitnessEvaluator instance.
            config:        GA hyperparameters.
            on_generation: Callback(gen_num, best_genome, population)
        """
        self.evaluator = evaluator
        self.config = config or GAConfig()
        self.on_generation = on_generation
        self.num_nets = len(evaluator.nets)

    def run(self) -> Genome:
        """Execute GA and return best genome found."""
        population = self._init_population()

        # Evaluate initial population
        for g in population:
            self.evaluator.evaluate(g)

        best_ever = min(population, key=lambda g: g.fitness).copy()
        stale_count = 0  # generations without improvement

        for gen in range(self.config.generations):
            population.sort(key=lambda g: g.fitness)

            # Elitism: carry top genomes unchanged
            next_gen = [g.copy() for g in
                        population[:self.config.elitism_count]]

            # Fill rest via selection + crossover + mutation
            while len(next_gen) < self.config.population_size:
                p1 = self._tournament(population)
                p2 = self._tournament(population)

                if random.random() < self.config.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = p1.copy()

                if random.random() < self.config.mutation_rate:
                    self._mutate(child)

                self.evaluator.evaluate(child)
                next_gen.append(child)

            population = next_gen[:self.config.population_size]

            gen_best = min(population, key=lambda g: g.fitness)
            if gen_best.fitness < best_ever.fitness:
                best_ever = gen_best.copy()
                stale_count = 0
            else:
                stale_count += 1

            if self.on_generation:
                self.on_generation(gen, gen_best, population)

            # Early stopping
            if stale_count >= self.config.early_stop_gens:
                break

        return best_ever

    # -- Internal --

    def _init_population(self) -> List[Genome]:
        """Random population with one default-weights baseline."""
        pop = [Genome.random(self.num_nets)
               for _ in range(self.config.population_size)]
        pop[0] = Genome.default(self.num_nets)
        return pop

    def _tournament(self, pop: List[Genome]) -> Genome:
        """Tournament selection."""
        contestants = random.sample(
            pop, min(self.config.tournament_size, len(pop))
        )
        return min(contestants, key=lambda g: g.fitness)

    def _crossover(self, a: Genome, b: Genome) -> Genome:
        """
        Order Crossover (OX) for net_order,
        blend crossover for weights.
        """
        child_order = self._order_crossover(a.net_order, b.net_order)
        alpha = random.random()
        return Genome(
            net_order=child_order,
            distance_w=alpha * a.distance_w + (1 - alpha) * b.distance_w,
            bend_w=alpha * a.bend_w + (1 - alpha) * b.bend_w,
            via_w=alpha * a.via_w + (1 - alpha) * b.via_w,
            heat_w=alpha * a.heat_w + (1 - alpha) * b.heat_w,
            congestion_w=alpha * a.congestion_w + (1 - alpha) * b.congestion_w,
        )

    def _order_crossover(self, p1: List[int],
                         p2: List[int]) -> List[int]:
        """
        OX crossover for permutations:
        1. Copy a random substring from parent1
        2. Fill remaining slots from parent2 in order
        """
        n = len(p1)
        if n < 2:
            return list(p1)

        start, end = sorted(random.sample(range(n), 2))
        child = [None] * n
        child[start:end + 1] = p1[start:end + 1]

        taken = set(child[start:end + 1])
        fill = [x for x in p2 if x not in taken]
        idx = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1

        return child

    def _mutate(self, genome: Genome) -> None:
        """
        50% chance: swap two positions in net_order.
        50% chance: Gaussian perturbation on weights.
        """
        if random.random() < 0.5 and len(genome.net_order) >= 2:
            i, j = random.sample(range(len(genome.net_order)), 2)
            genome.net_order[i], genome.net_order[j] = (
                genome.net_order[j], genome.net_order[i]
            )
        else:
            s = self.config.weight_mutation_sigma
            genome.distance_w = max(0.1, genome.distance_w + random.gauss(0, s * 0.1))
            genome.bend_w = max(0.1, genome.bend_w + random.gauss(0, s))
            genome.via_w = max(1.0, genome.via_w + random.gauss(0, s * 2))
            genome.heat_w = max(0.1, genome.heat_w + random.gauss(0, s * 0.5))
            genome.congestion_w = max(0.1, genome.congestion_w + random.gauss(0, s))
