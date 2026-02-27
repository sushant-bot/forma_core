"""
FormaCore AI - ai/genome.py
Genome representation for the genetic algorithm.
Encodes routing strategy: net ordering + cost function weights.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List
from copy import deepcopy

from router.cost import CostWeights


@dataclass
class Genome:
    """
    A routing strategy encoded for GA optimization.

    net_order:    Permutation of net indices — controls which nets
                  get routed first (and thus get open space).
    distance_w .. congestion_w:  Cost function weights passed to A*.
    fitness:      Cached score (lower = better). Inf = not evaluated.
    """
    net_order: List[int]
    distance_w: float = 1.0
    bend_w: float = 5.0
    via_w: float = 20.0
    heat_w: float = 2.0
    congestion_w: float = 3.0
    fitness: float = float('inf')

    @staticmethod
    def random(num_nets: int) -> Genome:
        """Create a genome with random order and weights."""
        order = list(range(num_nets))
        random.shuffle(order)
        return Genome(
            net_order=order,
            distance_w=random.uniform(0.5, 3.0),
            bend_w=random.uniform(1.0, 15.0),
            via_w=random.uniform(5.0, 50.0),
            heat_w=random.uniform(0.5, 5.0),
            congestion_w=random.uniform(0.5, 8.0),
        )

    @staticmethod
    def default(num_nets: int) -> Genome:
        """Create a genome with default weights and natural order."""
        return Genome(net_order=list(range(num_nets)))

    def to_cost_weights(self) -> CostWeights:
        """Convert to CostWeights for the A* router."""
        return CostWeights(
            step=self.distance_w,
            bend=self.bend_w,
            via=self.via_w,
            heat=self.heat_w,
            congestion=self.congestion_w,
        )

    def copy(self) -> Genome:
        return deepcopy(self)
