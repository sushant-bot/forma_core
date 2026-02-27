"""
FormaCore AI - ai/fitness.py
Evaluates a genome by routing all nets on a fresh grid copy
and computing a scalar fitness score (lower = better).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.grid import Grid, Net

from ai.genome import Genome


@dataclass
class FitnessWeights:
    """Weights for combining fitness sub-scores."""
    trace_length: float = 1.0
    via_count: float = 10.0
    congestion: float = 5.0
    failure_penalty: float = 1000.0
    bend_count: float = 0.5


class FitnessEvaluator:
    """
    Evaluates genomes by routing all nets from scratch.
    Each evaluation gets a clean copy of the grid.
    """

    def __init__(self, base_grid: 'Grid', nets: List['Net'],
                 weights: FitnessWeights = None):
        """
        Args:
            base_grid: Grid with components placed, heat applied, NO routes.
            nets:      Full list of nets to route.
            weights:   Fitness combination weights.
        """
        self.base_grid = base_grid
        self.nets = nets
        self.weights = weights or FitnessWeights()

    def evaluate(self, genome: Genome) -> float:
        """
        Route all nets using this genome's strategy, return fitness.

        Steps:
        1. Clone the base grid (clean slate)
        2. Create MultiNetRouter with genome's cost weights
        3. Route all nets in genome's net_order
        4. Compute weighted fitness (lower = better)
        """
        from router.multi_router import MultiNetRouter

        grid_copy = self.base_grid.clone()
        cost_weights = genome.to_cost_weights()
        router = MultiNetRouter(grid_copy, cost_weights)
        result = router.route_all(self.nets, net_order=genome.net_order)

        unrouted = len(result.failed_nets)
        fitness = (
            self.weights.trace_length * result.total_trace_length
            + self.weights.via_count * result.total_vias
            + self.weights.congestion * result.congestion_score
            + self.weights.failure_penalty * unrouted
            + self.weights.bend_count * result.total_bends
        )

        genome.fitness = fitness
        return fitness
