"""
FormaCore AI - router/cost.py
Configurable cost function for A* routing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostWeights:
    """Tunable weights for the routing cost function."""
    step: float = 1.0
    bend: float = 5.0
    via: float = 20.0
    heat: float = 2.0
    congestion: float = 3.0


def compute_cost(
    step_length: float,
    direction_change: bool,
    layer_change: bool,
    heat_value: float,
    congestion_value: int,
    weights: CostWeights,
) -> float:
    """
    Compute traversal cost for a single routing step.

    Args:
        step_length:       Base distance cost (usually 1.0).
        direction_change:  True if trace bends at this point.
        layer_change:      True if this is a via transition.
        heat_value:        Heat map value at the target cell.
        congestion_value:  Congestion counter at the target cell.
        weights:           Weight configuration.

    Returns:
        Total cost for this step.
    """
    cost = step_length * weights.step

    if direction_change:
        cost += weights.bend

    if layer_change:
        cost += weights.via

    cost += heat_value * weights.heat
    cost += congestion_value * weights.congestion

    return cost
