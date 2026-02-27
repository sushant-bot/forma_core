"""
FormaCore AI - router/multi_router.py
Sequential multi-net router. Routes nets one at a time,
marking cells occupied after each.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from core.grid import Grid, Net
from router.astar import AStarRouter, Node
from router.cost import CostWeights


@dataclass
class RoutingResult:
    """Aggregate result of routing all nets."""
    paths: Dict[str, List[Node]]
    failed_nets: List[str]
    total_trace_length: int
    total_vias: int
    total_bends: int
    congestion_score: float
    completion_rate: float


class MultiNetRouter:
    """Routes multiple nets sequentially with configurable ordering."""

    def __init__(self, grid: Grid,
                 weights: Optional[CostWeights] = None):
        self.grid = grid
        self.weights = weights or CostWeights()

    def route_all(self, nets: List[Net],
                  net_order: Optional[List[int]] = None) -> RoutingResult:
        """
        Route all nets in the given order.

        Args:
            nets:      List of Net objects.
            net_order: Permutation of indices into nets.
                       If None, routes in natural order.

        Returns:
            RoutingResult with paths and metrics.
        """
        router = AStarRouter(self.grid, self.weights)
        order = net_order if net_order is not None else list(range(len(nets)))

        paths: Dict[str, List[Node]] = {}
        failed: List[str] = []

        for idx in order:
            net = nets[idx]
            full_path: List[Node] = []
            success = True

            # Route pin-to-pin (chain for multi-pin nets)
            for i in range(len(net.pins) - 1):
                start = tuple(net.pins[i])
                goal = tuple(net.pins[i + 1])
                segment = router.route(start, goal)

                if segment is None:
                    success = False
                    break

                if full_path and segment:
                    full_path.extend(segment[1:])  # avoid duplicating junction
                else:
                    full_path.extend(segment)

            if success and full_path:
                paths[net.name] = full_path
                self.grid.mark_trace(full_path, net.name)
            else:
                failed.append(net.name)

        return self._compute_metrics(paths, failed, nets)

    def _compute_metrics(self, paths: Dict[str, List[Node]],
                         failed: List[str],
                         nets: List[Net]) -> RoutingResult:
        """Compute aggregate routing metrics."""
        total_length = 0
        total_vias = 0
        total_bends = 0

        for name, path in paths.items():
            total_length += len(path) - 1  # steps

            for i in range(1, len(path)):
                if path[i][2] != path[i - 1][2]:
                    total_vias += 1

            for i in range(2, len(path)):
                dx1 = path[i - 1][0] - path[i - 2][0]
                dy1 = path[i - 1][1] - path[i - 2][1]
                dx2 = path[i][0] - path[i - 1][0]
                dy2 = path[i][1] - path[i - 1][1]
                if (dx1, dy1) != (dx2, dy2):
                    total_bends += 1

        congestion_score = float(
            np.sum(np.maximum(self.grid.congestion - 1, 0))
        )

        total_nets = len(nets)
        routed = total_nets - len(failed)

        return RoutingResult(
            paths=paths,
            failed_nets=failed,
            total_trace_length=total_length,
            total_vias=total_vias,
            total_bends=total_bends,
            congestion_score=congestion_score,
            completion_rate=routed / total_nets if total_nets > 0 else 1.0,
        )
