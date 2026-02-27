"""
FormaCore AI - router/astar.py
A* pathfinder for single-net routing on a 2-layer PCB grid.
Manhattan movement only. Supports via transitions between layers.
"""
from __future__ import annotations

import heapq
from typing import Optional, List, Tuple, Dict

from router.cost import CostWeights, compute_cost

# Node: (x, y, layer)
Node = Tuple[int, int, int]

# 4-directional movement
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class AStarRouter:
    """A* pathfinder for single-net routing on a 2-layer grid."""

    def __init__(self, grid, weights: Optional[CostWeights] = None):
        self.grid = grid
        self.weights = weights or CostWeights()
        self._allowed: set = set()

    def route(self, start: Node, goal: Node,
              max_iter: int = 500_000) -> Optional[List[Node]]:
        """
        Find shortest path from start to goal.

        Returns:
            List of (x, y, layer) from start to goal, or None if unroutable.
        """
        if start == goal:
            return [start]

        # Pin cells may sit inside a component footprint (occupied).
        # The router must treat start and goal as walkable regardless.
        self._allowed = {start, goal}

        open_set: List[Tuple[float, int, Node]] = []
        counter = 0

        g_score: Dict[Node, float] = {start: 0.0}
        came_from: Dict[Node, Optional[Node]] = {start: None}

        h = self._heuristic(start, goal)
        heapq.heappush(open_set, (h, counter, start))
        counter += 1

        iterations = 0
        while open_set and iterations < max_iter:
            iterations += 1
            f_val, _, current = heapq.heappop(open_set)

            if current == goal:
                self._allowed = set()
                return self._reconstruct(came_from, current)

            # Skip if we already found a better path to this node
            if f_val > g_score.get(current, float('inf')) + self._heuristic(current, goal) + 1e-6:
                continue

            parent = came_from.get(current)

            for neighbor in self._neighbors(current):
                edge = self._edge_cost(current, neighbor, parent)
                tentative_g = g_score[current] + edge

                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1

        self._allowed = set()
        return None  # no path found

    def _heuristic(self, node: Node, goal: Node) -> float:
        """Manhattan distance + via penalty if layers differ."""
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        h = (dx + dy) * self.weights.step
        if node[2] != goal[2]:
            h += self.weights.via
        return h

    def _is_passable(self, x: int, y: int, layer: int) -> bool:
        """Check if a cell can be entered (free or explicitly allowed)."""
        if (x, y, layer) in self._allowed:
            return self.grid.in_bounds(x, y)
        return self.grid.is_free(x, y, layer)

    def _neighbors(self, node: Node) -> List[Node]:
        """4 cardinal directions on same layer + via to other layer."""
        x, y, layer = node
        result = []

        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if self._is_passable(nx, ny, layer):
                result.append((nx, ny, layer))

        # Via transition
        other = 1 - layer
        if self._is_passable(x, y, other):
            result.append((x, y, other))

        return result

    def _edge_cost(self, current: Node, neighbor: Node,
                   parent: Optional[Node]) -> float:
        """Cost of moving from current to neighbor."""
        nx, ny, nl = neighbor
        cx, cy, cl = current

        layer_change = (nl != cl)

        # Detect direction change (bend)
        direction_change = False
        if parent is not None and not layer_change:
            px, py, _ = parent
            prev_dx, prev_dy = cx - px, cy - py
            curr_dx, curr_dy = nx - cx, ny - cy
            if (prev_dx, prev_dy) != (curr_dx, curr_dy):
                direction_change = True

        return compute_cost(
            step_length=1.0,
            direction_change=direction_change,
            layer_change=layer_change,
            heat_value=float(self.grid.heat[nl, ny, nx]),
            congestion_value=int(self.grid.congestion[nl, ny, nx]),
            weights=self.weights,
        )

    def _reconstruct(self, came_from: Dict[Node, Optional[Node]],
                     current: Node) -> List[Node]:
        """Walk back from goal to start."""
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
