"""
FormaCore AI - core/heat.py
Gaussian thermal spread model for PCB heat sources.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.grid import Grid


class HeatSource:
    """A point heat source on the board."""
    __slots__ = ('x', 'y', 'intensity')

    def __init__(self, x: int, y: int, intensity: float):
        self.x = x
        self.y = y
        self.intensity = intensity


class HeatModel:
    """
    Generates a heat map by applying Gaussian decay from heat sources.
    Must be applied before routing so A* cost can factor in thermal cost.
    """

    def __init__(self, sigma: float = 10.0):
        """
        Args:
            sigma: Gaussian spread radius in grid cells.
                   Larger = heat spreads further.
        """
        self.sigma = sigma

    def apply(self, grid: Grid) -> None:
        """
        Compute heat map from all placed components with power_w > 0.
        Writes result into grid.heat on both layers.
        """
        grid.heat[:] = 0.0

        sources = self._extract_sources(grid)
        if not sources:
            return

        radius = int(3 * self.sigma) + 1

        for src in sources:
            self._apply_source(grid, src, radius)

    def apply_sources(self, grid: Grid,
                      sources: List[HeatSource]) -> None:
        """Apply explicit heat sources (for manual/test usage)."""
        grid.heat[:] = 0.0
        radius = int(3 * self.sigma) + 1
        for src in sources:
            self._apply_source(grid, src, radius)

    def _extract_sources(self, grid: Grid) -> List[HeatSource]:
        """Extract heat sources from placed components."""
        sources = []
        for comp in grid.components:
            if comp.power_w > 0:
                cx = comp.x + comp.width // 2
                cy = comp.y + comp.height // 2
                sources.append(HeatSource(cx, cy, comp.power_w))
        return sources

    def _apply_source(self, grid: Grid, src: HeatSource,
                      radius: int) -> None:
        """Apply Gaussian spread from a single heat source."""
        x_min = max(0, src.x - radius)
        x_max = min(grid.width, src.x + radius + 1)
        y_min = max(0, src.y - radius)
        y_max = min(grid.height, src.y + radius + 1)

        xs = np.arange(x_min, x_max)
        ys = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(xs, ys)

        dist_sq = (xx - src.x) ** 2 + (yy - src.y) ** 2
        gaussian = src.intensity * np.exp(
            -dist_sq / (2 * self.sigma ** 2)
        )

        # Heat conducts through both layers
        for layer in range(grid.num_layers):
            grid.heat[layer, y_min:y_max, x_min:x_max] += gaussian

    def get_max_heat(self, grid: Grid) -> float:
        """Peak heat value on the grid."""
        return float(np.max(grid.heat))
