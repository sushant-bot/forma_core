"""
FormaCore AI - core/congestion.py
Congestion tracking and analysis.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.grid import Grid


class CongestionTracker:
    """Analyzes congestion on the routing grid."""

    def __init__(self, grid: Grid):
        self.grid = grid

    def get_score(self) -> float:
        """
        Aggregate congestion score.
        Cells with congestion <= 1 are normal (single trace).
        Penalty grows quadratically for higher values.
        """
        excess = np.maximum(self.grid.congestion - 1, 0).astype(np.float32)
        return float(np.sum(excess ** 2))

    def get_hotspots(self, threshold: int = 3) -> List[Tuple[int, int, int]]:
        """Find cells where congestion >= threshold."""
        locations = np.argwhere(self.grid.congestion >= threshold)
        return [(int(c), int(r), int(l)) for l, r, c in locations]

    def get_density_map(self, layer: int) -> np.ndarray:
        """2D congestion array for a specific layer (for visualization)."""
        return self.grid.congestion[layer].copy()
