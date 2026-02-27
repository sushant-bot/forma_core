"""
FormaCore AI - core/grid.py
2-layer PCB grid representation using numpy arrays.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Optional, Dict


class Layer(IntEnum):
    TOP = 0
    BOTTOM = 1


@dataclass(frozen=True)
class Cell:
    """Read-only snapshot of a single grid cell."""
    x: int
    y: int
    layer: int
    occupied: bool
    heat: float
    congestion: int


@dataclass
class Component:
    """A placed component on the board."""
    name: str
    x: int              # grid col
    y: int              # grid row
    width: int          # cells wide
    height: int         # cells tall
    layer: Layer
    power_w: float = 0.0
    pins: List[Tuple[int, int]] = field(default_factory=list)  # relative (dx, dy)

    def absolute_pins(self) -> List[Tuple[int, int]]:
        """Pin positions in absolute grid coords."""
        return [(self.x + dx, self.y + dy) for dx, dy in self.pins]


@dataclass
class Net:
    """A connection between two or more pins."""
    name: str
    pins: List[Tuple[int, int, int]]  # (x, y, layer)


class Grid:
    """
    2-layer PCB grid.
    Data stored as numpy arrays for performance.
    Coordinate convention: (x=col, y=row), arrays indexed [layer, row, col].
    """

    def __init__(self, width: int, height: int, layers: int = 2,
                 resolution_mm: float = 0.5):
        self.width = width          # columns
        self.height = height        # rows
        self.num_layers = layers
        self.resolution = resolution_mm

        # Core data: shape = (layers, height, width)
        self.occupied = np.zeros((layers, height, width), dtype=np.int8)
        self.heat = np.zeros((layers, height, width), dtype=np.float32)
        self.congestion = np.zeros((layers, height, width), dtype=np.int32)

        # Metadata
        self.components: List[Component] = []
        self.routed_paths: Dict[str, List[Tuple[int, int, int]]] = {}

    # -- Queries --

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x: int, y: int, layer: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        return self.occupied[layer, y, x] == 0

    def get_cell(self, x: int, y: int, layer: int) -> Cell:
        return Cell(
            x=x, y=y, layer=layer,
            occupied=bool(self.occupied[layer, y, x]),
            heat=float(self.heat[layer, y, x]),
            congestion=int(self.congestion[layer, y, x]),
        )

    # -- Modifications --

    def place_component(self, comp: Component) -> None:
        """Mark component footprint cells as occupied."""
        ly = comp.layer
        for dy in range(comp.height):
            for dx in range(comp.width):
                cx, cy = comp.x + dx, comp.y + dy
                if self.in_bounds(cx, cy):
                    self.occupied[ly, cy, cx] = 1
        self.components.append(comp)

    def mark_trace(self, path: List[Tuple[int, int, int]],
                   net_name: str) -> None:
        """Mark all cells in a routed path as occupied, update congestion."""
        for (x, y, layer) in path:
            self.occupied[layer, y, x] = 1
            self.congestion[layer, y, x] += 1
        self.routed_paths[net_name] = list(path)

    def add_via(self, x: int, y: int) -> None:
        """Mark a via location on both layers."""
        for layer in range(self.num_layers):
            self.occupied[layer, y, x] = 1

    def clear_net(self, net_name: str) -> None:
        """Remove a previously routed net (rip-up)."""
        if net_name not in self.routed_paths:
            return
        for (x, y, layer) in self.routed_paths[net_name]:
            self.occupied[layer, y, x] = 0
            self.congestion[layer, y, x] = max(
                0, self.congestion[layer, y, x] - 1
            )
        del self.routed_paths[net_name]

    # -- Cloning --

    def clone(self) -> Grid:
        """Fast deep copy — clones numpy arrays, shallow-copies metadata."""
        g = Grid.__new__(Grid)
        g.width = self.width
        g.height = self.height
        g.num_layers = self.num_layers
        g.resolution = self.resolution
        g.occupied = self.occupied.copy()
        g.heat = self.heat.copy()
        g.congestion = self.congestion.copy()
        g.components = list(self.components)
        g.routed_paths = {}
        return g
