"""
FormaCore AI - executor/executor.py
Deterministic board operations. Every function takes a Grid,
validates inputs, applies the change in-memory, returns a result dict.
No file I/O. No side effects beyond the passed-in grid.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

from core.grid import Grid, Component, Net, Layer
from core.heat import HeatModel
from router.cost import CostWeights
from router.multi_router import MultiNetRouter, RoutingResult


# ------------------------------------------------------------------ #
# RESULT HELPERS
# ------------------------------------------------------------------ #

def _ok(**kwargs) -> Dict[str, Any]:
    result = {"status": "ok"}
    result.update(kwargs)
    return result


def _error(msg: str) -> Dict[str, Any]:
    return {"status": "error", "msg": msg}


# ------------------------------------------------------------------ #
# COMPONENT OPERATIONS
# ------------------------------------------------------------------ #

def place_component(grid: Grid, ref: str, x: int, y: int,
                    width: int, height: int,
                    layer: int = 0, power_w: float = 0.0,
                    pins: Optional[List[Tuple[int, int]]] = None
                    ) -> Dict[str, Any]:
    """Place a new component on the board."""
    if not grid.in_bounds(x, y):
        return _error(f"Position ({x},{y}) out of bounds")
    if not grid.in_bounds(x + width - 1, y + height - 1):
        return _error(f"Component extends beyond board edge")

    # Check name collision
    for comp in grid.components:
        if comp.name == ref:
            return _error(f"Component '{ref}' already exists")

    comp = Component(
        name=ref, x=x, y=y, width=width, height=height,
        layer=Layer(layer), power_w=power_w,
        pins=pins or [(0, 0), (width - 1, 0)],
    )
    grid.place_component(comp)
    return _ok(component=ref)


def move_component(grid: Grid, ref: str, x: int, y: int
                   ) -> Dict[str, Any]:
    """Move an existing component to new position."""
    comp = _find_component(grid, ref)
    if comp is None:
        return _error(f"Component '{ref}' not found")

    if not grid.in_bounds(x, y):
        return _error(f"Position ({x},{y}) out of bounds")
    if not grid.in_bounds(x + comp.width - 1, y + comp.height - 1):
        return _error(f"Component would extend beyond board edge")

    # Clear old footprint
    _clear_footprint(grid, comp)

    # Update position
    old_x, old_y = comp.x, comp.y
    comp.x = x
    comp.y = y

    # Re-stamp footprint
    _stamp_footprint(grid, comp)

    return _ok(component=ref, old_pos=(old_x, old_y), new_pos=(x, y))


def rotate_component(grid: Grid, ref: str,
                     rotation_deg: int) -> Dict[str, Any]:
    """Rotate a component (swap width/height for 90/270)."""
    comp = _find_component(grid, ref)
    if comp is None:
        return _error(f"Component '{ref}' not found")

    if rotation_deg not in (0, 90, 180, 270):
        return _error(f"Rotation must be 0, 90, 180, or 270")

    _clear_footprint(grid, comp)

    if rotation_deg in (90, 270):
        comp.width, comp.height = comp.height, comp.width
        # Rotate pin positions
        if rotation_deg == 90:
            comp.pins = [(dy, comp.width - 1 - dx)
                         for dx, dy in comp.pins]
        else:
            comp.pins = [(comp.height - 1 - dy, dx)
                         for dx, dy in comp.pins]

    if not grid.in_bounds(comp.x + comp.width - 1,
                          comp.y + comp.height - 1):
        # Undo rotation if it goes out of bounds
        if rotation_deg in (90, 270):
            comp.width, comp.height = comp.height, comp.width
        _stamp_footprint(grid, comp)
        return _error("Rotated component extends beyond board edge")

    _stamp_footprint(grid, comp)
    return _ok(component=ref, rotation=rotation_deg)


def remove_component(grid: Grid, ref: str) -> Dict[str, Any]:
    """Remove a component from the board."""
    comp = _find_component(grid, ref)
    if comp is None:
        return _error(f"Component '{ref}' not found")

    _clear_footprint(grid, comp)
    grid.components.remove(comp)
    return _ok(component=ref)


def move_group(grid: Grid, refs: List[str],
               dx: int, dy: int) -> Dict[str, Any]:
    """Move a group of components by (dx, dy) offset."""
    comps = []
    for ref in refs:
        comp = _find_component(grid, ref)
        if comp is None:
            return _error(f"Component '{ref}' not found")
        # Check destination bounds
        nx, ny = comp.x + dx, comp.y + dy
        if not grid.in_bounds(nx, ny) or not grid.in_bounds(
                nx + comp.width - 1, ny + comp.height - 1):
            return _error(
                f"Component '{ref}' would extend beyond board at "
                f"({nx},{ny})")
        comps.append(comp)

    # All validated — apply
    for comp in comps:
        _clear_footprint(grid, comp)

    for comp in comps:
        comp.x += dx
        comp.y += dy
        _stamp_footprint(grid, comp)

    return _ok(moved=refs, offset=(dx, dy))


# ------------------------------------------------------------------ #
# ROUTING OPERATIONS
# ------------------------------------------------------------------ #

def route_all_nets(grid: Grid, nets: List[Net],
                   weights: Optional[CostWeights] = None,
                   net_order: Optional[List[int]] = None
                   ) -> Dict[str, Any]:
    """Route all nets with given weights and order."""
    # Clear existing routes first
    for net_name in list(grid.routed_paths.keys()):
        grid.clear_net(net_name)

    w = weights or CostWeights()
    router = MultiNetRouter(grid, w)
    result = router.route_all(nets, net_order=net_order)

    return _ok(
        routed=len(result.paths),
        failed=result.failed_nets,
        total=len(nets),
        trace_length=result.total_trace_length,
        vias=result.total_vias,
        bends=result.total_bends,
        congestion=result.congestion_score,
    )


def clear_all_routes(grid: Grid) -> Dict[str, Any]:
    """Remove all routed traces from the board."""
    names = list(grid.routed_paths.keys())
    for name in names:
        grid.clear_net(name)
    return _ok(cleared=len(names))


def clear_net(grid: Grid, net_name: str) -> Dict[str, Any]:
    """Remove a specific routed net."""
    if net_name not in grid.routed_paths:
        return _error(f"Net '{net_name}' not routed")
    grid.clear_net(net_name)
    return _ok(net=net_name)


# ------------------------------------------------------------------ #
# HEAT / ANALYSIS
# ------------------------------------------------------------------ #

def apply_heat(grid: Grid, sigma: float = 10.0) -> Dict[str, Any]:
    """Apply thermal model to the board."""
    heat = HeatModel(sigma=sigma)
    heat.apply(grid)
    peak = heat.get_max_heat(grid)
    return _ok(peak_heat=round(peak, 4))


def get_board_info(grid: Grid) -> Dict[str, Any]:
    """Return a summary of the current board state."""
    return _ok(
        width=grid.width,
        height=grid.height,
        width_mm=grid.width * grid.resolution,
        height_mm=grid.height * grid.resolution,
        layers=grid.num_layers,
        resolution_mm=grid.resolution,
        components=len(grid.components),
        routed_nets=len(grid.routed_paths),
        component_list=[
            {"name": c.name, "x": c.x, "y": c.y,
             "width": c.width, "height": c.height,
             "layer": int(c.layer), "power_w": c.power_w}
            for c in grid.components
        ],
    )


def get_drc_report(grid: Grid, nets: List[Net]) -> Dict[str, Any]:
    """Run basic Design Rule Check on current board state."""
    issues = []

    # Check component overlaps
    for i, a in enumerate(grid.components):
        for b in grid.components[i + 1:]:
            if a.layer != b.layer:
                continue
            if _rects_overlap(a, b):
                issues.append({
                    "type": "overlap",
                    "severity": "error",
                    "msg": f"Components '{a.name}' and '{b.name}' overlap",
                })

    # Check components in bounds
    for comp in grid.components:
        if not grid.in_bounds(comp.x + comp.width - 1,
                              comp.y + comp.height - 1):
            issues.append({
                "type": "out_of_bounds",
                "severity": "error",
                "msg": f"Component '{comp.name}' extends beyond board",
            })

    # Check unrouted nets
    routed_names = set(grid.routed_paths.keys())
    for net in nets:
        if net.name not in routed_names:
            issues.append({
                "type": "unrouted",
                "severity": "warning",
                "msg": f"Net '{net.name}' is not routed",
            })

    errors = sum(1 for i in issues if i["severity"] == "error")
    warnings = sum(1 for i in issues if i["severity"] == "warning")

    return _ok(
        passed=(errors == 0),
        errors=errors,
        warnings=warnings,
        issues=issues,
    )


# ------------------------------------------------------------------ #
# SNAPSHOT
# ------------------------------------------------------------------ #

def snapshot_board(grid: Grid) -> Grid:
    """Return a deep copy of the current board state."""
    return grid.clone()


# ------------------------------------------------------------------ #
# INTERNAL HELPERS
# ------------------------------------------------------------------ #

def _find_component(grid: Grid, ref: str) -> Optional[Component]:
    for comp in grid.components:
        if comp.name == ref:
            return comp
    return None


def _clear_footprint(grid: Grid, comp: Component) -> None:
    """Un-stamp a component's occupied cells."""
    ly = comp.layer
    for dy in range(comp.height):
        for dx in range(comp.width):
            cx, cy = comp.x + dx, comp.y + dy
            if grid.in_bounds(cx, cy):
                grid.occupied[ly, cy, cx] = 0


def _stamp_footprint(grid: Grid, comp: Component) -> None:
    """Stamp a component's occupied cells."""
    ly = comp.layer
    for dy in range(comp.height):
        for dx in range(comp.width):
            cx, cy = comp.x + dx, comp.y + dy
            if grid.in_bounds(cx, cy):
                grid.occupied[ly, cy, cx] = 1


def _rects_overlap(a: Component, b: Component) -> bool:
    """Check if two component bounding boxes overlap."""
    return not (a.x + a.width <= b.x or b.x + b.width <= a.x or
                a.y + a.height <= b.y or b.y + b.height <= a.y)
