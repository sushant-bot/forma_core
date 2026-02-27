"""
FormaCore AI - kicad_interface/converter.py
Converts a parsed KiCad schematic into FormaCore Grid, Components, and Nets.
Since schematics don't contain PCB layout positions, components are
auto-placed on the grid using a simple packing algorithm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from core.grid import Grid, Component, Net, Layer
from kicad_interface.parser import KiSchematic, KiComponent


# ------------------------------------------------------------------ #
# Component sizing heuristics
# ------------------------------------------------------------------ #

# Map library prefixes to approximate grid sizes (width, height) and power
_SIZE_MAP = {
    # ICs / MCUs
    'NUCLEO':       (16, 20, 1.0),
    'TB6600':       (14, 10, 2.5),
    'XL4015':       (8, 6, 1.5),
    'MP1584':       (6, 5, 1.0),
    'LD1117':       (5, 4, 0.5),
    # Passives
    'Device:C_Pol': (3, 2, 0.0),
    'Device:C':     (2, 2, 0.0),
    'Device:R':     (2, 3, 0.0),
    'Device:L':     (3, 2, 0.0),
    'Device:LED':   (2, 2, 0.02),
    'Device:Fuse':  (3, 2, 0.0),
    'Device:R_Pot': (4, 4, 0.0),
    'Device:Q_':    (3, 3, 0.2),
    # Connectors
    'Connector:Screw_Terminal_01x05': (6, 4, 0.0),
    'Connector:Screw_Terminal_01x04': (5, 4, 0.0),
    'Connector:Screw_Terminal_01x03': (4, 3, 0.0),
    'Connector:Screw_Terminal_01x02': (3, 3, 0.0),
    'Connector:Conn_01x05':          (6, 4, 0.0),
    'Connector:TestPoint':           (2, 2, 0.0),
    'XT60':         (5, 4, 0.0),
    'SMBJ':         (3, 2, 0.0),
    # Switches
    'Switch:SW_D':  (5, 4, 0.0),
    'Switch:SW_P':  (3, 3, 0.0),
    # Mechanical
    'Mechanical:':  (3, 3, 0.0),
}


def _estimate_size(comp: KiComponent) -> Tuple[int, int, float]:
    """Estimate grid size (w, h) and power for a component."""
    lib_id = comp.lib_id
    ref = comp.reference.upper()

    for prefix, (w, h, pwr) in _SIZE_MAP.items():
        if prefix in lib_id or prefix in ref:
            return w, h, pwr

    # Default based on pin count
    n_pins = len(comp.pin_numbers)
    if n_pins <= 2:
        return 2, 2, 0.0
    elif n_pins <= 6:
        return 4, 3, 0.0
    elif n_pins <= 12:
        return 6, 5, 0.1
    else:
        side = max(4, int(math.sqrt(n_pins)) + 2)
        return side, side, 0.2


# ------------------------------------------------------------------ #
# Auto-placer (simple row-based bin packing)
# ------------------------------------------------------------------ #

def _auto_place(components: List[Tuple[str, int, int, float, int]],
                board_width: int, board_height: int,
                margin: int = 3) -> List[Tuple[int, int]]:
    """
    Place components on the grid using row-based bin packing.
    Input: list of (name, width, height, power, pin_count)
    Returns: list of (x, y) positions in grid coords.
    """
    # Sort: large components first
    indexed = list(enumerate(components))
    indexed.sort(key=lambda t: -(t[1][1] * t[1][2]))

    positions = [None] * len(components)
    placed_rects = []

    def overlaps(x, y, w, h):
        for px, py, pw, ph in placed_rects:
            if not (x + w + margin <= px or px + pw + margin <= x or
                    y + h + margin <= py or py + ph + margin <= y):
                return True
        return False

    cur_x = margin
    cur_y = margin
    row_height = 0

    for idx, (name, w, h, pwr, pins) in indexed:
        placed = False
        # Try current row position
        if cur_x + w + margin <= board_width and not overlaps(cur_x, cur_y, w, h):
            positions[idx] = (cur_x, cur_y)
            placed_rects.append((cur_x, cur_y, w, h))
            row_height = max(row_height, h)
            cur_x += w + margin
            placed = True
        else:
            # Try next row
            cur_x = margin
            cur_y += row_height + margin
            row_height = 0
            if cur_y + h + margin <= board_height and not overlaps(cur_x, cur_y, w, h):
                positions[idx] = (cur_x, cur_y)
                placed_rects.append((cur_x, cur_y, w, h))
                row_height = max(row_height, h)
                cur_x += w + margin
                placed = True

        if not placed:
            # Fallback: scan entire grid
            for sy in range(margin, board_height - h - margin, 2):
                for sx in range(margin, board_width - w - margin, 2):
                    if not overlaps(sx, sy, w, h):
                        positions[idx] = (sx, sy)
                        placed_rects.append((sx, sy, w, h))
                        placed = True
                        break
                if placed:
                    break

        if not placed:
            # Last resort: place at margin
            positions[idx] = (margin, margin)

    return positions


# ------------------------------------------------------------------ #
# Generate pins on component edges
# ------------------------------------------------------------------ #

def _generate_pins(width: int, height: int, n_pins: int) -> List[Tuple[int, int]]:
    """Generate pin positions along component edges."""
    pins = []
    if n_pins <= 0:
        return pins
    if n_pins == 1:
        return [(0, 0)]
    if n_pins == 2:
        return [(0, 0), (width - 1, 0)]

    # Distribute pins around the perimeter
    perimeter_cells = []
    # Top edge (left to right)
    for x in range(width):
        perimeter_cells.append((x, 0))
    # Right edge (top+1 to bottom)
    for y in range(1, height):
        perimeter_cells.append((width - 1, y))
    # Bottom edge (right-1 to left)
    for x in range(width - 2, -1, -1):
        perimeter_cells.append((x, height - 1))
    # Left edge (bottom-1 to top+1)
    for y in range(height - 2, 0, -1):
        perimeter_cells.append((0, y))

    if n_pins >= len(perimeter_cells):
        return perimeter_cells[:n_pins]

    # Space pins evenly
    step = max(1, len(perimeter_cells) / n_pins)
    for i in range(n_pins):
        idx = int(i * step) % len(perimeter_cells)
        pins.append(perimeter_cells[idx])

    return pins


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

@dataclass
class ConversionResult:
    """Result of converting a KiCad schematic to FormaCore objects."""
    grid: Grid
    components: List[Component]
    nets: List[Net]
    skipped_components: List[str]  # refs of components not placed
    skipped_nets: List[str]        # net names with < 2 pins
    stats: Dict


def convert_schematic(
    sch: KiSchematic,
    resolution_mm: float = 0.5,
    board_margin_mm: float = 5.0,
    max_components: int = 60,
) -> ConversionResult:
    """
    Convert a parsed KiCad schematic to FormaCore Grid + Components + Nets.

    Args:
        sch: Parsed KiCad schematic
        resolution_mm: Grid resolution in mm
        board_margin_mm: Margin around board edges

    Returns:
        ConversionResult with grid, components, nets, and stats.
    """
    # Filter out non-board components (power symbols, flags, mounting holes)
    board_comps = []
    skipped = []
    for comp in sch.components:
        ref = comp.reference
        lib = comp.lib_id.lower()

        # Skip power symbols, flags, mounting holes
        if ref.startswith('#') or ref.startswith('FLG'):
            skipped.append(ref)
            continue
        if 'pwr_flag' in lib or 'power:' in lib:
            skipped.append(ref)
            continue
        if 'mounting' in lib.lower() or 'mechanical:' in lib.lower():
            skipped.append(ref)
            continue

        board_comps.append(comp)

    # Limit components for routing performance
    if len(board_comps) > max_components:
        # Keep most connected components
        board_comps = board_comps[:max_components]

    # Estimate sizes
    comp_specs = []
    for comp in board_comps:
        w, h, pwr = _estimate_size(comp)
        n_pins = len(comp.pin_numbers)
        comp_specs.append((comp.reference, w, h, pwr, n_pins))

    # Calculate board size from total component area
    total_area = sum(w * h for _, w, h, _, _ in comp_specs)
    # Board should be ~3x component area for routing space
    target_area = total_area * 3.5
    aspect = 1.3  # slightly wide
    board_h_cells = max(40, int(math.sqrt(target_area / aspect)))
    board_w_cells = max(50, int(board_h_cells * aspect))
    # Round up
    board_w_cells = min(board_w_cells, 200)
    board_h_cells = min(board_h_cells, 160)

    # Auto-place
    positions = _auto_place(comp_specs, board_w_cells, board_h_cells)

    # Create grid
    grid = Grid(
        width=board_w_cells,
        height=board_h_cells,
        layers=2,
        resolution_mm=resolution_mm,
    )

    # Create FormaCore components
    fc_components = []
    comp_pin_map = {}  # "REF.pin_num" -> (abs_x, abs_y, layer)

    for i, (kcomp, (cx, cy)) in enumerate(zip(board_comps, positions)):
        w, h, pwr = _estimate_size(kcomp)
        n_pins = len(kcomp.pin_numbers)
        pins = _generate_pins(w, h, max(n_pins, 2))

        layer = Layer.TOP if i % 5 != 4 else Layer.BOTTOM

        fc_comp = Component(
            name=kcomp.reference,
            x=cx, y=cy,
            width=w, height=h,
            layer=layer,
            power_w=pwr,
            pins=pins,
        )
        grid.place_component(fc_comp)
        fc_components.append(fc_comp)

        # Map pin numbers to absolute positions
        for j, pin_num in enumerate(kcomp.pin_numbers):
            if j < len(pins):
                abs_x = cx + pins[j][0]
                abs_y = cy + pins[j][1]
                key = f"{kcomp.reference}.{pin_num}"
                comp_pin_map[key] = (abs_x, abs_y, int(layer))

    # Create nets from schematic net assignments
    fc_nets = []
    skipped_nets = []

    for net_name, pin_refs in sch.nets.items():
        if not pin_refs:
            continue

        # Resolve pin refs to grid coordinates
        pin_coords = []
        seen = set()
        for pref in pin_refs:
            if pref in comp_pin_map and pref not in seen:
                pin_coords.append(comp_pin_map[pref])
                seen.add(pref)

        # Need at least 2 pins for a net
        if len(pin_coords) >= 2:
            # Deduplicate by position
            unique = list(set(pin_coords))
            if len(unique) >= 2:
                fc_nets.append(Net(name=net_name, pins=unique))
            else:
                skipped_nets.append(net_name)
        else:
            skipped_nets.append(net_name)

    # Sort nets: shorter nets first (easier to route)
    def net_length(net):
        if len(net.pins) < 2:
            return 0
        xs = [p[0] for p in net.pins]
        ys = [p[1] for p in net.pins]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    fc_nets.sort(key=net_length)

    stats = {
        'schematic_title': sch.title,
        'schematic_company': sch.company,
        'total_sch_components': len(sch.components),
        'board_components': len(fc_components),
        'skipped_components': len(skipped),
        'board_size': f"{board_w_cells}x{board_h_cells}",
        'board_mm': f"{board_w_cells * resolution_mm:.0f}x{board_h_cells * resolution_mm:.0f}mm",
        'total_nets': len(fc_nets),
        'skipped_nets': len(skipped_nets),
        'total_pins_mapped': len(comp_pin_map),
    }

    return ConversionResult(
        grid=grid,
        components=fc_components,
        nets=fc_nets,
        skipped_components=skipped,
        skipped_nets=skipped_nets,
        stats=stats,
    )
