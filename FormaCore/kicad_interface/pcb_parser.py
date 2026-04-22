"""
FormaCore AI - kicad_interface/pcb_parser.py
Parses a subset of KiCad .kicad_pcb files for proof-mode imports.

This parser is intentionally conservative. It accepts footprint/pad/net
boards that can be mapped into FormaCore's simplified 2-layer routing
model and rejects empty or unsupported boards with a clear error.
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple

from core.grid import Grid, Component, Net, Layer
from errors import ParseError


logger = logging.getLogger(__name__)


def parse_kicad_pcb(content: str) -> tuple[Grid, List[Net]]:
    """Parse a .kicad_pcb file and extract components + nets.

    Supported input is limited to board files that contain footprint blocks
    and net/pad connectivity. Header-only files or layout formats outside
    this subset are rejected with ParseError so the UI can show a precise
    explanation instead of silently producing an empty board.
    """
    if not content or not content.strip():
        raise ParseError("KiCad PCB file is empty.")

    text = content.strip()
    if not text.lstrip().startswith("(kicad_pcb"):
        raise ParseError("Not a valid KiCad PCB file.")

    width_mm, height_mm = 50.0, 40.0

    edge_coords = re.findall(
        r'\(gr_line\s+\(start\s+([\d.]+)\s+([\d.]+)\)\s*'
        r'\(end\s+([\d.]+)\s+([\d.]+)\).*?Edge\.Cuts',
        text, re.DOTALL
    )
    if not edge_coords:
        edge_coords = re.findall(
            r'\(gr_rect\s+\(start\s+([\d.]+)\s+([\d.]+)\)\s*'
            r'\(end\s+([\d.]+)\s+([\d.]+)\)',
            text, re.DOTALL
        )

    if edge_coords:
        all_x, all_y = [], []
        for x1, y1, x2, y2 in edge_coords:
            all_x.extend([float(x1), float(x2)])
            all_y.extend([float(y1), float(y2)])
        if all_x and all_y:
            width_mm = max(all_x) - min(all_x)
            height_mm = max(all_y) - min(all_y)

    resolution = 0.5
    grid_w = max(40, int(width_mm / resolution))
    grid_h = max(30, int(height_mm / resolution))
    grid = Grid(width=grid_w, height=grid_h, layers=2, resolution_mm=resolution)

    fp_blocks = re.findall(
        r'\(footprint\s+"([^"]*)".*?\(at\s+([\d.]+)\s+([\d.]+).*?\)',
        text, re.DOTALL
    )

    net_defs = re.findall(r'\(net\s+(\d+)\s+"([^"]*)"\)', text)
    net_map = {nid: name for nid, name in net_defs if name.strip()}

    pad_blocks = re.findall(
        r'\(pad\s+"[^"]*"\s+\w+\s+\w+\s+\(at\s+([\d.]+)\s+([\d.]+).*?\)'
        r'.*?\(net\s+(\d+)',
        text, re.DOTALL
    )

    if not fp_blocks and not net_defs and not pad_blocks:
        raise ParseError(
            "KiCad PCB file appears empty or unsupported: no footprints, pads, or nets were found."
        )

    if not fp_blocks:
        raise ParseError(
            "KiCad PCB file contains no footprints. FormaCore's PCB importer currently requires footprint blocks."
        )

    components = []
    for name, x_str, y_str in fp_blocks:
        short_name = name.split(':')[-1][:8] if ':' in name else name[:8]
        gx = int(float(x_str) / resolution) % grid_w
        gy = int(float(y_str) / resolution) % grid_h
        comp = Component(
            name=short_name, x=gx, y=gy,
            width=4, height=3, layer=Layer.TOP,
            power_w=0.0,
            pins=[(0, 0), (3, 0), (0, 2), (3, 2)]
        )
        grid.place_component(comp)
        components.append(comp)

    pad_nets = {}
    for px, py, nid in pad_blocks:
        if nid in net_map and net_map[nid]:
            gx = int(float(px) / resolution) % grid_w
            gy = int(float(py) / resolution) % grid_h
            pad_nets.setdefault(nid, []).append((gx, gy, 0))

    nets = []
    for nid, pins in pad_nets.items():
        if len(pins) >= 2:
            name = net_map.get(nid, f"net_{nid}")
            unique_pins = list(dict.fromkeys(pins))
            if len(unique_pins) >= 2:
                nets.append(Net(name, unique_pins[:2]))

    if not nets:
        raise ParseError(
            "KiCad PCB file has footprints but no routable nets. FormaCore currently supports footprint/pad-based net extraction only."
        )

    logger.info(
        "Parsed KiCad PCB: %d footprints, %d nets, board %dx%d cells",
        len(components), len(nets), grid_w, grid_h,
    )
    return grid, nets