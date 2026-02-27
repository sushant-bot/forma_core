"""
FormaCore AI - kicad_interface/parser.py
Parses KiCad 8.x .kicad_sch files (S-expression format).
Extracts components, net labels, and wire connectivity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class KiPin:
    """A pin on a library symbol."""
    number: str
    name: str
    x: float
    y: float
    length: float = 2.54


@dataclass
class KiLibSymbol:
    """A library symbol definition (template)."""
    lib_id: str
    pins: List[KiPin] = field(default_factory=list)


@dataclass
class KiComponent:
    """A placed component instance on the schematic."""
    lib_id: str
    reference: str
    value: str
    at_x: float
    at_y: float
    rotation: float = 0.0
    mirror: bool = False
    unit: int = 1
    pin_numbers: List[str] = field(default_factory=list)
    pin_uuids: List[str] = field(default_factory=list)
    footprint: str = ""
    description: str = ""
    on_board: bool = True
    uuid: str = ""


@dataclass
class KiWire:
    """A wire segment connecting two points."""
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class KiLabel:
    """A net label (local or global) at a position."""
    name: str
    x: float
    y: float
    is_global: bool = False


@dataclass
class KiJunction:
    """A junction point where wires connect."""
    x: float
    y: float


@dataclass
class KiSchematic:
    """Complete parsed schematic."""
    title: str = ""
    company: str = ""
    lib_symbols: Dict[str, KiLibSymbol] = field(default_factory=dict)
    components: List[KiComponent] = field(default_factory=list)
    wires: List[KiWire] = field(default_factory=list)
    labels: List[KiLabel] = field(default_factory=list)
    junctions: List[KiJunction] = field(default_factory=list)
    nets: Dict[str, List[str]] = field(default_factory=dict)  # net_name -> [ref.pin, ...]


# ------------------------------------------------------------------ #
# S-expression tokenizer (regex-based for speed)
# ------------------------------------------------------------------ #

_TOKEN_RE = re.compile(
    r'"(?:[^"\\]|\\.)*"'   # quoted string
    r'|[()]'               # parens
    r"|[^\s()\"]+",        # unquoted atom
    re.DOTALL,
)


def _tokenize(text: str) -> List[str]:
    """Tokenize an S-expression string into a flat list of tokens."""
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group()
        if tok.startswith('"') and tok.endswith('"'):
            tokens.append(tok[1:-1])
        else:
            tokens.append(tok)
    return tokens


def _parse_sexpr(tokens: List[str], pos: int = 0) -> Tuple[Any, int]:
    """Parse tokens into nested lists (iterative to avoid stack overflow)."""
    if tokens[pos] != '(':
        return tokens[pos], pos + 1

    stack = []
    current: list = []
    pos += 1  # skip opening '('

    while pos < len(tokens):
        tok = tokens[pos]
        if tok == '(':
            stack.append(current)
            current = []
            pos += 1
        elif tok == ')':
            if stack:
                parent = stack.pop()
                parent.append(current)
                current = parent
            else:
                return current, pos + 1
            pos += 1
        else:
            current.append(tok)
            pos += 1

    return current, pos


def parse_sexpr(text: str) -> Any:
    """Parse an S-expression string into nested Python lists."""
    tokens = _tokenize(text)
    if not tokens:
        return []
    tree, _ = _parse_sexpr(tokens, 0)
    return tree


# ------------------------------------------------------------------ #
# Helpers to navigate the tree
# ------------------------------------------------------------------ #

def _find_child(node: list, tag: str) -> Optional[list]:
    """Find the first child list starting with the given tag."""
    if not isinstance(node, list):
        return None
    for child in node:
        if isinstance(child, list) and len(child) > 0 and child[0] == tag:
            return child
    return None


def _find_all(node: list, tag: str) -> List[list]:
    """Find all child lists starting with the given tag."""
    results = []
    if not isinstance(node, list):
        return results
    for child in node:
        if isinstance(child, list) and len(child) > 0 and child[0] == tag:
            results.append(child)
    return results


def _get_value(node: list, tag: str, default=None):
    """Get the value (2nd element) of a child with the given tag."""
    child = _find_child(node, tag)
    if child and len(child) > 1:
        return child[1]
    return default


def _get_property(node: list, prop_name: str, default="") -> str:
    """Get a property value from a node's property children."""
    for child in node:
        if isinstance(child, list) and len(child) >= 3 and child[0] == 'property':
            if child[1] == prop_name:
                return child[2]
    return default


# ------------------------------------------------------------------ #
# Main parser
# ------------------------------------------------------------------ #

def parse_kicad_sch(text: str) -> KiSchematic:
    """Parse a .kicad_sch file and return a KiSchematic."""
    tree = parse_sexpr(text)
    sch = KiSchematic()

    if not isinstance(tree, list) or tree[0] != 'kicad_sch':
        raise ValueError("Not a valid kicad_sch file")

    # Title block
    tb = _find_child(tree, 'title_block')
    if tb:
        sch.title = _get_value(tb, 'title', '')
        sch.company = _get_value(tb, 'company', '')

    # Library symbols
    lib_syms = _find_child(tree, 'lib_symbols')
    if lib_syms:
        for sym_node in _find_all(lib_syms, 'symbol'):
            lib_id = sym_node[1] if len(sym_node) > 1 else ''
            lib_sym = KiLibSymbol(lib_id=lib_id)

            # Pins are in sub-symbols (e.g. "Symbol_1_1")
            for sub in _find_all(sym_node, 'symbol'):
                for pin_node in _find_all(sub, 'pin'):
                    pin = _parse_lib_pin(pin_node)
                    if pin:
                        lib_sym.pins.append(pin)

            sch.lib_symbols[lib_id] = lib_sym

    # Component instances (top-level symbol nodes)
    for sym_node in _find_all(tree, 'symbol'):
        comp = _parse_component(sym_node)
        if comp and comp.on_board:
            sch.components.append(comp)

    # Wires
    for wire_node in _find_all(tree, 'wire'):
        wire = _parse_wire(wire_node)
        if wire:
            sch.wires.append(wire)

    # Labels (local)
    for label_node in _find_all(tree, 'label'):
        label = _parse_label(label_node, is_global=False)
        if label:
            sch.labels.append(label)

    # Global labels
    for gl_node in _find_all(tree, 'global_label'):
        label = _parse_label(gl_node, is_global=True)
        if label:
            sch.labels.append(label)

    # Junctions
    for junc_node in _find_all(tree, 'junction'):
        junc = _parse_junction(junc_node)
        if junc:
            sch.junctions.append(junc)

    # Build connectivity (net name -> component pins)
    _build_netlist(sch)

    return sch


# ------------------------------------------------------------------ #
# Node parsers
# ------------------------------------------------------------------ #

def _parse_lib_pin(pin_node: list) -> Optional[KiPin]:
    """Parse a pin node from a library symbol."""
    # pin_node: ['pin', type, graphic, ['at', x, y, angle], ['length', l],
    #            ['name', name, ...], ['number', num, ...]]
    try:
        at_node = _find_child(pin_node, 'at')
        name_node = _find_child(pin_node, 'name')
        num_node = _find_child(pin_node, 'number')

        x = float(at_node[1]) if at_node and len(at_node) > 1 else 0.0
        y = float(at_node[2]) if at_node and len(at_node) > 2 else 0.0
        name = name_node[1] if name_node and len(name_node) > 1 else ''
        number = num_node[1] if num_node and len(num_node) > 1 else ''
        length = 2.54
        len_node = _find_child(pin_node, 'length')
        if len_node and len(len_node) > 1:
            length = float(len_node[1])

        return KiPin(number=number, name=name, x=x, y=y, length=length)
    except (ValueError, IndexError):
        return None


def _parse_component(sym_node: list) -> Optional[KiComponent]:
    """Parse a placed component instance."""
    try:
        lib_id_val = _get_value(sym_node, 'lib_id', '')
        if not lib_id_val:
            return None

        at_node = _find_child(sym_node, 'at')
        at_x = float(at_node[1]) if at_node and len(at_node) > 1 else 0.0
        at_y = float(at_node[2]) if at_node and len(at_node) > 2 else 0.0
        rotation = float(at_node[3]) if at_node and len(at_node) > 3 else 0.0

        mirror = any(
            isinstance(c, list) and c[0] == 'mirror'
            for c in sym_node
        )
        unit = int(_get_value(sym_node, 'unit', '1'))
        on_board_val = _get_value(sym_node, 'on_board', 'yes')
        on_board = on_board_val == 'yes'

        reference = _get_property(sym_node, 'Reference', '')
        value = _get_property(sym_node, 'Value', '')
        footprint = _get_property(sym_node, 'Footprint', '')
        description = _get_property(sym_node, 'Description', '')
        uuid_val = _get_value(sym_node, 'uuid', '')

        # Pin assignments
        pin_numbers = []
        pin_uuids = []
        for pin_node in _find_all(sym_node, 'pin'):
            if len(pin_node) >= 2:
                pin_numbers.append(pin_node[1])
            if len(pin_node) >= 3:
                uuid_node = _find_child(pin_node, 'uuid')
                if uuid_node and len(uuid_node) > 1:
                    pin_uuids.append(uuid_node[1])
                elif isinstance(pin_node[2], list) and pin_node[2][0] == 'uuid':
                    pin_uuids.append(pin_node[2][1])

        comp = KiComponent(
            lib_id=lib_id_val,
            reference=reference,
            value=value,
            at_x=at_x,
            at_y=at_y,
            rotation=rotation,
            mirror=mirror,
            unit=unit,
            pin_numbers=pin_numbers,
            pin_uuids=pin_uuids,
            footprint=footprint,
            description=description,
            on_board=on_board,
            uuid=uuid_val,
        )
        return comp
    except (ValueError, IndexError):
        return None


def _parse_wire(wire_node: list) -> Optional[KiWire]:
    """Parse a wire segment."""
    try:
        pts_node = _find_child(wire_node, 'pts')
        if not pts_node:
            return None

        xy_nodes = _find_all(pts_node, 'xy')
        if len(xy_nodes) < 2:
            return None

        return KiWire(
            x1=float(xy_nodes[0][1]),
            y1=float(xy_nodes[0][2]),
            x2=float(xy_nodes[1][1]),
            y2=float(xy_nodes[1][2]),
        )
    except (ValueError, IndexError):
        return None


def _parse_label(label_node: list, is_global: bool) -> Optional[KiLabel]:
    """Parse a local or global label."""
    try:
        name = label_node[1] if len(label_node) > 1 else ''
        at_node = _find_child(label_node, 'at')
        x = float(at_node[1]) if at_node and len(at_node) > 1 else 0.0
        y = float(at_node[2]) if at_node and len(at_node) > 2 else 0.0

        return KiLabel(name=name, x=x, y=y, is_global=is_global)
    except (ValueError, IndexError):
        return None


def _parse_junction(junc_node: list) -> Optional[KiJunction]:
    """Parse a junction point."""
    try:
        at_node = _find_child(junc_node, 'at')
        x = float(at_node[1]) if at_node and len(at_node) > 1 else 0.0
        y = float(at_node[2]) if at_node and len(at_node) > 2 else 0.0
        return KiJunction(x=x, y=y)
    except (ValueError, IndexError):
        return None


# ------------------------------------------------------------------ #
# Netlist builder
# ------------------------------------------------------------------ #

def _build_netlist(sch: KiSchematic) -> None:
    """
    Build net connectivity from wires, labels, and component positions.
    Groups labels that share the same net name.
    """
    # Collect unique net names from labels
    net_names = set()
    for label in sch.labels:
        if label.name and not label.name.startswith('#'):
            net_names.add(label.name)

    # For each net name, record which label positions connect
    for name in sorted(net_names):
        sch.nets[name] = []

    # Simple approach: assign labels to nets by name
    # A label at position (x,y) connects to whatever wire/pin is at that point
    # For FormaCore, the important info is: which components connect via which nets
    # This requires computing pin world positions from component at + lib symbol pin offsets

    # Build component lookup by UUID for pin matching
    _assign_nets_from_labels(sch)


def _assign_nets_from_labels(sch: KiSchematic) -> None:
    """
    Use wire/label connectivity to figure out which component pins
    belong to which net. This is a simplified approach that uses
    wire-label-pin coordinate matching.
    """
    import math

    # Build a point → net_name mapping from labels
    label_points: Dict[Tuple[float, float], str] = {}
    for label in sch.labels:
        key = (round(label.x, 2), round(label.y, 2))
        if label.name and not label.name.startswith('#'):
            label_points[key] = label.name

    # Build wire graph: point → connected points
    wire_graph: Dict[Tuple[float, float], set] = {}
    for wire in sch.wires:
        p1 = (round(wire.x1, 2), round(wire.y1, 2))
        p2 = (round(wire.x2, 2), round(wire.y2, 2))
        wire_graph.setdefault(p1, set()).add(p2)
        wire_graph.setdefault(p2, set()).add(p1)

    # Add junction points as connectors
    for junc in sch.junctions:
        jp = (round(junc.x, 2), round(junc.y, 2))
        wire_graph.setdefault(jp, set())

    # Flood-fill to find connected regions with net names
    visited = set()
    point_to_net: Dict[Tuple[float, float], str] = {}

    def flood_fill(start, net_name):
        stack = [start]
        region = set()
        found_name = net_name
        while stack:
            pt = stack.pop()
            if pt in visited:
                continue
            visited.add(pt)
            region.add(pt)
            if pt in label_points and not found_name:
                found_name = label_points[pt]
            elif pt in label_points and found_name:
                pass  # already have a name
            for neighbor in wire_graph.get(pt, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        if found_name:
            for pt in region:
                point_to_net[pt] = found_name
        return found_name, region

    # Start from labeled points first
    for lp, name in label_points.items():
        if lp not in visited:
            flood_fill(lp, name)

    # Then fill unnamed regions
    for pt in list(wire_graph.keys()):
        if pt not in visited:
            flood_fill(pt, None)

    # Compute component pin world positions and match to nets
    for comp in sch.components:
        lib_sym = sch.lib_symbols.get(comp.lib_id)
        if not lib_sym:
            continue

        for pin in lib_sym.pins:
            px, py = _transform_pin(
                pin.x, pin.y, pin.length,
                comp.at_x, comp.at_y, comp.rotation, comp.mirror,
            )
            pin_pt = (round(px, 2), round(py, 2))
            net_name = point_to_net.get(pin_pt)
            if net_name and net_name in sch.nets:
                sch.nets[net_name].append(f"{comp.reference}.{pin.number}")


def _transform_pin(pin_x, pin_y, pin_length,
                    comp_x, comp_y, rotation, mirror) -> Tuple[float, float]:
    """Transform a library pin position to world coordinates."""
    import math
    # KiCad pin position is at the connection end
    # Library pins are defined relative to the symbol origin
    x = pin_x
    y = pin_y

    if mirror:
        x = -x

    # Rotate
    rad = math.radians(rotation)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    rx = x * cos_r - y * sin_r
    ry = x * sin_r + y * cos_r

    # Translate to world
    wx = comp_x + rx
    wy = comp_y + ry
    return wx, wy
