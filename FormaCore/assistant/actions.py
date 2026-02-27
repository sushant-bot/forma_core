"""
FormaCore AI - assistant/actions.py
Strict action schema definitions. Every allowed action is registered
here with its required/optional fields and types.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

# ------------------------------------------------------------------ #
# ACTION REGISTRY
# ------------------------------------------------------------------ #

# Each action has:
#   "fields": dict of field_name -> {"type": ..., "required": bool, ...}
#   "description": human-readable description

ACTION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "move_component": {
        "description": "Move a component to a new grid position",
        "fields": {
            "ref": {"type": "str", "required": True,
                    "description": "Component reference (e.g. 'MCU')"},
            "x": {"type": "int", "required": True, "min": 0,
                   "description": "New X position (grid cells)"},
            "y": {"type": "int", "required": True, "min": 0,
                   "description": "New Y position (grid cells)"},
        },
    },
    "rotate_component": {
        "description": "Rotate a component (90-degree increments)",
        "fields": {
            "ref": {"type": "str", "required": True,
                    "description": "Component reference"},
            "rotation_deg": {"type": "int", "required": True,
                             "allowed": [0, 90, 180, 270],
                             "description": "Rotation in degrees"},
        },
    },
    "move_group": {
        "description": "Move multiple components by an offset",
        "fields": {
            "refs": {"type": "list_str", "required": True,
                     "description": "List of component references"},
            "dx": {"type": "int", "required": True,
                   "description": "X offset (grid cells)"},
            "dy": {"type": "int", "required": True,
                   "description": "Y offset (grid cells)"},
        },
    },
    "place_component": {
        "description": "Place a new component on the board",
        "fields": {
            "ref": {"type": "str", "required": True,
                    "description": "Component name/reference"},
            "x": {"type": "int", "required": True, "min": 0},
            "y": {"type": "int", "required": True, "min": 0},
            "width": {"type": "int", "required": False,
                      "default": 4, "min": 1, "max": 30},
            "height": {"type": "int", "required": False,
                       "default": 3, "min": 1, "max": 30},
            "layer": {"type": "int", "required": False,
                      "default": 0, "allowed": [0, 1]},
            "power_w": {"type": "float", "required": False,
                        "default": 0.0, "min": 0.0},
        },
    },
    "remove_component": {
        "description": "Remove a component from the board",
        "fields": {
            "ref": {"type": "str", "required": True,
                    "description": "Component reference to remove"},
        },
    },
    "route_all": {
        "description": "Route all nets with default or custom weights",
        "fields": {
            "weights": {"type": "dict", "required": False,
                        "description": "Optional CostWeights overrides"},
        },
    },
    "optimize_routing": {
        "description": "Run GA optimization to find best routing strategy",
        "fields": {
            "pop_size": {"type": "int", "required": False,
                         "default": 15, "min": 5, "max": 30},
            "generations": {"type": "int", "required": False,
                            "default": 20, "min": 5, "max": 50},
            "early_stop": {"type": "int", "required": False,
                           "default": 5, "min": 3, "max": 15},
            "heat_sigma": {"type": "float", "required": False,
                           "default": 10.0, "min": 1.0, "max": 30.0},
        },
    },
    "clear_routes": {
        "description": "Remove all routed traces",
        "fields": {},
    },
    "clear_net": {
        "description": "Remove a specific routed net",
        "fields": {
            "net_name": {"type": "str", "required": True,
                         "description": "Name of net to clear"},
        },
    },
    "apply_heat": {
        "description": "Apply thermal model to the board",
        "fields": {
            "sigma": {"type": "float", "required": False,
                      "default": 10.0, "min": 1.0, "max": 30.0},
        },
    },
    "board_info": {
        "description": "Get current board state summary",
        "fields": {},
    },
}


def get_action_list() -> List[Dict[str, str]]:
    """Return list of available actions with descriptions."""
    return [
        {"id": aid, "description": schema["description"]}
        for aid, schema in ACTION_REGISTRY.items()
    ]


def get_action_schema(action_id: str) -> Optional[Dict[str, Any]]:
    """Return the schema for a specific action."""
    return ACTION_REGISTRY.get(action_id)


# ------------------------------------------------------------------ #
# UI-FRIENDLY ACTION LABELS
# ------------------------------------------------------------------ #

ACTION_LABELS = {
    "move_component": "Move Component",
    "rotate_component": "Rotate Component",
    "move_group": "Move Group",
    "place_component": "Place Component",
    "remove_component": "Remove Component",
    "route_all": "Route All Nets",
    "optimize_routing": "Optimize Routing (GA)",
    "clear_routes": "Clear All Routes",
    "clear_net": "Clear Specific Net",
    "apply_heat": "Apply Heat Model",
    "board_info": "Board Info",
}


def get_ui_actions() -> List[Tuple[str, str]]:
    """Return (action_id, label) pairs for UI dropdowns."""
    return [(aid, label) for aid, label in ACTION_LABELS.items()]
