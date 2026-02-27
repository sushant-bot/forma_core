"""
FormaCore AI - executor/sandbox.py
Runs actions on a board copy. Returns preview + DRC report.
Never modifies the original board.
"""
from __future__ import annotations

import io
from copy import deepcopy
from typing import Dict, Any, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.grid import Grid, Net
from executor import executor
from visualize.plot import BoardVisualizer
from router.multi_router import RoutingResult


class SandboxResult:
    """Result of running an action in sandbox."""

    def __init__(self, action_result: Dict[str, Any],
                 drc_report: Dict[str, Any],
                 preview_grid: Grid,
                 routing_result: Optional[RoutingResult] = None):
        self.action_result = action_result
        self.drc_report = drc_report
        self.preview_grid = preview_grid
        self.routing_result = routing_result

    @property
    def ok(self) -> bool:
        return self.action_result.get("status") == "ok"

    @property
    def drc_passed(self) -> bool:
        return self.drc_report.get("passed", False)


def run_in_sandbox(grid: Grid, nets: List[Net],
                   action_id: str, payload: Dict[str, Any]
                   ) -> SandboxResult:
    """
    Execute an action on a copy of the board.

    1. Deep-copy the grid
    2. Execute the action on the copy
    3. Run DRC
    4. Return results (no side effects on original)
    """
    board_copy = grid.clone()
    # Components are shallow-copied in clone(); deep-copy them
    board_copy.components = [deepcopy(c) for c in grid.components]
    # Preserve existing routed paths (clone() creates empty dict)
    board_copy.routed_paths = {
        k: list(v) for k, v in grid.routed_paths.items()
    }

    action_result = _dispatch(board_copy, nets, action_id, payload)

    drc_report = executor.get_drc_report(board_copy, nets)

    routing_result = None
    if action_id in ("route_all", "optimize_routing"):
        # Build a RoutingResult from routed paths
        routing_result = _build_routing_result(board_copy, nets)

    return SandboxResult(
        action_result=action_result,
        drc_report=drc_report,
        preview_grid=board_copy,
        routing_result=routing_result,
    )


def generate_preview_image(grid: Grid,
                           result: Optional[RoutingResult] = None
                           ) -> bytes:
    """Render the board state as a PNG image (bytes)."""
    viz = BoardVisualizer(grid)
    fig = viz.render(
        title="Preview",
        show_heat=True,
        result=result,
        headless=True,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ------------------------------------------------------------------ #
# ACTION DISPATCH
# ------------------------------------------------------------------ #

def _dispatch(grid: Grid, nets: List[Net],
              action_id: str, payload: Dict[str, Any]
              ) -> Dict[str, Any]:
    """Route an action_id to the correct executor function."""

    if action_id == "move_component":
        return executor.move_component(
            grid,
            ref=payload["ref"],
            x=payload["x"],
            y=payload["y"],
        )

    elif action_id == "rotate_component":
        return executor.rotate_component(
            grid,
            ref=payload["ref"],
            rotation_deg=payload["rotation_deg"],
        )

    elif action_id == "move_group":
        return executor.move_group(
            grid,
            refs=payload["refs"],
            dx=payload["dx"],
            dy=payload["dy"],
        )

    elif action_id == "place_component":
        return executor.place_component(
            grid,
            ref=payload["ref"],
            x=payload["x"],
            y=payload["y"],
            width=payload.get("width", 4),
            height=payload.get("height", 3),
            layer=payload.get("layer", 0),
            power_w=payload.get("power_w", 0.0),
            pins=payload.get("pins"),
        )

    elif action_id == "remove_component":
        return executor.remove_component(grid, ref=payload["ref"])

    elif action_id == "route_all":
        from router.cost import CostWeights
        weights = None
        if payload.get("weights"):
            w = payload["weights"]
            weights = CostWeights(
                step=w.get("step", 1.0),
                bend=w.get("bend", 5.0),
                via=w.get("via", 20.0),
                heat=w.get("heat", 2.0),
                congestion=w.get("congestion", 3.0),
            )
        return executor.route_all_nets(
            grid, nets, weights=weights,
            net_order=payload.get("net_order"),
        )

    elif action_id == "optimize_routing":
        return _run_ga_optimization(grid, nets, payload)

    elif action_id == "clear_routes":
        return executor.clear_all_routes(grid)

    elif action_id == "clear_net":
        return executor.clear_net(grid, net_name=payload["net_name"])

    elif action_id == "apply_heat":
        return executor.apply_heat(
            grid, sigma=payload.get("sigma", 10.0))

    elif action_id == "board_info":
        return executor.get_board_info(grid)

    else:
        return {"status": "error", "msg": f"Unknown action: {action_id}"}


def _run_ga_optimization(grid: Grid, nets: List[Net],
                         payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run GA optimization inside sandbox."""
    from ai.fitness import FitnessEvaluator
    from ai.ga import GeneticAlgorithm, GAConfig

    executor.apply_heat(grid, sigma=payload.get("heat_sigma", 10.0))

    config = GAConfig(
        population_size=payload.get("pop_size", 15),
        generations=payload.get("generations", 20),
        early_stop_gens=payload.get("early_stop", 5),
    )

    evaluator = FitnessEvaluator(grid, nets)
    ga = GeneticAlgorithm(evaluator, config)
    best = ga.run()

    # Clear and re-route with best genome
    for name in list(grid.routed_paths.keys()):
        grid.clear_net(name)

    from router.multi_router import MultiNetRouter
    router = MultiNetRouter(grid, best.to_cost_weights())
    result = router.route_all(nets, net_order=best.net_order)

    return {
        "status": "ok",
        "routed": len(result.paths),
        "failed": result.failed_nets,
        "total": len(nets),
        "trace_length": result.total_trace_length,
        "vias": result.total_vias,
        "bends": result.total_bends,
        "fitness": round(best.fitness, 1),
        "genome": {
            "net_order": best.net_order,
            "distance_w": round(best.distance_w, 3),
            "bend_w": round(best.bend_w, 3),
            "via_w": round(best.via_w, 3),
            "heat_w": round(best.heat_w, 3),
            "congestion_w": round(best.congestion_w, 3),
        },
    }


def _build_routing_result(grid: Grid, nets: List[Net]
                          ) -> Optional[RoutingResult]:
    """Build a RoutingResult from currently routed paths on grid."""
    import numpy as np

    if not grid.routed_paths:
        return None

    paths = dict(grid.routed_paths)
    routed_names = set(paths.keys())
    failed = [n.name for n in nets if n.name not in routed_names]

    total_length = 0
    total_vias = 0
    total_bends = 0

    for name, path in paths.items():
        total_length += len(path) - 1
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
        np.sum(np.maximum(grid.congestion - 1, 0))
    )

    total_nets = len(nets)
    routed_count = total_nets - len(failed)

    return RoutingResult(
        paths=paths,
        failed_nets=failed,
        total_trace_length=total_length,
        total_vias=total_vias,
        total_bends=total_bends,
        congestion_score=congestion_score,
        completion_rate=routed_count / total_nets if total_nets > 0 else 1.0,
    )
