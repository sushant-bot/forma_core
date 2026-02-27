"""
FormaCore AI - workflow/copilot.py
Design Insight Copilot: rule-based heuristics that convert raw routing
metrics into actionable, human-readable feedback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np

from core.grid import Grid, Net, Component
from router.multi_router import RoutingResult


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class DesignInsight:
    severity: str           # 'critical', 'warning', 'info'
    category: str           # 'thermal', 'routing', 'placement', 'congestion', 'general'
    title: str
    message: str
    suggestion: str

    @property
    def icon(self) -> str:
        return {'critical': 'X', 'warning': '!', 'info': 'i'}[self.severity]


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def analyze_board(
    grid: Grid,
    nets: List[Net],
    naive_result: RoutingResult,
    ga_result: RoutingResult,
    gen_log: List[Dict[str, Any]] | None = None,
) -> List[DesignInsight]:
    """
    Run all heuristic analysers and return a sorted list of insights.
    """
    insights: List[DesignInsight] = []

    insights.extend(_analyze_completion(naive_result, ga_result, nets))
    insights.extend(_analyze_vias(grid, ga_result))
    insights.extend(_analyze_thermal(grid, ga_result))
    insights.extend(_analyze_congestion(grid))
    insights.extend(_analyze_net_lengths(ga_result))
    insights.extend(_analyze_layer_balance(grid, ga_result))
    insights.extend(_analyze_placement(grid))
    insights.extend(_analyze_bends(ga_result))
    insights.extend(_analyze_convergence(gen_log))

    severity_order = {'critical': 0, 'warning': 1, 'info': 2}
    return sorted(insights, key=lambda i: severity_order[i.severity])


# ------------------------------------------------------------------ #
# Individual analysers
# ------------------------------------------------------------------ #

def _analyze_completion(
    naive: RoutingResult, ga: RoutingResult, nets: List[Net]
) -> List[DesignInsight]:
    out: List[DesignInsight] = []
    total = len(nets)

    if ga.completion_rate < 1.0:
        failed = ga.failed_nets
        out.append(DesignInsight(
            severity='critical',
            category='routing',
            title='Incomplete routing',
            message=(
                f"{len(failed)}/{total} nets failed to route: "
                f"{', '.join(failed[:5])}{'...' if len(failed) > 5 else ''}."
            ),
            suggestion=(
                "Try increasing the grid size, adjusting component placement "
                "to reduce congestion, or increasing GA population/generations."
            ),
        ))
    elif naive.completion_rate < 1.0 and ga.completion_rate >= 1.0:
        improvement = (1.0 - naive.completion_rate) * 100
        out.append(DesignInsight(
            severity='info',
            category='routing',
            title='GA recovered failed routes',
            message=(
                f"Naive routing failed {len(naive.failed_nets)} nets. "
                f"GA optimization achieved 100% completion "
                f"(+{improvement:.0f}% recovery)."
            ),
            suggestion="GA optimization is effective for this board layout.",
        ))

    # Via reduction
    if naive.total_vias > 0:
        via_red = (1 - ga.total_vias / naive.total_vias) * 100
        if via_red > 30:
            out.append(DesignInsight(
                severity='info',
                category='routing',
                title=f'Via count reduced by {via_red:.0f}%',
                message=(
                    f"GA reduced vias from {naive.total_vias} to "
                    f"{ga.total_vias} ({via_red:.0f}% reduction)."
                ),
                suggestion="Via reduction lowers manufacturing cost and improves signal integrity.",
            ))

    # Trace length reduction
    if naive.total_trace_length > 0:
        len_red = (1 - ga.total_trace_length / naive.total_trace_length) * 100
        if len_red > 20:
            out.append(DesignInsight(
                severity='info',
                category='routing',
                title=f'Trace length reduced by {len_red:.0f}%',
                message=(
                    f"Total trace length dropped from "
                    f"{naive.total_trace_length} to "
                    f"{ga.total_trace_length} cells."
                ),
                suggestion="Shorter traces improve signal integrity and reduce resistive losses.",
            ))

    return out


def _analyze_vias(grid: Grid, result: RoutingResult) -> List[DesignInsight]:
    out: List[DesignInsight] = []
    if not result.paths:
        return out

    # Collect all via locations
    via_locations: List[Tuple[int, int]] = []
    for net_name, path in result.paths.items():
        for i in range(1, len(path)):
            if path[i][2] != path[i - 1][2]:
                via_locations.append((path[i][0], path[i][1]))

    if len(via_locations) < 2:
        return out

    # Check for via clustering: count vias in 10x10 regions
    if via_locations:
        vx = np.array([v[0] for v in via_locations])
        vy = np.array([v[1] for v in via_locations])

        region_size = 10
        for rx in range(0, grid.width, region_size):
            for ry in range(0, grid.height, region_size):
                mask = (
                    (vx >= rx) & (vx < rx + region_size) &
                    (vy >= ry) & (vy < ry + region_size)
                )
                count = int(mask.sum())
                if count >= 4:
                    out.append(DesignInsight(
                        severity='warning',
                        category='routing',
                        title=f'Via cluster at region ({rx},{ry})',
                        message=(
                            f"{count} vias clustered in a {region_size}x"
                            f"{region_size} region near ({rx},{ry}). "
                            f"Dense via clusters increase crosstalk risk."
                        ),
                        suggestion=(
                            "Consider rotating nearby components to "
                            "reduce layer switching in this area."
                        ),
                    ))

    return out


def _analyze_thermal(grid: Grid, result: RoutingResult) -> List[DesignInsight]:
    out: List[DesignInsight] = []
    if not result.paths:
        return out

    heat_threshold = np.max(grid.heat) * 0.7 if np.max(grid.heat) > 0 else 0

    if heat_threshold == 0:
        return out

    hot_nets: List[str] = []
    for net_name, path in result.paths.items():
        heat_along_trace = [
            grid.heat[0, node[1], node[0]]
            if node[2] == 0
            else grid.heat[1, node[1], node[0]]
            for node in path
            if 0 <= node[0] < grid.width and 0 <= node[1] < grid.height
        ]
        if heat_along_trace and max(heat_along_trace) > heat_threshold:
            hot_nets.append(net_name)

    if hot_nets:
        out.append(DesignInsight(
            severity='warning',
            category='thermal',
            title=f'{len(hot_nets)} nets routed through thermal hotspots',
            message=(
                f"Nets passing through high-heat zones: "
                f"{', '.join(hot_nets[:5])}{'...' if len(hot_nets) > 5 else ''}. "
                f"Thermal stress can degrade trace reliability."
            ),
            suggestion=(
                "Increase heat penalty weight, move heat-generating "
                "components apart, or add thermal relief copper pours."
            ),
        ))

    # Check for specific hot components
    for comp in grid.components:
        if comp.power_w >= 2.0:
            out.append(DesignInsight(
                severity='warning',
                category='thermal',
                title=f'{comp.name} is a thermal hotspot ({comp.power_w}W)',
                message=(
                    f"Component {comp.name} dissipates {comp.power_w}W. "
                    f"Position: ({comp.x}, {comp.y})."
                ),
                suggestion=(
                    f"Consider increasing copper pour near {comp.name} "
                    f"or adding thermal vias underneath."
                ),
            ))

    return out


def _analyze_congestion(grid: Grid) -> List[DesignInsight]:
    out: List[DesignInsight] = []

    cong = grid.congestion
    if np.max(cong) == 0:
        return out

    # Find congestion hotspots (cells with value > 3x mean)
    mean_cong = np.mean(cong[cong > 0]) if np.any(cong > 0) else 0
    if mean_cong == 0:
        return out

    hotspot_threshold = max(mean_cong * 3, 3)
    hotspot_count = int(np.sum(cong > hotspot_threshold))

    if hotspot_count > 0:
        # Find the densest region
        max_loc = np.unravel_index(np.argmax(cong), cong.shape)
        max_val = int(cong[max_loc])

        out.append(DesignInsight(
            severity='warning' if hotspot_count > 10 else 'info',
            category='congestion',
            title=f'{hotspot_count} congestion hotspot cells detected',
            message=(
                f"Peak congestion: {max_val} traces sharing a cell "
                f"near ({max_loc[2]}, {max_loc[1]}). "
                f"{hotspot_count} cells exceed {hotspot_threshold:.0f}x "
                f"average congestion."
            ),
            suggestion=(
                "Spread components apart or increase board area "
                "to reduce routing density."
            ),
        ))

    return out


def _analyze_net_lengths(result: RoutingResult) -> List[DesignInsight]:
    out: List[DesignInsight] = []
    if not result.paths:
        return out

    lengths = {name: len(path) for name, path in result.paths.items()}
    if len(lengths) < 2:
        return out

    vals = list(lengths.values())
    mean_len = np.mean(vals)
    std_len = np.std(vals)

    if std_len == 0:
        return out

    outliers = [
        (name, l) for name, l in lengths.items()
        if l > mean_len + 2 * std_len
    ]

    for name, length in outliers:
        ratio = length / mean_len
        out.append(DesignInsight(
            severity='warning',
            category='routing',
            title=f'Net "{name}" is abnormally long',
            message=(
                f'Net "{name}" has {length} cells '
                f"({ratio:.1f}x average). "
                f"Long traces increase resistance and signal delay."
            ),
            suggestion=(
                f"Consider moving endpoint components of \"{name}\" "
                f"closer together, or review the via path."
            ),
        ))

    return out


def _analyze_layer_balance(
    grid: Grid, result: RoutingResult
) -> List[DesignInsight]:
    out: List[DesignInsight] = []
    if not result.paths:
        return out

    layer_counts = [0, 0]
    for path in result.paths.values():
        for node in path:
            if node[2] < 2:
                layer_counts[node[2]] += 1

    total = sum(layer_counts)
    if total == 0:
        return out

    ratio = max(layer_counts) / total
    if ratio > 0.85:
        dominant = 'TOP' if layer_counts[0] > layer_counts[1] else 'BOTTOM'
        out.append(DesignInsight(
            severity='info',
            category='routing',
            title=f'Layer imbalance: {ratio:.0%} on {dominant}',
            message=(
                f"{ratio:.0%} of routing is on the {dominant} layer. "
                f"Layer 0: {layer_counts[0]} cells, "
                f"Layer 1: {layer_counts[1]} cells."
            ),
            suggestion=(
                "Consider placing some components on the bottom layer "
                "to better utilise both routing layers."
            ),
        ))

    return out


def _analyze_placement(grid: Grid) -> List[DesignInsight]:
    out: List[DesignInsight] = []

    for comp in grid.components:
        # Check if near board edge (within 2 cells)
        margin = 2
        near_edge = (
            comp.x < margin or comp.y < margin or
            comp.x + comp.width > grid.width - margin or
            comp.y + comp.height > grid.height - margin
        )
        if near_edge:
            out.append(DesignInsight(
                severity='info',
                category='placement',
                title=f'{comp.name} is near board edge',
                message=(
                    f"Component {comp.name} at ({comp.x},{comp.y}) "
                    f"is within {margin} cells of the board boundary."
                ),
                suggestion=(
                    "Edge-mounted components can limit routing "
                    "flexibility. Move inward if possible."
                ),
            ))

    # Check inter-component spacing
    comps = grid.components
    for i, a in enumerate(comps):
        for b in comps[i + 1:]:
            gap_x = max(0, max(a.x, b.x) - min(a.x + a.width, b.x + b.width))
            gap_y = max(0, max(a.y, b.y) - min(a.y + a.height, b.y + b.height))
            gap = max(gap_x, gap_y) if gap_x > 0 or gap_y > 0 else 0
            if gap <= 1 and gap_x == 0 and gap_y == 0:
                # They might overlap or be adjacent
                pass  # DRC handles overlaps
            elif 0 < gap <= 2:
                out.append(DesignInsight(
                    severity='info',
                    category='placement',
                    title=f'{a.name} and {b.name} are closely spaced',
                    message=(
                        f"Only {gap} cell(s) gap between "
                        f"{a.name} and {b.name}. "
                        f"Tight spacing limits routing channels."
                    ),
                    suggestion=(
                        "Increase spacing to allow more routing "
                        "channels between these components."
                    ),
                ))

    return out


def _analyze_bends(result: RoutingResult) -> List[DesignInsight]:
    out: List[DesignInsight] = []
    if not result.paths:
        return out

    for net_name, path in result.paths.items():
        if len(path) < 3:
            continue

        bends = 0
        for i in range(2, len(path)):
            dx1 = path[i - 1][0] - path[i - 2][0]
            dy1 = path[i - 1][1] - path[i - 2][1]
            dx2 = path[i][0] - path[i - 1][0]
            dy2 = path[i][1] - path[i - 1][1]
            if (dx1, dy1) != (dx2, dy2):
                bends += 1

        length = len(path)
        if length > 0 and bends / length > 0.4:
            out.append(DesignInsight(
                severity='warning',
                category='routing',
                title=f'Excessive bends in net "{net_name}"',
                message=(
                    f'Net "{net_name}" has {bends} bends over '
                    f"{length} cells ({bends / length:.0%} bend ratio). "
                    f"Excessive bends increase impedance and EMI."
                ),
                suggestion=(
                    "Increase the bend penalty weight in GA parameters "
                    "or adjust component placement for straighter paths."
                ),
            ))

    return out


def _analyze_convergence(
    gen_log: List[Dict[str, Any]] | None,
) -> List[DesignInsight]:
    out: List[DesignInsight] = []

    if not gen_log or len(gen_log) < 3:
        return out

    best_scores = [g['best_fitness'] for g in gen_log]
    total_gens = len(best_scores)

    # Check if early stopping kicked in
    max_possible = gen_log[-1].get('total_gens', total_gens)
    if total_gens < max_possible * 0.3:
        out.append(DesignInsight(
            severity='info',
            category='general',
            title='GA converged quickly',
            message=(
                f"Converged in {total_gens}/{max_possible} generations "
                f"({total_gens / max_possible:.0%}). "
                f"The solution space may be simple."
            ),
            suggestion=(
                "Board complexity is low. Current GA settings are "
                "sufficient."
            ),
        ))
    elif total_gens >= max_possible * 0.95:
        # Check if still improving at the end
        late_improvement = abs(best_scores[-1] - best_scores[-max(5, total_gens // 10)])
        if late_improvement > best_scores[-1] * 0.02:
            out.append(DesignInsight(
                severity='warning',
                category='general',
                title='GA may not have fully converged',
                message=(
                    f"Optimization used all {total_gens} generations "
                    f"and was still improving at the end "
                    f"(late delta: {late_improvement:.0f})."
                ),
                suggestion=(
                    "Try increasing the generation count or "
                    "population size for better results."
                ),
            ))

    return out
