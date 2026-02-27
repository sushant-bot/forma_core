"""
FormaCore AI - workflow/report.py
Auto Report Generator: produces structured design reports in
Markdown and plain-text formats from routing results and analysis.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Optional

from core.grid import Grid, Net
from router.multi_router import RoutingResult
from workflow.copilot import DesignInsight


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def generate_report(
    grid: Grid,
    nets: List[Net],
    naive_result: RoutingResult,
    ga_result: RoutingResult,
    gen_log: List[Dict[str, Any]],
    best_genome: Any,
    naive_time: float,
    ga_time: float,
    insights: List[DesignInsight],
    scores: Dict[str, Any],
    fmt: str = 'markdown',
) -> str:
    """
    Generate a complete design report.

    Args:
        fmt: 'markdown' or 'text'

    Returns:
        The formatted report string.
    """
    ctx = _build_context(
        grid, nets, naive_result, ga_result, gen_log,
        best_genome, naive_time, ga_time, insights, scores,
    )
    if fmt == 'markdown':
        return _render_markdown(ctx)
    return _render_text(ctx)


# ------------------------------------------------------------------ #
# Build context dict
# ------------------------------------------------------------------ #

def _build_context(
    grid, nets, naive_result, ga_result, gen_log,
    best_genome, naive_time, ga_time, insights, scores,
) -> Dict[str, Any]:
    total = len(nets)
    n_naive = len(naive_result.paths)
    n_opt = len(ga_result.paths)

    via_red = (
        (1 - ga_result.total_vias / naive_result.total_vias) * 100
        if naive_result.total_vias > 0 else 0
    )
    len_red = (
        (1 - ga_result.total_trace_length / naive_result.total_trace_length) * 100
        if naive_result.total_trace_length > 0 else 0
    )

    weights = None
    if best_genome and hasattr(best_genome, 'weights'):
        w = best_genome.weights
        weights = {
            'step': w.step, 'bend': w.bend, 'via': w.via,
            'heat': w.heat, 'congestion': w.congestion,
        }

    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'board': {
            'width': grid.width,
            'height': grid.height,
            'width_mm': grid.width * grid.resolution,
            'height_mm': grid.height * grid.resolution,
            'layers': grid.num_layers,
            'resolution_mm': grid.resolution,
            'components': len(grid.components),
            'component_names': [c.name for c in grid.components],
            'nets': total,
        },
        'naive': {
            'routed': n_naive,
            'total': total,
            'completion': naive_result.completion_rate * 100,
            'vias': naive_result.total_vias,
            'trace_length': naive_result.total_trace_length,
            'bends': naive_result.total_bends,
            'time_s': naive_time,
        },
        'ga': {
            'routed': n_opt,
            'total': total,
            'completion': ga_result.completion_rate * 100,
            'vias': ga_result.total_vias,
            'trace_length': ga_result.total_trace_length,
            'bends': ga_result.total_bends,
            'time_s': ga_time,
            'generations': len(gen_log),
            'via_reduction': via_red,
            'length_reduction': len_red,
        },
        'weights': weights,
        'insights': insights,
        'scores': scores,
    }


# ------------------------------------------------------------------ #
# Markdown renderer
# ------------------------------------------------------------------ #

def _render_markdown(ctx: Dict[str, Any]) -> str:
    b = ctx['board']
    n = ctx['naive']
    g = ctx['ga']
    s = ctx['scores']
    lines: List[str] = []

    lines.append('# FormaCore AI - Design Report')
    lines.append(f'\n*Generated: {ctx["timestamp"]}*\n')

    # Board overview
    lines.append('## Board Overview\n')
    lines.append(f'| Property | Value |')
    lines.append(f'|---|---|')
    lines.append(f'| Grid Size | {b["width"]} x {b["height"]} cells |')
    lines.append(f'| Physical Size | {b["width_mm"]:.1f} x {b["height_mm"]:.1f} mm |')
    lines.append(f'| Layers | {b["layers"]} |')
    lines.append(f'| Resolution | {b["resolution_mm"]} mm/cell |')
    lines.append(f'| Components | {b["components"]} ({", ".join(b["component_names"])}) |')
    lines.append(f'| Nets | {b["nets"]} |')
    lines.append('')

    # Routing comparison
    lines.append('## Routing Comparison\n')
    lines.append(f'| Metric | Naive | GA Optimized | Improvement |')
    lines.append(f'|---|---|---|---|')
    lines.append(
        f'| Completion | {n["routed"]}/{n["total"]} '
        f'({n["completion"]:.0f}%) | '
        f'{g["routed"]}/{g["total"]} ({g["completion"]:.0f}%) | '
        f'{g["completion"] - n["completion"]:+.0f}% |'
    )
    lines.append(
        f'| Trace Length | {n["trace_length"]} | '
        f'{g["trace_length"]} | {g["length_reduction"]:+.1f}% |'
    )
    lines.append(
        f'| Vias | {n["vias"]} | '
        f'{g["vias"]} | {g["via_reduction"]:+.1f}% |'
    )
    lines.append(
        f'| Bends | {n["bends"]} | '
        f'{g["bends"]} | - |'
    )
    lines.append(
        f'| Runtime | {n["time_s"]:.2f}s | '
        f'{g["time_s"]:.1f}s ({g["generations"]} gens) | - |'
    )
    lines.append('')

    # GA convergence
    lines.append('## Optimization Summary\n')
    lines.append(f'- **Generations run:** {g["generations"]}')
    lines.append(f'- **GA runtime:** {g["time_s"]:.1f}s')
    if ctx['weights']:
        w = ctx['weights']
        lines.append(f'- **Optimized weights:** step={w["step"]:.1f}, '
                      f'bend={w["bend"]:.1f}, via={w["via"]:.1f}, '
                      f'heat={w["heat"]:.1f}, congestion={w["congestion"]:.1f}')
    lines.append('')

    # System scores
    lines.append('## System Score\n')
    overall = s.get('overall', 0)
    lines.append(f'**Overall: {overall:.0f}/100**\n')
    lines.append(f'| Category | Score | Weight |')
    lines.append(f'|---|---|---|')
    categories = [
        ('Completion', 'completion', '35%'),
        ('Via Efficiency', 'via_efficiency', '20%'),
        ('Trace Quality', 'trace_quality', '20%'),
        ('Bend Quality', 'bend_quality', '10%'),
        ('Speed', 'speed', '15%'),
    ]
    for label, key, weight in categories:
        val = s.get(key, 0)
        lines.append(f'| {label} | {val:.0f}/100 | {weight} |')
    lines.append('')

    # Design insights
    if ctx['insights']:
        lines.append('## Design Insights\n')
        for ins in ctx['insights']:
            icon = {'critical': 'CRITICAL', 'warning': 'WARNING', 'info': 'INFO'}
            lines.append(f'### [{icon[ins.severity]}] {ins.title}\n')
            lines.append(f'{ins.message}\n')
            lines.append(f'> **Suggestion:** {ins.suggestion}\n')

    # Thermal analysis
    lines.append('## Thermal Analysis\n')
    hot_comps = [c for c in ctx.get('_comps', []) if c.power_w > 1.0]
    thermal_insights = [i for i in ctx['insights'] if i.category == 'thermal']
    if thermal_insights:
        for ti in thermal_insights:
            lines.append(f'- {ti.message}')
    else:
        lines.append('No thermal concerns detected.')
    lines.append('')

    lines.append('---\n')
    lines.append('*Report generated by FormaCore AI*')

    return '\n'.join(lines)


# ------------------------------------------------------------------ #
# Plain-text renderer
# ------------------------------------------------------------------ #

def _render_text(ctx: Dict[str, Any]) -> str:
    b = ctx['board']
    n = ctx['naive']
    g = ctx['ga']
    s = ctx['scores']
    lines: List[str] = []

    lines.append('=' * 60)
    lines.append('  FORMACORE AI - DESIGN REPORT')
    lines.append('=' * 60)
    lines.append(f'  Generated: {ctx["timestamp"]}')
    lines.append('')

    lines.append('BOARD OVERVIEW')
    lines.append('-' * 40)
    lines.append(f'  Grid:        {b["width"]} x {b["height"]} cells')
    lines.append(f'  Physical:    {b["width_mm"]:.1f} x {b["height_mm"]:.1f} mm')
    lines.append(f'  Layers:      {b["layers"]}')
    lines.append(f'  Components:  {b["components"]} ({", ".join(b["component_names"])})')
    lines.append(f'  Nets:        {b["nets"]}')
    lines.append('')

    lines.append('ROUTING COMPARISON')
    lines.append('-' * 40)
    lines.append(f'  {"Metric":<18} {"Naive":>10} {"GA":>10} {"Improv.":>10}')
    lines.append(f'  {"Completion":<18} '
                  f'{n["routed"]}/{n["total"]:>7} '
                  f'{g["routed"]}/{g["total"]:>7} '
                  f'{g["completion"] - n["completion"]:>+9.0f}%')
    lines.append(f'  {"Trace Length":<18} '
                  f'{n["trace_length"]:>10} '
                  f'{g["trace_length"]:>10} '
                  f'{g["length_reduction"]:>+9.1f}%')
    lines.append(f'  {"Vias":<18} '
                  f'{n["vias"]:>10} '
                  f'{g["vias"]:>10} '
                  f'{g["via_reduction"]:>+9.1f}%')
    lines.append(f'  {"Bends":<18} '
                  f'{n["bends"]:>10} '
                  f'{g["bends"]:>10}')
    lines.append(f'  {"Runtime":<18} '
                  f'{n["time_s"]:>9.2f}s '
                  f'{g["time_s"]:>9.1f}s')
    lines.append('')

    lines.append('OPTIMIZATION')
    lines.append('-' * 40)
    lines.append(f'  Generations: {g["generations"]}')
    lines.append(f'  GA time:     {g["time_s"]:.1f}s')
    if ctx['weights']:
        w = ctx['weights']
        lines.append(f'  Weights:     step={w["step"]:.1f} bend={w["bend"]:.1f} '
                      f'via={w["via"]:.1f} heat={w["heat"]:.1f} '
                      f'cong={w["congestion"]:.1f}')
    lines.append('')

    lines.append('SYSTEM SCORE')
    lines.append('-' * 40)
    overall = s.get('overall', 0)
    lines.append(f'  Overall: {overall:.0f}/100')
    for label, key in [('Completion', 'completion'),
                        ('Via Efficiency', 'via_efficiency'),
                        ('Trace Quality', 'trace_quality'),
                        ('Bend Quality', 'bend_quality'),
                        ('Speed', 'speed')]:
        lines.append(f'  {label:<18} {s.get(key, 0):>6.0f}/100')
    lines.append('')

    if ctx['insights']:
        lines.append('DESIGN INSIGHTS')
        lines.append('-' * 40)
        for ins in ctx['insights']:
            tag = ins.severity.upper()
            lines.append(f'  [{tag}] {ins.title}')
            lines.append(f'    {ins.message}')
            lines.append(f'    -> {ins.suggestion}')
            lines.append('')

    lines.append('=' * 60)
    lines.append('  Generated by FormaCore AI')
    lines.append('=' * 60)

    return '\n'.join(lines)
