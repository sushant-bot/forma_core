"""
FormaCore AI - workflow/package.py
Submission Package Automation: one-click generation of a complete
design submission bundle (report + images + data + timeline).
Returns a ZIP file as bytes.
"""
from __future__ import annotations

import io
import csv
import json
import zipfile
from typing import List, Dict, Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.grid import Grid, Net
from router.multi_router import RoutingResult
from visualize.plot import BoardVisualizer
from workflow.copilot import DesignInsight, analyze_board
from workflow.report import generate_report
from workflow.memory import ProjectMemory


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def prepare_submission(
    grid: Grid,
    nets: List[Net],
    naive_result: RoutingResult,
    ga_result: RoutingResult,
    naive_grid: Grid,
    ga_grid: Grid,
    gen_log: List[Dict[str, Any]],
    best_genome: Any,
    naive_time: float,
    ga_time: float,
    insights: List[DesignInsight],
    scores: Dict[str, Any],
    memory: Optional[ProjectMemory] = None,
) -> bytes:
    """
    Generate a complete submission ZIP package.

    Contents:
        report.md           - Full Markdown design report
        report.txt          - Plain-text design report
        naive_routing.png   - Naive routing board image
        ga_routing.png      - GA-optimized routing board image
        traces.csv          - Per-net routing statistics
        metrics.json        - Machine-readable metrics
        timeline.json       - Project memory log (if available)
        insights.json       - Design insights

    Returns:
        ZIP file contents as bytes.
    """
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # ----- Reports ----- #
        md_report = generate_report(
            grid, nets, naive_result, ga_result, gen_log,
            best_genome, naive_time, ga_time, insights, scores,
            fmt='markdown',
        )
        zf.writestr('report.md', md_report)

        txt_report = generate_report(
            grid, nets, naive_result, ga_result, gen_log,
            best_genome, naive_time, ga_time, insights, scores,
            fmt='text',
        )
        zf.writestr('report.txt', txt_report)

        # ----- Board images ----- #
        naive_png = _render_board_png(
            naive_grid, 'Naive Routing'
        )
        zf.writestr('naive_routing.png', naive_png)

        ga_png = _render_board_png(
            ga_grid, 'GA-Optimized Routing'
        )
        zf.writestr('ga_routing.png', ga_png)

        # ----- Traces CSV ----- #
        traces_csv = _build_traces_csv(ga_result)
        zf.writestr('traces.csv', traces_csv)

        # ----- Metrics JSON ----- #
        metrics = _build_metrics_json(
            grid, nets, naive_result, ga_result,
            gen_log, naive_time, ga_time, scores,
        )
        zf.writestr('metrics.json', metrics)

        # ----- Insights JSON ----- #
        insights_data = [
            {
                'severity': ins.severity,
                'category': ins.category,
                'title': ins.title,
                'message': ins.message,
                'suggestion': ins.suggestion,
            }
            for ins in insights
        ]
        zf.writestr(
            'insights.json',
            json.dumps(insights_data, indent=2),
        )

        # ----- Timeline ----- #
        if memory and memory.count > 0:
            zf.writestr('timeline.json', memory.export_json())

    return buf.getvalue()


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _render_board_png(grid: Grid, title: str) -> bytes:
    """Render a board state as a PNG image (bytes)."""
    viz = BoardVisualizer(grid)
    fig = viz.render(title=title, show_heat=True, headless=True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_traces_csv(result: RoutingResult) -> str:
    """Build a CSV with per-net routing statistics."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        'net_name', 'routed', 'trace_length', 'vias', 'bends',
        'start_x', 'start_y', 'start_layer',
        'end_x', 'end_y', 'end_layer',
    ])

    for net_name, path in result.paths.items():
        vias = sum(
            1 for i in range(1, len(path))
            if path[i][2] != path[i - 1][2]
        )
        bends = sum(
            1 for i in range(2, len(path))
            if (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            != (path[i - 1][0] - path[i - 2][0], path[i - 1][1] - path[i - 2][1])
        )
        start = path[0] if path else (0, 0, 0)
        end = path[-1] if path else (0, 0, 0)

        writer.writerow([
            net_name, 'yes', len(path), vias, bends,
            start[0], start[1], start[2],
            end[0], end[1], end[2],
        ])

    for name in result.failed_nets:
        writer.writerow([
            name, 'no', 0, 0, 0, '', '', '', '', '', '',
        ])

    return buf.getvalue()


def _build_metrics_json(
    grid, nets, naive_result, ga_result,
    gen_log, naive_time, ga_time, scores,
) -> str:
    """Build a machine-readable metrics JSON."""
    data = {
        'board': {
            'width': grid.width,
            'height': grid.height,
            'layers': grid.num_layers,
            'resolution_mm': grid.resolution,
            'components': len(grid.components),
            'nets': len(nets),
        },
        'naive': {
            'routed': len(naive_result.paths),
            'total': len(nets),
            'completion_pct': round(naive_result.completion_rate * 100, 1),
            'trace_length': naive_result.total_trace_length,
            'vias': naive_result.total_vias,
            'bends': naive_result.total_bends,
            'time_s': round(naive_time, 3),
        },
        'ga': {
            'routed': len(ga_result.paths),
            'total': len(nets),
            'completion_pct': round(ga_result.completion_rate * 100, 1),
            'trace_length': ga_result.total_trace_length,
            'vias': ga_result.total_vias,
            'bends': ga_result.total_bends,
            'time_s': round(ga_time, 3),
            'generations': len(gen_log),
        },
        'scores': scores,
    }
    return json.dumps(data, indent=2)
