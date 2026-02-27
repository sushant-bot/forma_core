"""
FormaCore AI - ui/app.py
Enhanced Streamlit dashboard for PCB routing optimization.

Run: streamlit run ui/app.py
"""
from __future__ import annotations

import sys
import os
import time
import io
import math

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')  # must be before any pyplot import
import matplotlib.pyplot as plt

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.grid import Grid, Component, Net, Layer
from core.heat import HeatModel
from core.congestion import CongestionTracker
from router.cost import CostWeights
from router.multi_router import MultiNetRouter, RoutingResult
from ai.genome import Genome
from ai.fitness import FitnessEvaluator
from ai.ga import GeneticAlgorithm, GAConfig
from visualize.plot import BoardVisualizer
from controller.api import BoardController
from assistant.actions import get_ui_actions, ACTION_REGISTRY
from assistant.validator import validate_action
from workflow.copilot import analyze_board, DesignInsight
from workflow.report import generate_report
from workflow.memory import ProjectMemory
from workflow.package import prepare_submission


# ------------------------------------------------------------------ #
# PAGE CONFIG & THEME
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="FormaCore AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional dark-themed UI
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global text styling */
    .main .block-container {
        padding-top: 1rem;
        max-width: 1200px;
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .header-banner h1 {
        color: #e0e0ff;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .header-banner .subtitle {
        color: #9d9dcc;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 400;
    }
    .header-badge {
        display: inline-block;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-left: 8px;
        vertical-align: middle;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.15);
    }
    .metric-label {
        color: #8888aa;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        color: #e0e0ff;
        font-size: 1.6rem;
        font-weight: 700;
        font-family: 'Inter', monospace;
        line-height: 1.2;
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    .metric-delta.positive { color: #22c55e; }
    .metric-delta.negative { color: #ef4444; }
    .metric-delta.neutral { color: #8888aa; }

    /* Score ring */
    .score-container {
        text-align: center;
        padding: 1.5rem;
    }
    .score-ring {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        border: 6px solid;
        margin-bottom: 0.8rem;
    }
    .score-ring.excellent { border-color: #22c55e; background: rgba(34, 197, 94, 0.08); }
    .score-ring.good { border-color: #3b82f6; background: rgba(59, 130, 246, 0.08); }
    .score-ring.fair { border-color: #f59e0b; background: rgba(245, 158, 11, 0.08); }
    .score-ring.poor { border-color: #ef4444; background: rgba(239, 68, 68, 0.08); }
    .score-number {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        line-height: 1;
    }
    .score-label {
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 2px;
    }
    .score-number.excellent { color: #22c55e; }
    .score-number.good { color: #3b82f6; }
    .score-number.fair { color: #f59e0b; }
    .score-number.poor { color: #ef4444; }
    .score-label.excellent { color: #22c55e; }
    .score-label.good { color: #3b82f6; }
    .score-label.fair { color: #f59e0b; }
    .score-label.poor { color: #ef4444; }

    /* Sub-score bars */
    .subscore-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.6rem;
        gap: 0.8rem;
    }
    .subscore-label {
        color: #9d9dcc;
        font-size: 0.8rem;
        min-width: 100px;
        text-align: right;
        font-weight: 500;
    }
    .subscore-bar-bg {
        flex: 1;
        background: rgba(255,255,255,0.06);
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    .subscore-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    .subscore-value {
        color: #e0e0ff;
        font-size: 0.8rem;
        min-width: 35px;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        color: #c0c0e0;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 1rem;
        margin-top: 0.5rem;
    }

    /* Strategy explanation cards */
    .strategy-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #1e1e3a 100%);
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
    .strategy-card .weight-name {
        color: #a5b4fc;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .strategy-card .weight-desc {
        color: #9d9dcc;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }

    /* Table styling */
    .net-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(99, 102, 241, 0.15);
    }
    .net-table thead th {
        background: rgba(99, 102, 241, 0.12);
        color: #a5b4fc;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 0.6rem 0.8rem;
        text-align: left;
    }
    .net-table tbody td {
        padding: 0.5rem 0.8rem;
        font-size: 0.85rem;
        color: #d0d0e8;
        border-top: 1px solid rgba(255,255,255,0.04);
    }
    .net-table tbody tr:hover {
        background: rgba(99, 102, 241, 0.06);
    }
    .status-routed {
        color: #22c55e;
        font-weight: 600;
    }
    .status-failed {
        color: #ef4444;
        font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #c0c0e0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }

    /* Info box restyle */
    .custom-info {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        color: #9d9dcc;
    }
    .custom-info .icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .custom-info .message { font-size: 0.95rem; }

    /* Divider */
    .styled-divider {
        border: none;
        border-top: 1px solid rgba(99, 102, 241, 0.15);
        margin: 1.2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# SAMPLE BOARD
# ------------------------------------------------------------------ #

def create_sample_board(width: int = 80, height: int = 60) -> tuple:
    """Create sample 2-layer PCB with dense routing."""
    grid = Grid(width=width, height=height, layers=2, resolution_mm=0.5)

    components = [
        Component("MCU", x=25, y=20, width=12, height=12,
                  layer=Layer.TOP, power_w=0.8,
                  pins=[(0, 0), (11, 0), (0, 11), (11, 11),
                        (5, 0), (6, 0), (0, 5), (0, 6),
                        (11, 5), (11, 6), (5, 11), (6, 11)]),
        Component("REG", x=5, y=5, width=6, height=4,
                  layer=Layer.TOP, power_w=3.0,
                  pins=[(0, 0), (5, 0), (0, 3), (5, 3)]),
        Component("FLASH", x=55, y=8, width=8, height=6,
                  layer=Layer.TOP, power_w=0.3,
                  pins=[(0, 0), (7, 0), (0, 5), (7, 5)]),
        Component("CONN", x=58, y=42, width=10, height=8,
                  layer=Layer.BOTTOM, power_w=0.0,
                  pins=[(0, 0), (9, 0), (0, 7), (9, 7),
                        (3, 0), (6, 0)]),
        Component("C1", x=20, y=12, width=3, height=2,
                  layer=Layer.TOP, power_w=0.0,
                  pins=[(0, 0), (2, 0)]),
        Component("C2", x=40, y=18, width=3, height=2,
                  layer=Layer.TOP, power_w=0.0,
                  pins=[(0, 0), (2, 0)]),
        Component("C3", x=20, y=38, width=3, height=2,
                  layer=Layer.TOP, power_w=0.0,
                  pins=[(0, 0), (2, 0)]),
        Component("R1", x=50, y=28, width=2, height=4,
                  layer=Layer.TOP, power_w=0.05,
                  pins=[(0, 0), (0, 3)]),
        Component("R2", x=10, y=42, width=2, height=4,
                  layer=Layer.TOP, power_w=0.05,
                  pins=[(0, 0), (0, 3)]),
        Component("LED", x=70, y=5, width=3, height=3,
                  layer=Layer.TOP, power_w=0.04,
                  pins=[(0, 0), (2, 0)]),
    ]

    for comp in components:
        grid.place_component(comp)

    nets = [
        Net("VCC",       [(10, 5, 0), (25, 20, 0)]),
        Net("GND",       [(5, 8, 0),  (25, 31, 0)]),
        Net("SPI_CLK",   [(36, 20, 0), (55, 8, 0)]),
        Net("SPI_MOSI",  [(36, 25, 0), (55, 13, 0)]),
        Net("SPI_MISO",  [(36, 26, 0), (62, 8, 0)]),
        Net("SPI_CS",    [(36, 31, 0), (62, 13, 0)]),
        Net("TX",        [(25, 26, 0), (58, 42, 1)]),
        Net("RX",        [(30, 31, 0), (67, 42, 1)]),
        Net("DATA2",     [(31, 31, 0), (61, 42, 1)]),
        Net("DECOUP1",   [(20, 12, 0), (25, 25, 0)]),
        Net("DECOUP2",   [(42, 18, 0), (36, 20, 0)]),
        Net("DECOUP3",   [(20, 38, 0), (25, 31, 0)]),
        Net("LED_SIG",   [(36, 25, 0), (50, 28, 0)]),
        Net("LED_R",     [(50, 31, 0), (70, 5, 0)]),
        Net("RESET",     [(10, 45, 0), (70, 5, 0)]),
    ]

    return grid, nets


# ------------------------------------------------------------------ #
# KiCad PARSER
# ------------------------------------------------------------------ #

def parse_kicad_pcb(content: str) -> tuple:
    """Parse a .kicad_pcb file and extract components + nets."""
    import re

    width_mm, height_mm = 50.0, 40.0

    edge_coords = re.findall(
        r'\(gr_line\s+\(start\s+([\d.]+)\s+([\d.]+)\)\s*'
        r'\(end\s+([\d.]+)\s+([\d.]+)\).*?Edge\.Cuts',
        content, re.DOTALL
    )
    if not edge_coords:
        edge_coords = re.findall(
            r'\(gr_rect\s+\(start\s+([\d.]+)\s+([\d.]+)\)\s*'
            r'\(end\s+([\d.]+)\s+([\d.]+)\)',
            content, re.DOTALL
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

    components = []
    fp_blocks = re.findall(
        r'\(footprint\s+"([^"]*)".*?\(at\s+([\d.]+)\s+([\d.]+).*?\)',
        content
    )

    for i, (name, x_str, y_str) in enumerate(fp_blocks):
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

    net_defs = re.findall(r'\(net\s+(\d+)\s+"([^"]*)"\)', content)
    net_map = {nid: name for nid, name in net_defs if name.strip()}

    pad_nets = {}
    pad_blocks = re.findall(
        r'\(pad\s+"[^"]*"\s+\w+\s+\w+\s+\(at\s+([\d.]+)\s+([\d.]+).*?\)'
        r'.*?\(net\s+(\d+)',
        content, re.DOTALL
    )
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

    return grid, nets


# ------------------------------------------------------------------ #
# ROUTING FUNCTIONS
# ------------------------------------------------------------------ #

def run_naive(grid: Grid, nets: list) -> RoutingResult:
    naive_weights = CostWeights(step=1.0, bend=0.0, via=5.0,
                                heat=0.0, congestion=0.0)
    router = MultiNetRouter(grid, naive_weights)
    return router.route_all(nets)


def run_ga(base_grid: Grid, nets: list, config: GAConfig,
           progress_callback=None) -> tuple:
    evaluator = FitnessEvaluator(base_grid, nets)
    gen_log = []

    def on_gen(gen, best, pop):
        avg = sum(g.fitness for g in pop) / len(pop)
        gen_log.append({
            'gen': gen, 'best': best.fitness, 'avg': avg
        })
        if progress_callback:
            progress_callback(gen, config.generations, best.fitness, avg)

    ga = GeneticAlgorithm(evaluator, config, on_generation=on_gen)
    best = ga.run()

    final_grid = base_grid.clone()
    router = MultiNetRouter(final_grid, best.to_cost_weights())
    result = router.route_all(nets, net_order=best.net_order)

    return result, best, final_grid, gen_log


# ------------------------------------------------------------------ #
# HELPER FUNCTIONS
# ------------------------------------------------------------------ #

def fig_to_image(fig):
    """Convert matplotlib figure to bytes for st.image()."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def export_csv(grid: Grid) -> str:
    """Export routed traces as CSV string."""
    lines = ["net,step,x,y,layer,x_mm,y_mm"]
    for net_name, path in grid.routed_paths.items():
        for i, (x, y, layer) in enumerate(path):
            mm_x = x * grid.resolution
            mm_y = y * grid.resolution
            lines.append(
                f"{net_name},{i},{x},{y},{layer},{mm_x:.3f},{mm_y:.3f}"
            )
    return "\n".join(lines)


def render_metric_card(label: str, value: str, delta: str = "",
                       delta_type: str = "neutral") -> str:
    """Render a styled metric card as HTML."""
    delta_html = ""
    if delta:
        delta_html = (
            f'<div class="metric-delta {delta_type}">{delta}</div>'
        )
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def compute_system_score(naive_result: RoutingResult,
                         ga_result: RoutingResult,
                         total_nets: int,
                         ga_time: float) -> dict:
    """Compute system score with sub-categories (0-100 each)."""
    n_opt = len(ga_result.paths)

    # Completion (0-100)
    completion = (n_opt / max(total_nets, 1)) * 100

    # Via efficiency (compare to naive, 100 = halved vias or fewer)
    if naive_result.total_vias > 0:
        via_ratio = ga_result.total_vias / naive_result.total_vias
        via_score = max(0, min(100, (1 - via_ratio) * 200))
    else:
        via_score = 100 if ga_result.total_vias == 0 else 50

    # Trace quality (shorter avg length per net = better)
    n_naive = max(len(naive_result.paths), 1)
    avg_naive = naive_result.total_trace_length / n_naive
    avg_opt = ga_result.total_trace_length / max(n_opt, 1)
    if avg_naive > 0:
        trace_ratio = avg_opt / avg_naive
        trace_score = max(0, min(100, (2 - trace_ratio) * 100))
    else:
        trace_score = 50

    # Bend quality
    if naive_result.total_bends > 0:
        bend_ratio = ga_result.total_bends / naive_result.total_bends
        bend_score = max(0, min(100, (1.5 - bend_ratio) * 100))
    else:
        bend_score = 80

    # Speed score (target <15s)
    if ga_time <= 5:
        speed_score = 100
    elif ga_time <= 15:
        speed_score = 100 - (ga_time - 5) * 3
    else:
        speed_score = max(0, 70 - (ga_time - 15) * 2)

    overall = (
        completion * 0.35 +
        via_score * 0.20 +
        trace_score * 0.20 +
        bend_score * 0.10 +
        speed_score * 0.15
    )

    return {
        'overall': overall,
        'completion': completion,
        'via_efficiency': via_score,
        'trace_quality': trace_score,
        'bend_quality': bend_score,
        'speed': speed_score,
    }


def get_score_class(score: float) -> str:
    if score >= 85:
        return "excellent"
    elif score >= 65:
        return "good"
    elif score >= 45:
        return "fair"
    else:
        return "poor"


def get_score_label(score: float) -> str:
    if score >= 85:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 45:
        return "Fair"
    else:
        return "Needs Work"


def get_per_net_stats(grid: Grid, result: RoutingResult,
                      nets: list) -> list:
    """Compute per-net routing statistics."""
    stats = []
    for net in nets:
        name = net.name
        if name in result.paths:
            path = result.paths[name]
            length = len(path) - 1
            vias = sum(1 for i in range(1, len(path))
                       if path[i][2] != path[i-1][2])
            bends = sum(1 for i in range(2, len(path))
                        if (path[i-1][0]-path[i-2][0],
                            path[i-1][1]-path[i-2][1]) !=
                           (path[i][0]-path[i-1][0],
                            path[i][1]-path[i-1][1]))
            cross_layer = "Yes" if vias > 0 else "No"
            stats.append({
                'Net': name,
                'Status': 'Routed',
                'Length': length,
                'Vias': vias,
                'Bends': bends,
                'Cross-Layer': cross_layer,
            })
        else:
            stats.append({
                'Net': name,
                'Status': 'Failed',
                'Length': '-',
                'Vias': '-',
                'Bends': '-',
                'Cross-Layer': '-',
            })
    return stats


# ------------------------------------------------------------------ #
# PLOTLY CHARTS
# ------------------------------------------------------------------ #

def create_fitness_chart(gen_log: list) -> go.Figure:
    """Create interactive Plotly fitness convergence chart."""
    df = pd.DataFrame(gen_log)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['gen'], y=df['best'],
        mode='lines+markers',
        name='Best Fitness',
        line=dict(color='#6366f1', width=3),
        marker=dict(size=6, color='#6366f1'),
        hovertemplate='Gen %{x}<br>Best: %{y:.0f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=df['gen'], y=df['avg'],
        mode='lines',
        name='Avg Fitness',
        line=dict(color='#f43f5e', width=1.5, dash='dash'),
        opacity=0.7,
        hovertemplate='Gen %{x}<br>Avg: %{y:.0f}<extra></extra>',
    ))

    # Add improvement annotation
    if len(df) >= 2:
        improvement = df['best'].iloc[0] - df['best'].iloc[-1]
        pct = improvement / df['best'].iloc[0] * 100 if df['best'].iloc[0] > 0 else 0
        fig.add_annotation(
            x=df['gen'].iloc[-1], y=df['best'].iloc[-1],
            text=f"  {pct:.1f}% improved",
            showarrow=False,
            font=dict(color='#22c55e', size=12),
            xanchor='left',
        )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.6)',
        font=dict(family='Inter', color='#c0c0e0'),
        xaxis=dict(
            title='Generation',
            gridcolor='rgba(99,102,241,0.1)',
            zeroline=False,
        ),
        yaxis=dict(
            title='Fitness (lower = better)',
            gridcolor='rgba(99,102,241,0.1)',
            zeroline=False,
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1,
        ),
        margin=dict(l=50, r=20, t=30, b=50),
        height=320,
        hovermode='x unified',
    )
    return fig


def create_weight_radar(genome: Genome) -> go.Figure:
    """Create radar chart of genome cost weights."""
    categories = ['Distance', 'Bend', 'Via', 'Heat', 'Congestion']
    values = [
        genome.distance_w,
        genome.bend_w,
        genome.via_w / 10,  # scale down for visibility
        genome.heat_w,
        genome.congestion_w,
    ]
    # Normalize to 0-1 range for radar
    max_val = max(values) if max(values) > 0 else 1
    norm_values = [v / max_val for v in values]
    norm_values.append(norm_values[0])  # close the polygon
    categories.append(categories[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=norm_values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.15)',
        line=dict(color='#6366f1', width=2),
        marker=dict(size=6, color='#a5b4fc'),
        name='Weights',
        hovertemplate='%{theta}: %{r:.2f}<extra></extra>',
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            bgcolor='rgba(26,26,46,0.4)',
            radialaxis=dict(
                visible=True, range=[0, 1.1],
                gridcolor='rgba(99,102,241,0.15)',
                linecolor='rgba(99,102,241,0.15)',
            ),
            angularaxis=dict(
                gridcolor='rgba(99,102,241,0.15)',
                linecolor='rgba(99,102,241,0.15)',
            ),
        ),
        font=dict(family='Inter', color='#c0c0e0', size=11),
        margin=dict(l=60, r=60, t=30, b=30),
        height=280,
        showlegend=False,
    )
    return fig


def create_comparison_bar(naive_result: RoutingResult,
                          ga_result: RoutingResult) -> go.Figure:
    """Create grouped bar chart comparing naive vs GA metrics."""
    categories = ['Trace Length', 'Vias', 'Bends']
    naive_vals = [
        naive_result.total_trace_length,
        naive_result.total_vias,
        naive_result.total_bends,
    ]
    ga_vals = [
        ga_result.total_trace_length,
        ga_result.total_vias,
        ga_result.total_bends,
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Naive', x=categories, y=naive_vals,
        marker_color='#475569',
        text=naive_vals,
        textposition='outside',
        textfont=dict(color='#94a3b8', size=11),
    ))
    fig.add_trace(go.Bar(
        name='GA-Optimized', x=categories, y=ga_vals,
        marker_color='#6366f1',
        text=ga_vals,
        textposition='outside',
        textfont=dict(color='#a5b4fc', size=11),
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.4)',
        font=dict(family='Inter', color='#c0c0e0'),
        barmode='group',
        xaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
        yaxis=dict(gridcolor='rgba(99,102,241,0.1)', title='Count'),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1,
        ),
        margin=dict(l=50, r=20, t=30, b=40),
        height=300,
    )
    return fig


# ------------------------------------------------------------------ #
# STREAMLIT APP
# ------------------------------------------------------------------ #

def main():
    # -- Header --
    st.markdown("""
    <div class="header-banner">
        <h1>FormaCore AI <span class="header-badge">v1.0</span></h1>
        <div class="subtitle">
            AI-Assisted Strategy-Optimized PCB Router &mdash; 2-Layer Board Optimization
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -- Sidebar --
    st.sidebar.markdown("### Board Configuration")

    source = st.sidebar.radio(
        "Board Source",
        ["Sample Board", "Upload .kicad_pcb"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### GA Parameters")
    pop_size = st.sidebar.slider("Population size", 5, 30, 15)
    generations = st.sidebar.slider("Generations", 5, 50, 20)
    early_stop = st.sidebar.slider("Early stop (stale gens)", 3, 15, 5)
    heat_sigma = st.sidebar.slider("Heat spread (sigma)", 3.0, 20.0, 10.0)

    st.sidebar.markdown("---")

    # -- Board loading --
    grid = None
    nets = None

    if source == "Upload .kicad_pcb":
        uploaded = st.sidebar.file_uploader(
            "Upload KiCad PCB file", type=["kicad_pcb"]
        )
        if uploaded:
            try:
                content = uploaded.read().decode('utf-8')
                grid, nets = parse_kicad_pcb(content)
                st.sidebar.success(
                    f"Parsed: {len(grid.components)} components, "
                    f"{len(nets)} nets"
                )
            except Exception as e:
                st.sidebar.error(f"Parse error: {e}")
                grid, nets = None, None
    else:
        grid, nets = create_sample_board()
        st.sidebar.info(
            f"Sample: {grid.width}x{grid.height} grid, "
            f"{len(grid.components)} components, {len(nets)} nets"
        )

    # -- Optimize button --
    run_btn = st.sidebar.button(
        "Run Optimization", type="primary",
        disabled=(grid is None),
        use_container_width=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align:center;color:#6b7280;font-size:0.75rem;'>"
        "FormaCore AI &mdash; PCB Routing Engine<br>"
        "A* Pathfinding + Genetic Algorithm"
        "</div>",
        unsafe_allow_html=True
    )

    if not run_btn and 'fc' not in st.session_state:
        # Show initial state
        st.markdown("""
        <div class="custom-info">
            <div class="icon">&#9889;</div>
            <div class="message">
                Configure parameters in the sidebar and click
                <strong>Run Optimization</strong> to start.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if grid is not None:
            heat = HeatModel(sigma=heat_sigma)
            heat.apply(grid)
            viz = BoardVisualizer(grid)
            fig = viz.render(
                title="Board Layout (Unrouted)",
                show_heat=True, headless=True
            )
            st.pyplot(fig)
            plt.close(fig)
        return

    # ============================================================== #
    # RUN OPTIMIZATION (or restore from cache)
    # ============================================================== #

    if run_btn:
        # Apply heat
        heat = HeatModel(sigma=heat_sigma)
        heat.apply(grid)
        base_grid = grid.clone()

        # -- Naive routing --
        with st.spinner("Running naive routing..."):
            t0 = time.time()
            naive_result = run_naive(grid, nets)
            naive_time = time.time() - t0
        naive_grid = grid

        # -- GA routing --
        config = GAConfig(
            population_size=pop_size,
            generations=generations,
            early_stop_gens=early_stop,
        )

        progress_bar = st.progress(0, text="GA Optimization: Initializing...")
        status_text = st.empty()

        def progress_cb(gen, total_gens, best_fit, avg_fit):
            pct = min((gen + 1) / total_gens, 1.0)
            progress_bar.progress(
                pct,
                text=f"Generation {gen+1}/{total_gens}  |  "
                     f"Best: {best_fit:.0f}  |  Avg: {avg_fit:.0f}"
            )

        t0 = time.time()
        ga_result, best_genome, ga_grid, gen_log = run_ga(
            base_grid, nets, config, progress_callback=progress_cb
        )
        ga_time = time.time() - t0

        progress_bar.progress(1.0, text="Optimization complete!")

        total = len(nets)
        n_naive = len(naive_result.paths)
        n_opt = len(ga_result.paths)

        # Cache results in session state for persistence
        st.session_state['fc'] = {
            'naive_result': naive_result, 'ga_result': ga_result,
            'ga_grid': ga_grid, 'naive_grid': naive_grid,
            'gen_log': gen_log, 'best_genome': best_genome,
            'ga_time': ga_time, 'naive_time': naive_time,
            'nets': nets, 'total': total,
            'n_naive': n_naive, 'n_opt': n_opt,
            'pop_size': pop_size, 'early_stop': early_stop,
        }
        # Reset assistant state for fresh optimization
        st.session_state.pop('board_ctrl', None)
        st.session_state.pop('asst_preview_result', None)
        st.session_state.pop('asst_apply_result', None)

    # Retrieve results (from fresh run or session cache)
    _fc = st.session_state['fc']
    naive_result = _fc['naive_result']
    ga_result = _fc['ga_result']
    ga_grid = _fc['ga_grid']
    naive_grid = _fc['naive_grid']
    gen_log = _fc['gen_log']
    best_genome = _fc['best_genome']
    ga_time = _fc['ga_time']
    naive_time = _fc['naive_time']
    nets = _fc['nets']
    total = _fc['total']
    n_naive = _fc['n_naive']
    n_opt = _fc['n_opt']
    pop_size = _fc['pop_size']
    early_stop = _fc['early_stop']

    # ============================================================== #
    # PROJECT MEMORY & COPILOT
    # ============================================================== #

    # Initialize project memory
    if 'project_memory' not in st.session_state:
        st.session_state['project_memory'] = ProjectMemory()
    memory: ProjectMemory = st.session_state['project_memory']

    # Log optimization if this was a fresh run
    if run_btn:
        w_dict = None
        if best_genome and hasattr(best_genome, 'weights'):
            w = best_genome.weights
            w_dict = {
                'step': w.step, 'bend': w.bend, 'via': w.via,
                'heat': w.heat, 'congestion': w.congestion,
            }
        memory.log_optimization(
            naive_routed=n_naive, naive_total=total,
            ga_routed=n_opt, ga_total=total,
            naive_vias=naive_result.total_vias,
            ga_vias=ga_result.total_vias,
            naive_trace=naive_result.total_trace_length,
            ga_trace=ga_result.total_trace_length,
            ga_generations=len(gen_log),
            ga_time=ga_time,
            weights=w_dict,
        )

    # Run Copilot analysis
    insights = analyze_board(
        ga_grid, nets, naive_result, ga_result, gen_log,
    )

    # Compute system scores (used by multiple tabs)
    scores = compute_system_score(
        naive_result, ga_result, total, ga_time
    )

    # ============================================================== #
    # RESULTS TABS
    # ============================================================== #

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "PCB Comparison",
        "Optimization Details",
        "System Score",
        "Export & Reports",
        "Assistant",
        "Design Copilot",
        "Project Timeline",
    ])

    # ================================================================ #
    # TAB 1: PCB COMPARISON
    # ================================================================ #
    with tab1:
        st.markdown('<div class="section-header">Routing Metrics</div>',
                    unsafe_allow_html=True)

        # Metric cards row
        avg_n = naive_result.total_trace_length / max(n_naive, 1)
        avg_o = ga_result.total_trace_length / max(n_opt, 1)

        if naive_result.total_vias > 0:
            via_red = ((naive_result.total_vias - ga_result.total_vias)
                       / naive_result.total_vias * 100)
        else:
            via_red = 0

        cols = st.columns(5)
        with cols[0]:
            delta_nets = n_opt - n_naive
            delta_str = f"+{delta_nets} nets" if delta_nets > 0 else (
                f"{delta_nets} nets" if delta_nets < 0 else "same"
            )
            dt = "positive" if delta_nets > 0 else (
                "negative" if delta_nets < 0 else "neutral"
            )
            st.markdown(render_metric_card(
                "Completion", f"{n_opt}/{total}", delta_str, dt
            ), unsafe_allow_html=True)

        with cols[1]:
            avg_diff = avg_o - avg_n
            dt = "positive" if avg_diff < 0 else (
                "negative" if avg_diff > 0 else "neutral"
            )
            st.markdown(render_metric_card(
                "Avg Length/Net", f"{avg_o:.1f}",
                f"{avg_diff:+.1f} steps", dt
            ), unsafe_allow_html=True)

        with cols[2]:
            vd = ga_result.total_vias - naive_result.total_vias
            dt = "positive" if vd < 0 else (
                "negative" if vd > 0 else "neutral"
            )
            st.markdown(render_metric_card(
                "Vias", str(ga_result.total_vias),
                f"{vd:+d} vs naive", dt
            ), unsafe_allow_html=True)

        with cols[3]:
            bd = ga_result.total_bends - naive_result.total_bends
            dt = "positive" if bd < 0 else (
                "negative" if bd > 0 else "neutral"
            )
            st.markdown(render_metric_card(
                "Bends", str(ga_result.total_bends),
                f"{bd:+d} vs naive", dt
            ), unsafe_allow_html=True)

        with cols[4]:
            st.markdown(render_metric_card(
                "Via Reduction",
                f"{via_red:.0f}%" if naive_result.total_vias > 0 else "N/A",
                "from naive baseline",
                "positive" if via_red > 0 else "neutral"
            ), unsafe_allow_html=True)

        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

        # Comparison bar chart
        st.markdown('<div class="section-header">Metric Comparison</div>',
                    unsafe_allow_html=True)
        fig_bar = create_comparison_bar(naive_result, ga_result)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

        # Side-by-side images
        st.markdown(
            '<div class="section-header">Board Visualization</div>',
            unsafe_allow_html=True
        )
        left, right = st.columns(2)

        with left:
            st.markdown("**Naive Routing**")
            viz_n = BoardVisualizer(naive_grid)
            fig_n = viz_n.render(
                title="Naive Routing",
                show_heat=True, result=naive_result, headless=True
            )
            st.pyplot(fig_n)
            plt.close(fig_n)

        with right:
            st.markdown("**GA-Optimized Routing**")
            viz_o = BoardVisualizer(ga_grid)
            fig_o = viz_o.render(
                title="GA-Optimized Routing",
                show_heat=True, result=ga_result, headless=True
            )
            st.pyplot(fig_o)
            plt.close(fig_o)

        # Full comparison
        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
        viz_comp = BoardVisualizer(naive_grid)
        fig_c = viz_comp.render_comparison(
            naive_grid, naive_result,
            ga_grid, ga_result,
            title_a="Naive", title_b="GA-Optimized",
            headless=True
        )
        st.pyplot(fig_c)
        plt.close(fig_c)

        # Failed nets
        if naive_result.failed_nets:
            st.warning(
                f"Naive router failed: {', '.join(naive_result.failed_nets)}"
            )
        if ga_result.failed_nets:
            st.error(
                f"GA router failed: {', '.join(ga_result.failed_nets)}"
            )

    # ================================================================ #
    # TAB 2: OPTIMIZATION DETAILS
    # ================================================================ #
    with tab2:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown(
                '<div class="section-header">'
                'Fitness Convergence</div>',
                unsafe_allow_html=True
            )
            if gen_log:
                fig_fit = create_fitness_chart(gen_log)
                st.plotly_chart(fig_fit, use_container_width=True)

            st.markdown('<hr class="styled-divider">',
                        unsafe_allow_html=True)

            # Per-net routing table
            st.markdown(
                '<div class="section-header">'
                'Per-Net Routing Details</div>',
                unsafe_allow_html=True
            )
            net_stats = get_per_net_stats(ga_grid, ga_result, nets)
            # Build HTML table
            table_html = '<table class="net-table"><thead><tr>'
            for col_name in ['Net', 'Status', 'Length', 'Vias',
                             'Bends', 'Cross-Layer']:
                table_html += f'<th>{col_name}</th>'
            table_html += '</tr></thead><tbody>'
            for row in net_stats:
                table_html += '<tr>'
                for key in ['Net', 'Status', 'Length', 'Vias',
                            'Bends', 'Cross-Layer']:
                    val = row[key]
                    if key == 'Status':
                        cls = ('status-routed' if val == 'Routed'
                               else 'status-failed')
                        table_html += f'<td class="{cls}">{val}</td>'
                    else:
                        table_html += f'<td>{val}</td>'
                table_html += '</tr>'
            table_html += '</tbody></table>'
            st.markdown(table_html, unsafe_allow_html=True)

        with col_right:
            st.markdown(
                '<div class="section-header">'
                'Weight Strategy</div>',
                unsafe_allow_html=True
            )
            fig_radar = create_weight_radar(best_genome)
            st.plotly_chart(fig_radar, use_container_width=True)

            # Genome details
            st.markdown(
                '<div class="section-header">'
                'Best Genome</div>',
                unsafe_allow_html=True
            )
            genome_data = {
                "net_order": best_genome.net_order,
                "distance_w": round(best_genome.distance_w, 3),
                "bend_w": round(best_genome.bend_w, 3),
                "via_w": round(best_genome.via_w, 3),
                "heat_w": round(best_genome.heat_w, 3),
                "congestion_w": round(best_genome.congestion_w, 3),
                "fitness": round(best_genome.fitness, 1),
            }
            st.json(genome_data)

            # Runtime stats
            st.markdown(
                '<div class="section-header">'
                'Runtime</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"""
            - Naive routing: **{naive_time:.3f}s**
            - GA optimization: **{ga_time:.1f}s**
            - Generations run: **{len(gen_log)}**
            - Population: **{pop_size}**
            - Early stop: **{early_stop}** stale gens
            """)

        # AI Explainability
        st.markdown('<hr class="styled-divider">',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">'
            'AI Strategy Explanation</div>',
            unsafe_allow_html=True
        )

        exp_cols = st.columns(2)
        explanations = []

        if best_genome.via_w > 25:
            explanations.append((
                "High Via Penalty",
                f"Weight: {best_genome.via_w:.1f}",
                "AI discourages layer switching, reducing manufacturing "
                "cost and improving signal integrity.",
                "#22c55e"
            ))
        if best_genome.bend_w > 8:
            explanations.append((
                "High Bend Penalty",
                f"Weight: {best_genome.bend_w:.1f}",
                "AI prefers straight traces, minimizing impedance "
                "discontinuities for better signal quality.",
                "#3b82f6"
            ))
        if best_genome.heat_w > 3:
            explanations.append((
                "Thermal Avoidance",
                f"Weight: {best_genome.heat_w:.1f}",
                "AI actively routes traces away from thermal hotspots "
                "to improve reliability.",
                "#f59e0b"
            ))
        if best_genome.congestion_w > 4:
            explanations.append((
                "Congestion Spreading",
                f"Weight: {best_genome.congestion_w:.1f}",
                "AI distributes traces to avoid routing bottlenecks "
                "and maintain clearance.",
                "#8b5cf6"
            ))
        if best_genome.distance_w < 0.7:
            explanations.append((
                "Relaxed Distance",
                f"Weight: {best_genome.distance_w:.2f}",
                "AI trades longer paths for better via/bend/thermal "
                "performance.",
                "#06b6d4"
            ))
        if not explanations:
            explanations.append((
                "Balanced Strategy",
                "All weights moderate",
                "AI found a general-purpose balanced strategy that "
                "optimizes all objectives simultaneously.",
                "#6366f1"
            ))

        for i, (title, subtitle, desc, color) in enumerate(explanations):
            col = exp_cols[i % 2]
            with col:
                st.markdown(f"""
                <div class="strategy-card" style="border-left-color:{color};">
                    <div class="weight-name">{title}
                        <span style="color:{color};font-size:0.75rem;">
                            ({subtitle})
                        </span>
                    </div>
                    <div class="weight-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # ================================================================ #
    # TAB 3: SYSTEM SCORE
    # ================================================================ #
    with tab3:
        overall = scores['overall']
        sc = get_score_class(overall)
        sl = get_score_label(overall)

        score_col, detail_col = st.columns([1, 2])

        with score_col:
            st.markdown(f"""
            <div class="score-container">
                <div class="score-ring {sc}">
                    <div class="score-number {sc}">{overall:.0f}</div>
                    <div class="score-label {sc}">{sl}</div>
                </div>
                <div style="color:#9d9dcc;font-size:0.85rem;margin-top:0.5rem;">
                    Overall System Score
                </div>
            </div>
            """, unsafe_allow_html=True)

        with detail_col:
            st.markdown(
                '<div class="section-header">'
                'Score Breakdown</div>',
                unsafe_allow_html=True
            )

            sub_scores = [
                ('Completion', scores['completion'], '#22c55e'),
                ('Via Efficiency', scores['via_efficiency'], '#6366f1'),
                ('Trace Quality', scores['trace_quality'], '#3b82f6'),
                ('Bend Quality', scores['bend_quality'], '#8b5cf6'),
                ('Speed', scores['speed'], '#f59e0b'),
            ]

            for label, val, color in sub_scores:
                st.markdown(f"""
                <div class="subscore-row">
                    <div class="subscore-label">{label}</div>
                    <div class="subscore-bar-bg">
                        <div class="subscore-bar-fill"
                             style="width:{val:.0f}%;background:{color};">
                        </div>
                    </div>
                    <div class="subscore-value">{val:.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<hr class="styled-divider">',
                        unsafe_allow_html=True)

            # Score interpretation
            if overall >= 85:
                interp = (
                    "The GA optimizer produced an excellent routing solution. "
                    "High completion rate, efficient via usage, and fast "
                    "convergence indicate the strategy is well-suited for "
                    "this board layout."
                )
            elif overall >= 65:
                interp = (
                    "Good routing quality with room for improvement. Consider "
                    "increasing the population size or generations to allow "
                    "the GA to explore more of the solution space."
                )
            elif overall >= 45:
                interp = (
                    "The routing solution is functional but sub-optimal. "
                    "The board may have challenging routing constraints. "
                    "Try adjusting heat spread, increasing population size, "
                    "or tuning early stop parameters."
                )
            else:
                interp = (
                    "The routing solution needs significant improvement. "
                    "This may indicate a very constrained board layout or "
                    "insufficient GA parameters. Consider larger population "
                    "size and more generations."
                )

            st.markdown(f"""
            <div class="strategy-card" style="border-left-color:{
                {'excellent':'#22c55e','good':'#3b82f6',
                 'fair':'#f59e0b','poor':'#ef4444'}[sc]
            };">
                <div class="weight-name">Interpretation</div>
                <div class="weight-desc">{interp}</div>
            </div>
            """, unsafe_allow_html=True)

        # Comparison summary
        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">'
            'Naive vs Optimized Summary</div>',
            unsafe_allow_html=True
        )

        sum_cols = st.columns(4)
        metrics = [
            ("Nets Routed",
             f"{n_naive}/{total}", f"{n_opt}/{total}"),
            ("Total Trace Length",
             str(naive_result.total_trace_length),
             str(ga_result.total_trace_length)),
            ("Total Vias",
             str(naive_result.total_vias),
             str(ga_result.total_vias)),
            ("Total Bends",
             str(naive_result.total_bends),
             str(ga_result.total_bends)),
        ]
        for i, (lbl, naive_v, ga_v) in enumerate(metrics):
            with sum_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{lbl}</div>
                    <div style="display:flex;justify-content:center;
                                align-items:center;gap:0.8rem;
                                margin-top:0.4rem;">
                        <div style="text-align:center;">
                            <div style="color:#64748b;font-size:0.7rem;">
                                NAIVE</div>
                            <div style="color:#94a3b8;font-size:1.1rem;
                                        font-weight:600;">{naive_v}</div>
                        </div>
                        <div style="color:#4b5563;font-size:1rem;">
                            &rarr;</div>
                        <div style="text-align:center;">
                            <div style="color:#818cf8;font-size:0.7rem;">
                                GA</div>
                            <div style="color:#a5b4fc;font-size:1.1rem;
                                        font-weight:600;">{ga_v}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ================================================================ #
    # TAB 4: EXPORT
    # ================================================================ #
    with tab4:
        st.markdown(
            '<div class="section-header">Export Results</div>',
            unsafe_allow_html=True
        )

        # CSV export
        csv_data = export_csv(ga_grid)
        st.download_button(
            label="Download Routed Traces (CSV)",
            data=csv_data,
            file_name="formacore_routes.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Download Images</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            viz_n2 = BoardVisualizer(naive_grid)
            fig_dl = viz_n2.render(
                title="Naive Routing", show_heat=True,
                result=naive_result, headless=True
            )
            buf = fig_to_image(fig_dl)
            st.download_button(
                label="Naive Routing (PNG)",
                data=buf,
                file_name="naive_routing.png",
                mime="image/png",
                use_container_width=True,
            )

        with col2:
            viz_o2 = BoardVisualizer(ga_grid)
            fig_dl2 = viz_o2.render(
                title="GA-Optimized Routing", show_heat=True,
                result=ga_result, headless=True
            )
            buf2 = fig_to_image(fig_dl2)
            st.download_button(
                label="Optimized Routing (PNG)",
                data=buf2,
                file_name="ga_routing.png",
                mime="image/png",
                use_container_width=True,
            )

        with col3:
            viz_c2 = BoardVisualizer(naive_grid)
            fig_comp = viz_c2.render_comparison(
                naive_grid, naive_result,
                ga_grid, ga_result,
                headless=True
            )
            buf3 = fig_to_image(fig_comp)
            st.download_button(
                label="Comparison (PNG)",
                data=buf3,
                file_name="comparison.png",
                mime="image/png",
                use_container_width=True,
            )

        # Summary report
        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Results Summary</div>',
            unsafe_allow_html=True
        )

        via_red_display = (
            f"{via_red:.1f}%" if naive_result.total_vias > 0 else "N/A"
        )
        summary = (
            f"FormaCore AI Routing Report\n"
            f"{'=' * 40}\n"
            f"Board: {ga_grid.width}x{ga_grid.height} grid "
            f"({ga_grid.width * ga_grid.resolution:.0f}mm x "
            f"{ga_grid.height * ga_grid.resolution:.0f}mm)\n"
            f"Components: {len(ga_grid.components)}\n"
            f"Nets: {total}\n\n"
            f"Naive:     {n_naive}/{total} routed, "
            f"len={naive_result.total_trace_length}, "
            f"vias={naive_result.total_vias}, "
            f"bends={naive_result.total_bends}\n"
            f"Optimized: {n_opt}/{total} routed, "
            f"len={ga_result.total_trace_length}, "
            f"vias={ga_result.total_vias}, "
            f"bends={ga_result.total_bends}\n\n"
            f"Via reduction: {via_red_display}\n"
            f"GA time: {ga_time:.1f}s ({len(gen_log)} generations)\n"
            f"System Score: {scores['overall']:.0f}/100 ({sl})\n"
        )
        st.code(summary, language="text")

        st.download_button(
            label="Download Report (TXT)",
            data=summary,
            file_name="formacore_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # ----- Design Report Generation ----- #
        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Design Report</div>',
            unsafe_allow_html=True
        )

        report_cols = st.columns(2)
        with report_cols[0]:
            md_report = generate_report(
                ga_grid, nets, naive_result, ga_result, gen_log,
                best_genome, naive_time, ga_time, insights, scores,
                fmt='markdown',
            )
            st.download_button(
                label="Download Report (Markdown)",
                data=md_report,
                file_name="formacore_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with report_cols[1]:
            txt_full_report = generate_report(
                ga_grid, nets, naive_result, ga_result, gen_log,
                best_genome, naive_time, ga_time, insights, scores,
                fmt='text',
            )
            st.download_button(
                label="Download Full Report (TXT)",
                data=txt_full_report,
                file_name="formacore_full_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # ----- Submission Package ----- #
        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Submission Package</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            "One-click export: report + images + data + timeline, "
            "bundled as a ZIP file."
        )

        if st.button(
            "Prepare Submission Package (ZIP)",
            use_container_width=True,
            type="primary",
            key="submission_pkg_btn",
        ):
            with st.spinner("Building submission package..."):
                zip_bytes = prepare_submission(
                    grid=ga_grid,
                    nets=nets,
                    naive_result=naive_result,
                    ga_result=ga_result,
                    naive_grid=naive_grid,
                    ga_grid=ga_grid,
                    gen_log=gen_log,
                    best_genome=best_genome,
                    naive_time=naive_time,
                    ga_time=ga_time,
                    insights=insights,
                    scores=scores,
                    memory=memory,
                )
                st.session_state['submission_zip'] = zip_bytes
                memory.log_export('submission_package',
                                  'formacore_submission.zip')

        if 'submission_zip' in st.session_state:
            st.download_button(
                label="Download Submission Package (ZIP)",
                data=st.session_state['submission_zip'],
                file_name="formacore_submission.zip",
                mime="application/zip",
                use_container_width=True,
            )

    # ================================================================ #
    # TAB 5: ASSISTANT
    # ================================================================ #
    with tab5:
        st.markdown(
            '<div class="section-header">Board Assistant</div>',
            unsafe_allow_html=True
        )

        # Initialize BoardController in session state
        if 'board_ctrl' not in st.session_state:
            st.session_state['board_ctrl'] = BoardController(
                ga_grid, nets
            )

        ctrl = st.session_state['board_ctrl']

        # --- Two-column layout ---
        action_col, preview_col = st.columns([2, 3])

        with action_col:
            st.markdown(
                '<div class="section-header">Action</div>',
                unsafe_allow_html=True
            )

            # Action dropdown
            ui_actions = get_ui_actions()
            action_options = {label: aid for aid, label in ui_actions}
            selected_label = st.selectbox(
                "Select Action",
                list(action_options.keys()),
                key="assistant_action"
            )
            selected_action = action_options[selected_label]
            schema = ACTION_REGISTRY[selected_action]
            st.caption(schema['description'])

            # Dynamic parameter form
            st.markdown("**Parameters**")
            payload = {}
            fields = schema.get('fields', {})

            if not fields:
                st.info("No parameters needed for this action.")

            for field_name, spec in fields.items():
                ftype = spec.get('type', 'str')
                required = spec.get('required', False)
                desc = spec.get('description', field_name)
                default = spec.get('default')
                label_txt = f"{desc} {'*' if required else ''}"
                widget_key = f"asst_{selected_action}_{field_name}"

                if 'allowed' in spec:
                    val = st.selectbox(
                        label_txt, spec['allowed'], key=widget_key
                    )
                    payload[field_name] = val

                elif ftype == 'int':
                    min_v = spec.get('min', 0)
                    max_v = spec.get('max', 200)
                    def_v = default if default is not None else min_v
                    val = st.number_input(
                        label_txt, min_value=min_v, max_value=max_v,
                        value=def_v, step=1, key=widget_key
                    )
                    payload[field_name] = int(val)

                elif ftype == 'float':
                    min_v = spec.get('min', 0.0)
                    max_v = spec.get('max', 100.0)
                    def_v = default if default is not None else min_v
                    val = st.number_input(
                        label_txt, min_value=min_v, max_value=max_v,
                        value=float(def_v), step=0.1, key=widget_key
                    )
                    payload[field_name] = float(val)

                elif ftype == 'str':
                    if field_name == 'ref' and selected_action != 'place_component':
                        comp_names = [c.name for c in ctrl.grid.components]
                        if comp_names:
                            val = st.selectbox(
                                label_txt, comp_names, key=widget_key
                            )
                        else:
                            val = st.text_input(
                                label_txt, value="", key=widget_key
                            )
                    elif field_name == 'net_name':
                        routed = list(ctrl.grid.routed_paths.keys())
                        if routed:
                            val = st.selectbox(
                                label_txt, routed, key=widget_key
                            )
                        else:
                            val = st.text_input(
                                label_txt, value="", key=widget_key
                            )
                    else:
                        val = st.text_input(
                            label_txt,
                            value=str(default) if default else "",
                            key=widget_key,
                        )
                    payload[field_name] = val

                elif ftype == 'list_str':
                    comp_names = [c.name for c in ctrl.grid.components]
                    val = st.multiselect(
                        label_txt, comp_names, key=widget_key
                    )
                    payload[field_name] = val

                elif ftype == 'dict':
                    st.info("Using default weights.")

            # Filter payload: keep required fields and non-empty optionals
            cleaned_payload = {}
            for k, v in payload.items():
                if v is not None and v != "" and v != []:
                    cleaned_payload[k] = v
                elif fields.get(k, {}).get('required', False):
                    cleaned_payload[k] = v

            st.markdown(
                '<hr class="styled-divider">',
                unsafe_allow_html=True
            )

            # Action buttons
            btn_cols = st.columns(4)
            preview_clicked = btn_cols[0].button(
                "Preview", type="primary",
                use_container_width=True, key="asst_preview_btn"
            )
            has_preview = (
                'asst_preview_result' in st.session_state
                and st.session_state['asst_preview_result'].get(
                    'status') == 'preview_ready'
            )
            apply_clicked = btn_cols[1].button(
                "Apply",
                use_container_width=True,
                key="asst_apply_btn",
                disabled=(not has_preview),
            )
            undo_clicked = btn_cols[2].button(
                "Undo", use_container_width=True, key="asst_undo_btn"
            )
            redo_clicked = btn_cols[3].button(
                "Redo", use_container_width=True, key="asst_redo_btn"
            )

            # Handle button clicks
            if preview_clicked:
                result = ctrl.preview(selected_action, cleaned_payload)
                st.session_state['asst_preview_result'] = result
                st.session_state.pop('asst_apply_result', None)
                st.rerun()

            if apply_clicked and has_preview:
                result = ctrl.apply()
                st.session_state['asst_apply_result'] = result
                st.session_state.pop('asst_preview_result', None)
                st.rerun()

            if undo_clicked:
                result = ctrl.undo()
                st.session_state['asst_apply_result'] = result
                st.session_state.pop('asst_preview_result', None)
                st.rerun()

            if redo_clicked:
                result = ctrl.redo()
                st.session_state['asst_apply_result'] = result
                st.session_state.pop('asst_preview_result', None)
                st.rerun()

        with preview_col:
            st.markdown(
                '<div class="section-header">Preview & Results</div>',
                unsafe_allow_html=True
            )

            # Show apply feedback
            if 'asst_apply_result' in st.session_state:
                ar = st.session_state['asst_apply_result']
                if ar.get('status') == 'applied':
                    st.success(
                        f"Applied: {ar.get('action', 'action')} "
                        f"(version {ar.get('version', '?')})"
                    )
                elif ar.get('status') == 'ok':
                    st.success(ar.get('msg', 'Done'))
                elif ar.get('status') == 'error':
                    st.error(ar.get('msg', 'Error'))

            # Show preview result
            preview_result = st.session_state.get('asst_preview_result')
            if preview_result:
                pstatus = preview_result.get('status')

                if pstatus == 'validation_error':
                    st.error("Validation failed:")
                    for err in preview_result.get('errors', []):
                        st.markdown(
                            f"- **{err['field']}**: {err['msg']}"
                        )

                elif pstatus == 'preview_ready':
                    # Action result
                    action_result = preview_result.get(
                        'action_result', {}
                    )
                    if action_result.get('status') == 'ok':
                        st.success("Action succeeded in sandbox")
                    else:
                        st.warning(
                            f"Action issue: "
                            f"{action_result.get('msg', 'unknown')}"
                        )

                    # DRC report
                    drc = preview_result.get('drc', {})
                    if preview_result.get('drc_passed'):
                        st.success(
                            f"DRC Passed "
                            f"({drc.get('warnings', 0)} warnings)"
                        )
                    else:
                        st.error(
                            f"DRC Failed: "
                            f"{drc.get('errors', 0)} errors, "
                            f"{drc.get('warnings', 0)} warnings"
                        )
                        for issue in drc.get('issues', [])[:10]:
                            sev = issue.get('severity', '')
                            icon = "ERR" if sev == 'error' else "WARN"
                            st.markdown(
                                f"- **[{icon}]** {issue['msg']}"
                            )

                    # Preview image
                    preview_img = ctrl.get_preview_image()
                    if preview_img:
                        st.image(
                            preview_img, caption="Preview",
                            use_container_width=True
                        )

                    # Details expander
                    with st.expander("Action Details"):
                        st.json(action_result)

            else:
                # No preview yet — show instruction
                st.markdown("""
                <div class="custom-info">
                    <div class="icon">&#128269;</div>
                    <div class="message">
                        Select an action, configure parameters,
                        and click <strong>Preview</strong> to see
                        changes before applying.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Current board state
            st.markdown(
                '<hr class="styled-divider">',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="section-header">Current Board State</div>',
                unsafe_allow_html=True
            )
            viz_asst = BoardVisualizer(ctrl.grid)
            fig_asst = viz_asst.render(
                title="Current Board",
                show_heat=True, headless=True
            )
            st.pyplot(fig_asst)
            plt.close(fig_asst)

            # Board info metrics
            board_info = ctrl.get_board_info()
            info_cols = st.columns(4)
            with info_cols[0]:
                st.markdown(render_metric_card(
                    "Components",
                    str(board_info.get('components', 0)),
                ), unsafe_allow_html=True)
            with info_cols[1]:
                st.markdown(render_metric_card(
                    "Routed Nets",
                    str(board_info.get('routed_nets', 0)),
                ), unsafe_allow_html=True)
            with info_cols[2]:
                st.markdown(render_metric_card(
                    "Grid Size",
                    f"{board_info.get('width', 0)}x"
                    f"{board_info.get('height', 0)}",
                ), unsafe_allow_html=True)
            with info_cols[3]:
                st.markdown(render_metric_card(
                    "Board (mm)",
                    f"{board_info.get('width_mm', 0):.0f}x"
                    f"{board_info.get('height_mm', 0):.0f}",
                ), unsafe_allow_html=True)

        # Version history (full width below columns)
        st.markdown(
            '<hr class="styled-divider">',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-header">Version History</div>',
            unsafe_allow_html=True
        )
        history = ctrl.get_history()
        if history:
            hist_data = []
            for h in history:
                hist_data.append({
                    'Version': h['version'],
                    'Action': h['action'],
                    'Status': h['status'],
                    'Time': h['timestamp'],
                })
            st.dataframe(
                pd.DataFrame(hist_data),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No version history yet.")

        # Export action log
        if len(history) > 1:
            log_json = ctrl.export_history_log()
            st.download_button(
                label="Download Action Log (JSON)",
                data=log_json,
                file_name="formacore_action_log.json",
                mime="application/json",
                use_container_width=True,
            )

    # ================================================================ #
    # TAB 6: DESIGN COPILOT
    # ================================================================ #
    with tab6:
        st.markdown(
            '<div class="section-header">Design Insight Copilot</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            "Rule-based analysis of your routing results. "
            "Raw metrics converted into actionable feedback."
        )

        if not insights:
            st.success("No issues detected. Your design looks clean.")
        else:
            # Summary counts
            n_crit = sum(1 for i in insights if i.severity == 'critical')
            n_warn = sum(1 for i in insights if i.severity == 'warning')
            n_info = sum(1 for i in insights if i.severity == 'info')

            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.markdown(render_metric_card(
                    "Critical", str(n_crit),
                ), unsafe_allow_html=True)
            with summary_cols[1]:
                st.markdown(render_metric_card(
                    "Warnings", str(n_warn),
                ), unsafe_allow_html=True)
            with summary_cols[2]:
                st.markdown(render_metric_card(
                    "Info", str(n_info),
                ), unsafe_allow_html=True)

            st.markdown(
                '<hr class="styled-divider">',
                unsafe_allow_html=True
            )

            # Filter by category
            categories = sorted(
                set(i.category for i in insights)
            )
            cat_filter = st.multiselect(
                "Filter by category",
                categories,
                default=categories,
                key="copilot_cat_filter",
            )

            for ins in insights:
                if ins.category not in cat_filter:
                    continue

                if ins.severity == 'critical':
                    color = '#ef4444'
                    badge = 'CRITICAL'
                elif ins.severity == 'warning':
                    color = '#f59e0b'
                    badge = 'WARNING'
                else:
                    color = '#6366f1'
                    badge = 'INFO'

                st.markdown(f"""
                <div style="
                    border-left: 4px solid {color};
                    padding: 12px 16px;
                    margin: 8px 0;
                    background: rgba(255,255,255,0.03);
                    border-radius: 0 8px 8px 0;
                ">
                    <div style="display:flex;align-items:center;gap:8px;
                                margin-bottom:6px;">
                        <span style="
                            background:{color};
                            color:white;
                            padding:2px 8px;
                            border-radius:4px;
                            font-size:0.7rem;
                            font-weight:700;
                        ">{badge}</span>
                        <span style="
                            color:{color};
                            font-size:0.75rem;
                            text-transform:uppercase;
                        ">{ins.category}</span>
                    </div>
                    <div style="font-weight:600;font-size:0.95rem;
                                margin-bottom:4px;">
                        {ins.title}
                    </div>
                    <div style="font-size:0.85rem;opacity:0.85;
                                margin-bottom:8px;">
                        {ins.message}
                    </div>
                    <div style="
                        font-size:0.8rem;
                        padding:8px 12px;
                        background:rgba(99,102,241,0.1);
                        border-radius:6px;
                        border:1px solid rgba(99,102,241,0.2);
                    ">
                        Suggestion: {ins.suggestion}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ================================================================ #
    # TAB 7: PROJECT TIMELINE
    # ================================================================ #
    with tab7:
        st.markdown(
            '<div class="section-header">Project Timeline</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            "Automatic knowledge capture: every optimization, "
            "assistant action, and export is logged with context."
        )

        # Add note form
        with st.expander("Add a Design Note"):
            note_title = st.text_input(
                "Title", key="memory_note_title",
                placeholder="e.g. Changed layout strategy",
            )
            note_text = st.text_area(
                "Note", key="memory_note_text",
                placeholder="Describe what you changed and why...",
            )
            if st.button("Save Note", key="memory_save_note"):
                if note_title and note_text:
                    memory.log_note(note_title, note_text)
                    st.success("Note saved to timeline.")
                    st.rerun()

        st.markdown(
            '<hr class="styled-divider">',
            unsafe_allow_html=True
        )

        timeline = memory.get_timeline()

        if not timeline:
            st.info("No events recorded yet. Run an optimization to start.")
        else:
            # Display event count
            st.markdown(render_metric_card(
                "Events", str(len(timeline)),
            ), unsafe_allow_html=True)

            st.markdown(
                '<hr class="styled-divider">',
                unsafe_allow_html=True
            )

            # Timeline display
            type_icons = {
                'optimization': 'OPT',
                'assistant': 'AST',
                'parameter': 'CFG',
                'export': 'EXP',
                'note': 'NOTE',
            }
            type_colors = {
                'optimization': '#6366f1',
                'assistant': '#10b981',
                'parameter': '#f59e0b',
                'export': '#8b5cf6',
                'note': '#06b6d4',
            }

            for entry in reversed(timeline):
                etype = entry['type']
                icon = type_icons.get(etype, etype.upper()[:3])
                color = type_colors.get(etype, '#6b7280')

                st.markdown(f"""
                <div style="
                    display:flex;
                    gap:12px;
                    padding:10px 0;
                    border-bottom:1px solid rgba(255,255,255,0.06);
                ">
                    <div style="
                        min-width:50px;
                        text-align:center;
                    ">
                        <span style="
                            background:{color};
                            color:white;
                            padding:3px 8px;
                            border-radius:4px;
                            font-size:0.65rem;
                            font-weight:700;
                        ">{icon}</span>
                    </div>
                    <div style="flex:1;">
                        <div style="
                            font-weight:600;
                            font-size:0.9rem;
                            margin-bottom:2px;
                        ">{entry['title']}</div>
                        <div style="
                            font-size:0.8rem;
                            opacity:0.75;
                        ">{entry['description']}</div>
                        <div style="
                            font-size:0.7rem;
                            opacity:0.5;
                            margin-top:4px;
                        ">{entry['timestamp']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Export timeline
        st.markdown(
            '<hr class="styled-divider">',
            unsafe_allow_html=True
        )
        export_cols = st.columns(2)
        with export_cols[0]:
            if memory.count > 0:
                st.download_button(
                    label="Export Timeline (JSON)",
                    data=memory.export_json(),
                    file_name="formacore_timeline.json",
                    mime="application/json",
                    use_container_width=True,
                )
        with export_cols[1]:
            if memory.count > 0:
                st.download_button(
                    label="Export Summary (TXT)",
                    data=memory.get_summary(),
                    file_name="formacore_session_summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
