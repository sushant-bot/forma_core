"""
FormaCore AI - generate_samples.py
Generates sample PNG images for documentation and README.
Outputs to samples/ directory.
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from core.grid import Grid, Component, Net, Layer
from core.heat import HeatModel
from core.congestion import CongestionTracker
from router.cost import CostWeights
from router.multi_router import MultiNetRouter
from ai.genome import Genome
from ai.fitness import FitnessEvaluator, FitnessWeights
from ai.ga import GeneticAlgorithm, GAConfig
from visualize.plot import BoardVisualizer
from workflow.copilot import analyze_board


# ------------------------------------------------------------------ #
# Setup
# ------------------------------------------------------------------ #

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'samples')
os.makedirs(SAMPLES_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(SAMPLES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Board setup (same as main.py)
# ------------------------------------------------------------------ #

def create_sample_board():
    grid = Grid(width=80, height=60, layers=2, resolution_mm=0.5)

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
# Run routing
# ------------------------------------------------------------------ #

print("=" * 52)
print("  FormaCore AI — Sample Image Generator")
print("=" * 52)

grid, nets = create_sample_board()
heat = HeatModel(sigma=10.0)
heat.apply(grid)
base_grid = grid.clone()

# Naive routing
print("\n[1/6] Running naive routing...")
naive_weights = CostWeights(step=1.0, bend=0.0, via=5.0, heat=0.0, congestion=0.0)
naive_router = MultiNetRouter(grid, naive_weights)
t0 = time.time()
naive_result = naive_router.route_all(nets)
naive_time = time.time() - t0
naive_grid = grid
print(f"  Naive: {len(naive_result.paths)}/{len(nets)} routed in {naive_time:.3f}s")

# GA optimization
print("\n[2/6] Running GA optimization...")
gen_log = []

def on_gen(gen, best, pop):
    avg = sum(g.fitness for g in pop) / len(pop)
    gen_log.append({
        'generation': gen,
        'best_fitness': best.fitness,
        'avg_fitness': avg,
    })
    if gen % 5 == 0 or gen == 1:
        print(f"  Gen {gen:3d} | Best: {best.fitness:8.1f} | Avg: {avg:8.1f}")

config = GAConfig()
evaluator = FitnessEvaluator(base_grid, nets)
ga = GeneticAlgorithm(evaluator, config, on_generation=on_gen)
t0 = time.time()
best_genome = ga.run()
ga_time = time.time() - t0

ga_grid = base_grid.clone()
ga_router = MultiNetRouter(ga_grid, best_genome.to_cost_weights())
ga_result = ga_router.route_all(nets, net_order=best_genome.net_order)
print(f"  GA: {len(ga_result.paths)}/{len(nets)} routed in {ga_time:.1f}s ({len(gen_log)} gens)")


# ------------------------------------------------------------------ #
# Generate images
# ------------------------------------------------------------------ #

# 1. Naive routing board
print("\n[3/6] Generating naive_routing.png...")
viz_naive = BoardVisualizer(naive_grid)
fig = viz_naive.render(title="Naive Routing", show_heat=True,
                       result=naive_result, headless=True)
save(fig, "naive_routing.png")

# 2. GA-optimized routing board
print("\n[4/6] Generating ga_routing.png...")
viz_ga = BoardVisualizer(ga_grid)
fig = viz_ga.render(title="GA-Optimized Routing", show_heat=True,
                    result=ga_result, headless=True)
save(fig, "ga_routing.png")

# 3. Side-by-side comparison
print("\n[5/6] Generating comparison.png...")
viz_comp = BoardVisualizer(naive_grid)
fig = viz_comp.render_comparison(
    naive_grid, naive_result,
    ga_grid, ga_result,
    title_a="Naive Routing", title_b="GA-Optimized",
    headless=True,
)
save(fig, "comparison.png")

# 4. GA convergence chart
print("\n[6/6] Generating ga_convergence.png...")
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0E1117')
ax.set_facecolor('#1A1D23')

gens = [e['generation'] for e in gen_log]
best_fit = [e['best_fitness'] for e in gen_log]
avg_fit = [e['avg_fitness'] for e in gen_log]

ax.plot(gens, best_fit, color='#22c55e', linewidth=2, label='Best Fitness')
ax.fill_between(gens, best_fit, alpha=0.15, color='#22c55e')
ax.plot(gens, avg_fit, color='#f97316', linewidth=1.5, alpha=0.8,
        linestyle='--', label='Avg Fitness')

ax.set_title('GA Fitness Convergence', fontsize=14, fontweight='bold', color='white')
ax.set_xlabel('Generation', fontsize=11, color='#cccccc')
ax.set_ylabel('Fitness (lower = better)', fontsize=11, color='#cccccc')
ax.legend(fontsize=10, facecolor='#1A1D23', edgecolor='#444',
          labelcolor='white')
ax.tick_params(colors='#999999')
for spine in ax.spines.values():
    spine.set_color('#333333')
ax.grid(True, alpha=0.15, color='#555555')

save(fig, "ga_convergence.png")


# ------------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------------ #

print(f"\n{'=' * 52}")
print(f"  All images saved to: {os.path.abspath(SAMPLES_DIR)}")
print(f"{'=' * 52}")
print(f"  naive_routing.png   — Board before optimization")
print(f"  ga_routing.png      — Board after GA optimization")
print(f"  comparison.png      — Side-by-side comparison")
print(f"  ga_convergence.png  — GA fitness convergence chart")
print(f"\nMetrics:")
print(f"  Naive:  {len(naive_result.paths)}/{len(nets)} nets, "
      f"{naive_result.total_vias} vias, "
      f"{naive_result.total_trace_length} trace length")
print(f"  GA:     {len(ga_result.paths)}/{len(nets)} nets, "
      f"{ga_result.total_vias} vias, "
      f"{ga_result.total_trace_length} trace length")
if naive_result.total_vias > 0:
    via_red = (1 - ga_result.total_vias / naive_result.total_vias) * 100
    print(f"  Via reduction: {via_red:.0f}%")
