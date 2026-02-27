"""
FormaCore AI - main.py
Entry point. Creates a sample board, runs naive vs GA-optimized routing,
prints metrics, and shows side-by-side visualization.
"""
from __future__ import annotations

import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.grid import Grid, Component, Net, Layer
from core.heat import HeatModel
from core.congestion import CongestionTracker
from router.cost import CostWeights
from router.multi_router import MultiNetRouter, RoutingResult
from ai.genome import Genome
from ai.fitness import FitnessEvaluator, FitnessWeights
from ai.ga import GeneticAlgorithm, GAConfig
from visualize.plot import BoardVisualizer


# ------------------------------------------------------------------ #
# SAMPLE BOARD DEFINITION
# ------------------------------------------------------------------ #

def create_sample_board() -> tuple:
    """
    Create a sample 2-layer PCB with dense, conflicting routing.
    Board: 80 x 60 grid cells (40mm x 30mm at 0.5mm resolution).
    10 components, 15 nets with heavy crossing.
    """
    grid = Grid(width=80, height=60, layers=2, resolution_mm=0.5)

    components = [
        # Central MCU — many pins, high connectivity
        Component("MCU", x=25, y=20, width=12, height=12,
                  layer=Layer.TOP, power_w=0.8,
                  pins=[(0, 0), (11, 0), (0, 11), (11, 11),
                        (5, 0), (6, 0), (0, 5), (0, 6),
                        (11, 5), (11, 6), (5, 11), (6, 11)]),

        # Power regulator — hot, upper-left
        Component("REG", x=5, y=5, width=6, height=4,
                  layer=Layer.TOP, power_w=3.0,
                  pins=[(0, 0), (5, 0), (0, 3), (5, 3)]),

        # Flash — right side
        Component("FLASH", x=55, y=8, width=8, height=6,
                  layer=Layer.TOP, power_w=0.3,
                  pins=[(0, 0), (7, 0), (0, 5), (7, 5)]),

        # Connector — bottom right, opposite layer
        Component("CONN", x=58, y=42, width=10, height=8,
                  layer=Layer.BOTTOM, power_w=0.0,
                  pins=[(0, 0), (9, 0), (0, 7), (9, 7),
                        (3, 0), (6, 0)]),

        # Decoupling caps — scattered near MCU
        Component("C1", x=20, y=12, width=3, height=2,
                  layer=Layer.TOP, power_w=0.0,
                  pins=[(0, 0), (2, 0)]),
        Component("C2", x=40, y=18, width=3, height=2,
                  layer=Layer.TOP, power_w=0.0,
                  pins=[(0, 0), (2, 0)]),
        Component("C3", x=20, y=38, width=3, height=2,
                  layer=Layer.TOP, power_w=0.0,
                  pins=[(0, 0), (2, 0)]),

        # Resistors
        Component("R1", x=50, y=28, width=2, height=4,
                  layer=Layer.TOP, power_w=0.05,
                  pins=[(0, 0), (0, 3)]),
        Component("R2", x=10, y=42, width=2, height=4,
                  layer=Layer.TOP, power_w=0.05,
                  pins=[(0, 0), (0, 3)]),

        # LED — far upper right
        Component("LED", x=70, y=5, width=3, height=3,
                  layer=Layer.TOP, power_w=0.04,
                  pins=[(0, 0), (2, 0)]),
    ]

    for comp in components:
        grid.place_component(comp)

    # Nets designed to create crossing conflicts.
    # Many routes must cross each other's natural shortest path.
    nets = [
        # Power: REG → MCU (horizontal-ish)
        Net("VCC",       [(10, 5, 0), (25, 20, 0)]),
        Net("GND",       [(5, 8, 0),  (25, 31, 0)]),

        # MCU → FLASH: must cross vertically
        Net("SPI_CLK",   [(36, 20, 0), (55, 8, 0)]),
        Net("SPI_MOSI",  [(36, 25, 0), (55, 13, 0)]),
        Net("SPI_MISO",  [(36, 26, 0), (62, 8, 0)]),
        Net("SPI_CS",    [(36, 31, 0), (62, 13, 0)]),

        # MCU → CONN: long diagonal, cross-layer
        Net("TX",        [(25, 26, 0), (58, 42, 1)]),
        Net("RX",        [(30, 31, 0), (67, 42, 1)]),
        Net("DATA2",     [(31, 31, 0), (61, 42, 1)]),

        # Decoupling: short but routing-blocking
        Net("DECOUP1",   [(20, 12, 0), (25, 25, 0)]),
        Net("DECOUP2",   [(42, 18, 0), (36, 20, 0)]),
        Net("DECOUP3",   [(20, 38, 0), (25, 31, 0)]),

        # LED: crosses MCU→FLASH paths
        Net("LED_SIG",   [(36, 25, 0), (50, 28, 0)]),
        Net("LED_R",     [(50, 31, 0), (70, 5, 0)]),

        # Cross-board: bottom-left to upper-right (max conflict)
        Net("RESET",     [(10, 45, 0), (70, 5, 0)]),
    ]

    return grid, nets


# ------------------------------------------------------------------ #
# ROUTING ROUTINES
# ------------------------------------------------------------------ #

def run_naive(grid: Grid, nets: list) -> RoutingResult:
    """Route with default order and minimal weights (truly naive)."""
    naive_weights = CostWeights(step=1.0, bend=0.0, via=5.0,
                                heat=0.0, congestion=0.0)
    router = MultiNetRouter(grid, naive_weights)
    return router.route_all(nets)


def run_ga(base_grid: Grid, nets: list,
           config: GAConfig = None) -> tuple:
    """Run GA optimization, return (result, best_genome, final_grid)."""
    config = config or GAConfig()  # uses new defaults: pop=15, gen=20, early_stop=5

    evaluator = FitnessEvaluator(base_grid, nets)

    def on_gen(gen, best, pop):
        avg = sum(g.fitness for g in pop) / len(pop)
        print(f"  Gen {gen:3d}  |  Best: {best.fitness:8.1f}  "
              f"|  Avg: {avg:8.1f}")

    ga = GeneticAlgorithm(evaluator, config, on_generation=on_gen)
    best = ga.run()

    # Final routing with best genome
    final_grid = base_grid.clone()
    router = MultiNetRouter(final_grid, best.to_cost_weights())
    result = router.route_all(nets, net_order=best.net_order)

    return result, best, final_grid


# ------------------------------------------------------------------ #
# OUTPUT
# ------------------------------------------------------------------ #

def print_result(label: str, result: RoutingResult) -> None:
    total = len(result.paths) + len(result.failed_nets)
    print(f"\n{'=' * 52}")
    print(f"  {label}")
    print(f"{'=' * 52}")
    print(f"  Completion:     {result.completion_rate:.0%} "
          f"({len(result.paths)}/{total})")
    print(f"  Trace length:   {result.total_trace_length}")
    print(f"  Vias:           {result.total_vias}")
    print(f"  Bends:          {result.total_bends}")
    print(f"  Congestion:     {result.congestion_score:.0f}")
    if result.failed_nets:
        print(f"  FAILED nets:    {result.failed_nets}")


def print_comparison(naive: RoutingResult, optimized: RoutingResult,
                     total_nets: int) -> None:
    print(f"\n{'=' * 52}")
    print(f"  COMPARISON")
    print(f"{'=' * 52}")

    n_naive = len(naive.paths)
    n_opt = len(optimized.paths)

    print(f"  Completion:         {n_naive}/{total_nets} vs "
          f"{n_opt}/{total_nets}")

    if naive.total_trace_length > 0:
        # Per-net average comparison (fair when net counts differ)
        avg_naive = naive.total_trace_length / max(n_naive, 1)
        avg_opt = optimized.total_trace_length / max(n_opt, 1)
        avg_imp = (avg_naive - avg_opt) / avg_naive * 100
        print(f"  Naive total length: {naive.total_trace_length} "
              f"(avg {avg_naive:.1f}/net)")
        print(f"  Opt total length:   {optimized.total_trace_length} "
              f"(avg {avg_opt:.1f}/net)")
        print(f"  Avg/net improvement:{avg_imp:+.1f}%")

    if naive.total_vias > 0:
        via_imp = ((naive.total_vias - optimized.total_vias)
                   / naive.total_vias * 100)
        print(f"  Via reduction:      {via_imp:.1f}% "
              f"({naive.total_vias} -> {optimized.total_vias})")

    bend_imp = naive.total_bends - optimized.total_bends
    print(f"  Bends:              {naive.total_bends} -> "
          f"{optimized.total_bends} ({bend_imp:+d})")

    if naive.failed_nets:
        print(f"  Naive FAILED:       {naive.failed_nets}")
    if optimized.failed_nets:
        print(f"  Optimized FAILED:   {optimized.failed_nets}")


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #

def main():
    print("=" * 52)
    print("  FormaCore AI — 2-Layer PCB Router")
    print("=" * 52)

    # 1. Create board
    grid, nets = create_sample_board()
    print(f"\nBoard: {grid.width}x{grid.height} grid "
          f"({grid.width * grid.resolution:.0f}mm x "
          f"{grid.height * grid.resolution:.0f}mm)")
    print(f"Components: {len(grid.components)}")
    print(f"Nets: {len(nets)}")

    # 2. Apply heat model
    heat = HeatModel(sigma=10.0)
    heat.apply(grid)
    print(f"Heat model applied (peak: {heat.get_max_heat(grid):.3f})")

    # 3. Save clean state before routing
    base_grid = grid.clone()

    # 4. Naive routing
    print("\n--- Naive Routing (default order + weights) ---")
    t0 = time.time()
    naive_result = run_naive(grid, nets)
    naive_time = time.time() - t0
    print_result("Naive Router", naive_result)
    print(f"  Time:           {naive_time:.3f}s")

    naive_grid = grid  # keep reference for visualization

    # 5. GA-optimized routing
    print("\n--- GA Optimization ---")
    t0 = time.time()
    ga_result, best_genome, ga_grid = run_ga(base_grid, nets)
    ga_time = time.time() - t0
    print_result("GA-Optimized Router", ga_result)
    print(f"  Time:           {ga_time:.1f}s")
    print(f"  Net order:      {best_genome.net_order}")
    print(f"  Weights:        d={best_genome.distance_w:.2f} "
          f"b={best_genome.bend_w:.2f} v={best_genome.via_w:.2f} "
          f"h={best_genome.heat_w:.2f} c={best_genome.congestion_w:.2f}")

    # 6. Comparison
    print_comparison(naive_result, ga_result, len(nets))

    # 7. Visualize
    print("\n--- Visualization ---")

    viz = BoardVisualizer(naive_grid)
    viz.render(title="Naive Routing", show_heat=True,
               result=naive_result,
               save_path="naive_routing.png")

    viz_ga = BoardVisualizer(ga_grid)
    viz_ga.render(title="GA-Optimized Routing", show_heat=True,
                  result=ga_result,
                  save_path="ga_routing.png")

    viz.render_comparison(
        naive_grid, naive_result,
        ga_grid, ga_result,
        title_a="Naive", title_b="GA-Optimized",
        save_path="comparison.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
