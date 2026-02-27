"""
FormaCore AI - visualize/plot.py
Matplotlib-based PCB board renderer.
Shows traces (red=top, blue=bottom), vias (green),
components (gray), heat overlay, and routing stats.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.grid import Grid
    from router.multi_router import RoutingResult

# Colors
COLOR_TOP = '#DD3333'
COLOR_BOT = '#3333DD'
COLOR_VIA = '#22BB22'
COLOR_COMP = '#999999'
HEAT_CMAP = 'YlOrRd'


class BoardVisualizer:
    """Renders a routed PCB board using matplotlib."""

    def __init__(self, grid: 'Grid'):
        self.grid = grid

    def render(self, title: str = "FormaCore PCB Router",
               show_heat: bool = True,
               result: Optional['RoutingResult'] = None,
               save_path: Optional[str] = None,
               headless: bool = False) -> Optional[object]:
        """
        Render the board with traces on both layers side by side.
        If headless=True, returns the figure without calling plt.show().
        """
        import matplotlib
        if headless:
            matplotlib.use('Agg')

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        layer_names = ['Top Layer (Front Cu)', 'Bottom Layer (Back Cu)']
        layer_colors = [COLOR_TOP, COLOR_BOT]

        for layer_idx, ax in enumerate(axes):
            ax.set_title(layer_names[layer_idx], fontsize=11)
            ax.set_xlim(0, self.grid.width)
            ax.set_ylim(self.grid.height, 0)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.08, linewidth=0.3)

            # Heat map
            if show_heat:
                heat = self.grid.heat[layer_idx]
                if np.max(heat) > 0:
                    ax.imshow(heat, cmap=HEAT_CMAP, alpha=0.3,
                              extent=[0, self.grid.width,
                                      self.grid.height, 0],
                              interpolation='bilinear')

            # Components
            for comp in self.grid.components:
                if comp.layer == layer_idx:
                    rect = patches.Rectangle(
                        (comp.x, comp.y), comp.width, comp.height,
                        linewidth=1.5, edgecolor='black',
                        facecolor=COLOR_COMP, alpha=0.7)
                    ax.add_patch(rect)
                    ax.text(comp.x + comp.width / 2,
                            comp.y + comp.height / 2,
                            comp.name, ha='center', va='center',
                            fontsize=7, fontweight='bold', color='white')

                    # Draw pins
                    for dx, dy in comp.pins:
                        px, py = comp.x + dx, comp.y + dy
                        ax.plot(px + 0.5, py + 0.5, 's',
                                color='yellow', markersize=3,
                                markeredgecolor='black',
                                markeredgewidth=0.3)

            # Traces
            for net_name, path in self.grid.routed_paths.items():
                self._draw_trace(ax, path, layer_idx,
                                 layer_colors[layer_idx])

            # Vias
            for net_name, path in self.grid.routed_paths.items():
                self._draw_vias(ax, path)

            ax.set_xlabel('X (grid cells)', fontsize=9)
            ax.set_ylabel('Y (grid cells)', fontsize=9)

        # Stats text
        if result:
            stats = (
                f"Nets: {len(result.paths)}/{len(result.paths)+len(result.failed_nets)}  "
                f"Length: {result.total_trace_length}  "
                f"Vias: {result.total_vias}  "
                f"Bends: {result.total_bends}  "
                f"Congestion: {result.congestion_score:.0f}"
            )
            fig.text(0.5, 0.01, stats, ha='center', fontsize=9,
                     style='italic', color='#555555')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        if headless:
            return fig
        plt.show()
        return None

    def render_comparison(self,
                          grid_a: 'Grid', result_a: 'RoutingResult',
                          grid_b: 'Grid', result_b: 'RoutingResult',
                          title_a: str = "Naive Routing",
                          title_b: str = "GA-Optimized",
                          save_path: Optional[str] = None,
                          headless: bool = False) -> Optional[object]:
        """Side-by-side comparison of two routing solutions (top layer)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("FormaCore AI — Routing Comparison",
                     fontsize=14, fontweight='bold')

        for ax, grid, result, title, color in [
            (ax1, grid_a, result_a, title_a, COLOR_TOP),
            (ax2, grid_b, result_b, title_b, COLOR_TOP),
        ]:
            ax.set_title(title, fontsize=11)
            ax.set_xlim(0, grid.width)
            ax.set_ylim(grid.height, 0)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.08)

            # Components
            for comp in grid.components:
                rect = patches.Rectangle(
                    (comp.x, comp.y), comp.width, comp.height,
                    linewidth=1.2, edgecolor='black',
                    facecolor=COLOR_COMP, alpha=0.6)
                ax.add_patch(rect)
                ax.text(comp.x + comp.width / 2,
                        comp.y + comp.height / 2,
                        comp.name, ha='center', va='center',
                        fontsize=6, fontweight='bold', color='white')

            # Traces (both layers)
            for net_name, path in grid.routed_paths.items():
                self._draw_trace(ax, path, 0, COLOR_TOP)
                self._draw_trace(ax, path, 1, COLOR_BOT)
                self._draw_vias(ax, path)

            # Stats below title
            stats = (
                f"Len: {result.total_trace_length}  "
                f"Vias: {result.total_vias}  "
                f"Bends: {result.total_bends}"
            )
            ax.set_xlabel(stats, fontsize=9, style='italic')

        # Improvement summary
        if result_a.total_trace_length > 0:
            n_a = max(len(result_a.paths), 1)
            n_b = max(len(result_b.paths), 1)
            avg_a = result_a.total_trace_length / n_a
            avg_b = result_b.total_trace_length / n_b
            avg_imp = (avg_a - avg_b) / avg_a * 100
            via_imp = ((result_a.total_vias - result_b.total_vias)
                       / max(result_a.total_vias, 1) * 100)
            comp_a = f"{n_a}/{n_a + len(result_a.failed_nets)}"
            comp_b = f"{n_b}/{n_b + len(result_b.failed_nets)}"
            summary = (f"Completion: {comp_a} vs {comp_b}  |  "
                       f"Avg length/net: {avg_imp:+.1f}%  |  "
                       f"Via reduction: {via_imp:.1f}%")
            fig.text(0.5, 0.01, summary, ha='center', fontsize=10,
                     fontweight='bold', color='#227722')

        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        if headless:
            return fig
        plt.show()
        return None

    # -- Drawing helpers --

    def _draw_trace(self, ax, path: List[Tuple[int, int, int]],
                    target_layer: int, color: str) -> None:
        """Draw trace segments on a specific layer."""
        segments: List[List[Tuple[float, float]]] = []
        current: List[Tuple[float, float]] = []

        for (x, y, layer) in path:
            if layer == target_layer:
                current.append((x + 0.5, y + 0.5))
            else:
                if len(current) >= 2:
                    segments.append(current)
                current = []

        if len(current) >= 2:
            segments.append(current)

        for seg in segments:
            xs = [p[0] for p in seg]
            ys = [p[1] for p in seg]
            ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.8)

    def _draw_vias(self, ax, path: List[Tuple[int, int, int]]) -> None:
        """Draw via markers at layer transitions."""
        for i in range(1, len(path)):
            if path[i][2] != path[i - 1][2]:
                x = path[i][0] + 0.5
                y = path[i][1] + 0.5
                ax.plot(x, y, 'o', color=COLOR_VIA,
                        markersize=4, markeredgecolor='black',
                        markeredgewidth=0.5)
