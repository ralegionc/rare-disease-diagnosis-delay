"""Chart styling: one palette, one set of rules, applied everywhere.

Colours are the validated categorical set (fixed slot order, never cycled),
with a single sequential hue for magnitude and a reserved status pair. Slot
order is the colour-vision-deficiency safety mechanism, so diseases are always
assigned in :data:`DISEASE_ORDER` -- a chart that repaints a disease when the
series count changes is a chart that cannot be compared with its neighbour.

Marks follow the same rules throughout: 2px lines, recessive hairline grid on
one axis only, no chart junk, direct labels in preference to a legend where
there is room, and a legend whenever there are two or more series so identity
is never carried by colour alone.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots, in fixed order.
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

#: Fixed presentation order: fastest to slowest odyssey. Ordering the panel by
#: the quantity under study makes every chart in the report read the same way.
DISEASE_ORDER = ["ALS", "GD1", "LOPD", "HAE", "hEDS", "FABRY"]
DISEASE_COLOR = {k: SERIES[i] for i, k in enumerate(DISEASE_ORDER)}

DISEASE_LABEL = {
    "ALS": "ALS",
    "GD1": "Gaucher type 1",
    "LOPD": "Late-onset Pompe",
    "HAE": "Hereditary angioedema",
    "hEDS": "Hypermobile EDS",
    "FABRY": "Fabry",
}

# Phase colours reuse the first four slots; phases never appear on the same
# chart as diseases, so there is no collision to resolve.
PHASE_COLOR = {0: SERIES[0], 1: SERIES[1], 2: SERIES[2], 3: SERIES[3]}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def finish(ax, title=None, subtitle=None, xlabel=None, ylabel=None, grid_axis="y"):
    """Apply the shared chrome rules to one axes."""
    ax.grid(axis=grid_axis, which="major")
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    if title:
        ax.set_title(title, loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(
            0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color=INK_2,
        )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def caption(fig, text: str, y: float = -0.02) -> None:
    """Footnote under the plot.

    ``y`` is in figure coordinates; push it down when the figure carries a
    legend below the axes, or the two will overlap.
    """
    fig.text(0.0, y, text, ha="left", va="top", fontsize=8.5, color=MUTED, wrap=True)


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)
