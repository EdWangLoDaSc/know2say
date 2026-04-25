"""Shared matplotlib style for all Reasoning Theater figures.

Usage:
    import plot_style
    plot_style.apply_style()            # standard figures
    plot_style.apply_style(large=True)  # hero / wrapfigure (larger fonts)

Exports:
    PAL, MARKERS, LS          — per-model color / marker / linestyle
    COL_GENUINE, COL_THEATER,
    COL_WRONG, COL_BOUNDARY   — theater-map category colors
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ── Per-model visual encoding ─────────────────────────────────────────────────
# ColorBrewer-derived palette: colorblind-safe, print-friendly

PAL: dict[str, str] = {
    "32B-Think":    "#2171b5",  # strong blue
    "32B-NoThink":  "#cb181d",  # rich red
    "8B-Think":     "#6a51a3",  # purple
    "8B-NoThink":   "#d94801",  # dark orange
    "GPT-OSS-120B": "#238b45",  # forest green
}

MARKERS: dict[str, str] = {
    "32B-Think":    "o",
    "32B-NoThink":  "s",
    "8B-Think":     "^",
    "8B-NoThink":   "D",
    "GPT-OSS-120B": "P",
}

LS: dict[str, str] = {
    "32B-Think":    "-",
    "32B-NoThink":  "--",
    "8B-Think":     "-",
    "8B-NoThink":   "--",
    "GPT-OSS-120B": "-.",
}

# ── Theater-map category colors ───────────────────────────────────────────────

COL_GENUINE  = "#6baed6"  # cornflower blue  — committed & correct
COL_THEATER  = "#fd8d3c"  # warm orange      — theater (committed late)
COL_WRONG    = "#bdbdbd"  # neutral gray     — incorrect
COL_BOUNDARY = "#252525"  # near-black       — boundary lines

# ── rcParams presets ──────────────────────────────────────────────────────────

_BASE: dict = {
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.titleweight":    "bold",
    "axes.linewidth":      0.8,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.color":          "#e0e0e0",
    "grid.linewidth":      0.5,
    "grid.alpha":          1.0,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "#cccccc",
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.1,
}

_STANDARD: dict = {
    **_BASE,
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "legend.fontsize":   8,
}

_LARGE: dict = {
    **_BASE,
    "font.size":          17,
    "axes.labelsize":     18,
    "axes.titlesize":     19,
    "axes.linewidth":     1.2,
    "xtick.labelsize":    16,
    "ytick.labelsize":    16,
    "xtick.major.size":   5.0,
    "ytick.major.size":   5.0,
    "xtick.major.width":  1.2,
    "ytick.major.width":  1.2,
    "legend.fontsize":    14,
    "savefig.pad_inches": 0.08,
}


def apply_style(large: bool = False) -> None:
    """Apply publication rcParams.

    Args:
        large: Use larger fonts for hero / wrapfigure placements.
    """
    plt.rcParams.update(_LARGE if large else _STANDARD)
