"""Redesigned theater map: 32B-Think vs 8B-NoThink side by side."""

import json
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import paths
import plot_style

plot_style.apply_style()
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.grid': False})

COL_GENUINE  = "#3182bd"  # rich blue
COL_THEATER  = "#fdae6b"  # warm amber
COL_WRONG    = "#d9d9d9"  # soft gray
COL_BOUNDARY = "#222222"

PATHS = {
    "32B-Think":  paths.paper_data("qwen3_32b_thinking_full500"),
    "8B-NoThink": paths.paper_data("qwen3_8b_no_thinking_full500"),
}
OUTPUT_DIR = paths.FIGURES_DIR


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


ALL_DATA = {name: load(path) for name, path in PATHS.items()}


def draw_panel(ax, results, title):
    solvable = []
    unsolvable = []
    for r in results:
        cf = r.get('commitment_fraction')
        correct = r['selected_rollout_correct']
        if cf is None:
            cf = 1.0
        if correct:
            solvable.append(cf)
        else:
            unsolvable.append(cf)

    solvable.sort()
    n_s = len(solvable)
    n_u = len(unsolvable)
    n_total = n_s + n_u

    # Draw solvable: blue genuine + amber theater
    for i, cf in enumerate(solvable):
        ax.barh(i, cf, height=1.0, color=COL_GENUINE, linewidth=0)
        ax.barh(i, 1.0 - cf, height=1.0, color=COL_THEATER, linewidth=0, left=cf)

    # Draw unsolvable: gray
    for i in range(n_u):
        ax.barh(n_s + i, 1.0, height=1.0, color=COL_WRONG, linewidth=0)

    # Commitment boundary
    ax.plot(solvable, range(n_s), color=COL_BOUNDARY, linewidth=1.5, alpha=0.85)

    # Separator
    ax.axhline(n_s - 0.5, color='#aa3333', linewidth=0.9, alpha=0.5)

    # Theater fraction annotation
    theater_frac = np.mean([1 - cf for cf in solvable])
    ax.annotate(f'Theater {theater_frac:.0%}',
                xy=(0.78, n_s * 0.22), fontsize=12, fontweight='bold',
                color='#7f4500', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#e6a756',
                          alpha=0.88, linewidth=1.0))
    ax.annotate('Genuine',
                xy=(0.13, n_s * 0.62), fontsize=11, fontweight='bold',
                color='#1a4d7a', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#6baed6',
                          alpha=0.88, linewidth=1.0))

    # Unsolvable label
    if n_u > 5:
        ax.text(0.50, n_s + n_u * 0.5, f'Unsolvable ({n_u})',
                fontsize=9.5, color='#666666', ha='center', va='center',
                fontstyle='italic')

    # Count annotations on right edge
    ax.text(1.03, n_s * 0.5, f'{n_s}', transform=ax.get_yaxis_transform(),
            fontsize=10, color='#1a4d7a', va='center', ha='left', fontweight='bold')
    if n_u > 0:
        ax.text(1.03, n_s + n_u * 0.5, f'{n_u}', transform=ax.get_yaxis_transform(),
                fontsize=10, color='#888888', va='center', ha='left', fontweight='bold')

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, n_total - 0.5)
    ax.set_xlabel('Fraction of CoT')
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.set_title(title, fontsize=14, pad=8)


def fig_theater_map():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.8), sharey=False)

    draw_panel(ax1, ALL_DATA["32B-Think"], "32B-Think")
    ax1.set_ylabel('Problems (sorted by commitment)')

    draw_panel(ax2, ALL_DATA["8B-NoThink"], "8B-NoThink")

    # Shared legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor=COL_GENUINE, ec='#888', label='Genuine reasoning'),
        mpatches.Patch(facecolor=COL_THEATER, ec='#888', label='Post-commitment (theater)'),
        mpatches.Patch(facecolor=COL_WRONG, ec='#888', label='Unsolvable'),
        Line2D([0], [0], color=COL_BOUNDARY, lw=1.5, label='Commitment boundary'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, frameon=True, edgecolor='#cccccc',
               bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig_theater_map.{ext}'))
    plt.close(fig)
    print('Saved fig_theater_map')


if __name__ == '__main__':
    fig_theater_map()
