"""Commitment map: 32B-Think vs 8B-NoThink, vertical layout for wrapfigure."""

import json
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import paths
import plot_style

plot_style.apply_style()
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.grid': False})

COL_GENUINE    = "#3182bd"
COL_POSTCOMMIT = "#fdae6b"
COL_WRONG      = "#d9d9d9"
COL_BOUNDARY   = "#222222"

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
    solvable, unsolvable = [], []
    for r in results:
        cf = r.get('commitment_fraction')
        if cf is None:
            cf = 1.0
        if r['selected_rollout_correct']:
            solvable.append(cf)
        else:
            unsolvable.append(cf)

    solvable.sort()
    n_s, n_u = len(solvable), len(unsolvable)
    n_total = n_s + n_u

    for i, cf in enumerate(solvable):
        ax.barh(i, cf, height=1.0, color=COL_GENUINE, linewidth=0)
        ax.barh(i, 1.0 - cf, height=1.0, color=COL_POSTCOMMIT, linewidth=0, left=cf)

    for i in range(n_u):
        ax.barh(n_s + i, 1.0, height=1.0, color=COL_WRONG, linewidth=0)

    ax.plot(solvable, range(n_s), color=COL_BOUNDARY, linewidth=1.8, alpha=0.85)
    ax.axhline(n_s - 0.5, color='#aa3333', linewidth=0.8, alpha=0.4)

    pc_frac = np.mean([1 - cf for cf in solvable])
    ax.annotate(f'Post-commit\n{pc_frac:.0%}',
                xy=(0.78, n_s * 0.22), fontsize=13, fontweight='bold',
                color='#7f4500', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#e6a756',
                          alpha=0.90, linewidth=1.0))
    ax.annotate('Pre-commit',
                xy=(0.14, n_s * 0.62), fontsize=12, fontweight='bold',
                color='#1a4d7a', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#6baed6',
                          alpha=0.90, linewidth=1.0))

    if n_u > 5:
        ax.text(0.50, n_s + n_u * 0.5, f'Unsolvable ({n_u})',
                fontsize=10, color='#666666', ha='center', va='center',
                fontstyle='italic')

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, n_total - 0.5)
    ax.set_xlabel('Fraction of CoT')
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.set_title(title, fontsize=14, pad=8)


def fig_commitment_map():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8.5),
                                    gridspec_kw={'hspace': 0.30})

    draw_panel(ax1, ALL_DATA["32B-Think"], "(a)  32B-Think")
    ax1.set_ylabel('Problems (sorted by commitment)', fontsize=12)

    draw_panel(ax2, ALL_DATA["8B-NoThink"], "(b)  8B-NoThink")
    ax2.set_ylabel('Problems (sorted by commitment)', fontsize=12)

    # Shared legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor=COL_GENUINE, ec='#888', label='Pre-commitment'),
        mpatches.Patch(facecolor=COL_POSTCOMMIT, ec='#888', label='Post-commitment'),
        mpatches.Patch(facecolor=COL_WRONG, ec='#888', label='Unsolvable'),
        Line2D([0], [0], color=COL_BOUNDARY, lw=1.5, label='Commitment boundary'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10.5, frameon=True, edgecolor='#cccccc',
               bbox_to_anchor=(0.5, -0.02))

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig_theater_map.{ext}'))
    plt.close(fig)
    print('Saved fig_theater_map')


if __name__ == '__main__':
    fig_commitment_map()
