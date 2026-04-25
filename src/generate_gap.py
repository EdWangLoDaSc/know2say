"""Redesigned detection-extraction gap figure for wrapfigure placement."""

import json
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter

import paths
import plot_style

# Larger text for wrapfigure placement
plot_style.apply_style(large=True)
plt.rcParams.update({'font.size': 18, 'axes.labelsize': 20, 'axes.titlesize': 22,
                     'xtick.labelsize': 18, 'ytick.labelsize': 18,
                     'legend.fontsize': 15, 'grid.linewidth': 0.6})

from plot_style import PAL, MARKERS, LS  # noqa: E402

PATHS = {
    "32B-Think":    paths.paper_data("qwen3_32b_thinking_full500"),
    "32B-NoThink":  paths.paper_data("qwen3_32b_no_thinking_full500"),
    "8B-Think":     paths.paper_data("qwen3_8b_thinking_full500"),
    "8B-NoThink":   paths.paper_data("qwen3_8b_no_thinking_full500"),
    "GPT-OSS-120B": paths.paper_data("gpt_oss_120b_full500"),
}
OUTPUT_DIR = paths.FIGURES_DIR


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


ALL_DATA = {name: load(path) for name, path in PATHS.items()}


def get_curves(results):
    fracs = sorted(set(pr['fraction'] for r in results for pr in r['prefix_results']))
    efa_acc, psc_acc = [], []
    for f in fracs:
        efa = [pr['efa_correct'] for r in results for pr in r['prefix_results']
               if abs(pr['fraction'] - f) < 0.01]
        psc = [pr['psc_agreement_rate'] for r in results for pr in r['prefix_results']
               if abs(pr['fraction'] - f) < 0.01]
        efa_acc.append(np.mean(efa))
        psc_acc.append(np.mean(psc))
    return fracs, efa_acc, psc_acc


def fig_gap():
    """Detection-extraction gap figure.

    Panel (a): EFA failure-mode breakdown for the 208 gap instances
    (32B-Think, f=0.10, PSC-high but EFA-wrong).  Horizontal stacked bar
    makes the *why* of the gap immediately clear — very different visually
    from every other figure in the paper.

    Panel (b): gap magnitude trajectories across all five models.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 10.5),
                                    gridspec_kw={'hspace': 0.55})

    # ── Panel (a): EFA failure-mode breakdown ────────────────────────────────
    # Hard-coded from the mechanistic analysis (Appendix §app:gap_mechanism)
    # 208 gap instances: 32B-Think, f=0.10, PSC≥75% but EFA wrong
    failure_labels = [
        'Premature\ntermination',
        'Intermediate\nvalue',
        'Sign / parity\nerror',
    ]
    failure_pcts   = [0.59, 0.30, 0.11]   # 59 % / 30 % / 11 %
    failure_colors = ['#d62728', '#ff7f0e', '#f7b731']
    failure_desc   = [
        '≤2-char output\n("answer reflex")',
        'Partial result\nemitted early',
        'Correct magnitude,\nwrong sign/index',
    ]

    y_pos = [2, 1, 0]
    bar_height = 0.55

    bars = ax1.barh(y_pos, failure_pcts, height=bar_height,
                    color=failure_colors, edgecolor='white', linewidth=1.2,
                    zorder=3)

    # Percentage labels inside bars
    for bar, pct in zip(bars, failure_pcts):
        ax1.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                 f'{pct:.0%}',
                 ha='center', va='center',
                 fontsize=20, fontweight='bold', color='white', zorder=4)

    # Short description to the right of each bar
    for i, (desc, pct) in enumerate(zip(failure_desc, failure_pcts)):
        ax1.text(pct + 0.015, y_pos[i],
                 desc,
                 ha='left', va='center', fontsize=13, color='#333333',
                 linespacing=1.35)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(failure_labels, fontsize=17, fontweight='bold')
    ax1.set_xlabel('Fraction of EFA failures  (n = 208)', fontsize=19, fontweight='bold')
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_xlim(0, 0.88)
    ax1.set_ylim(-0.5, 2.8)
    ax1.set_title('(a) Why EFA fails when PSC succeeds\n'
                  '32B-Think · f = 10% · 208 gap instances',
                  pad=10, fontsize=19, fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', alpha=0.2, linewidth=0.5, zorder=0)
    ax1.set_axisbelow(True)

    # ── Panel (b): Gap magnitude across all models ────────────────────────────
    ax2.fill_between([0.05, 0.95], 0, 0.65,
                     color='#fff0ee', alpha=0.55, zorder=0)
    ax2.axhline(0, color='#aaaaaa', linewidth=0.8)

    for name, results in ALL_DATA.items():
        fracs, efa, psc = get_curves(results)
        gaps = [p - e for p, e in zip(psc, efa)]
        ax2.plot(fracs, gaps, color=PAL[name], marker=MARKERS[name],
                 ls=LS[name], markersize=5.5, linewidth=2.0, label=name)

    ax2.set_xlabel('Prefix fraction of CoT', fontsize=19, fontweight='bold')
    ax2.set_ylabel('PSC $-$ EFA  (gap)', fontsize=19, fontweight='bold')
    ax2.set_title('(b) Gap magnitude across all models', pad=10,
                  fontsize=19, fontweight='bold', loc='left')
    ax2.legend(fontsize=14, loc='upper right', framealpha=0.95, ncol=1)
    ax2.set_ylim(-0.05, 0.75)
    ax2.set_xlim(0.05, 0.95)
    ax2.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax2.set_xticklabels(['10%', '30%', '50%', '70%', '90%'], fontsize=18, fontweight='bold')
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.grid(alpha=0.15, linewidth=0.5)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig1_gap.{ext}'))
    plt.close(fig)
    print('Saved fig1_gap')


if __name__ == '__main__':
    fig_gap()
