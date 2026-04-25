"""fig2_main v3: 1x4 with data-driven annotations, larger font."""

import json, os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import paths
import plot_style

plot_style.apply_style()
# This figure uses slightly larger fonts than the standard preset
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11,
                     'savefig.pad_inches': 0.08})

from plot_style import PAL, MARKERS, LS  # noqa: E402

OUT = paths.FIGURES_DIR


def load(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]


def _style(n): return dict(color=PAL[n], marker=MARKERS[n], ls=LS[n], markersize=4.5, linewidth=1.8)


MATH = {n: load(paths.paper_data(p)) for n, p in [
    ("32B-Think", "qwen3_32b_thinking_full500"), ("32B-NoThink", "qwen3_32b_no_thinking_full500"),
    ("8B-Think", "qwen3_8b_thinking_full500"), ("8B-NoThink", "qwen3_8b_no_thinking_full500"),
    ("GPT-OSS-120B", "gpt_oss_120b_full500")]}


def fig2_main():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    fig.subplots_adjust(wspace=0.40, left=0.04, right=0.98, top=0.84, bottom=0.19)

    # ── (a) EFA accuracy ──
    ax = axes[0]
    for name, results in MATH.items():
        fracs = sorted(set(pr['fraction'] for r in results for pr in r['prefix_results']))
        efa = [np.mean([pr['efa_correct'] for r in results for pr in r['prefix_results']
               if abs(pr['fraction'] - f) < 0.01]) for f in fracs]
        ax.plot(fracs, efa, **_style(name), label=name)
    ax.set_xlabel('Prefix fraction')
    ax.set_ylabel('EFA accuracy')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('(a) Early Forced Answering', fontsize=12)
    ax.set_ylim(0, 0.85)
    ax.set_xlim(0.05, 0.95)
    ax.legend(fontsize=8.5, loc='lower right', ncol=1, handletextpad=0.4,
              columnspacing=0.5, borderpad=0.35)
    # Annotation: EFA plateaus at ~70-78%
    ax.annotate('plateaus\n$\\leq$78%',
                xy=(0.80, 0.75), fontsize=9, ha='center', color='#555',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#999', alpha=0.9, lw=0.6))
    # Annotation: starts low
    ax.annotate('21--34%\nat 10%',
                xy=(0.10, 0.30), xytext=(0.25, 0.08),
                fontsize=8.5, color='#555', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#777', lw=1.0),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#999', alpha=0.9, lw=0.6))

    # ── (b) PSC agreement ──
    ax = axes[1]
    for name, results in MATH.items():
        fracs = sorted(set(pr['fraction'] for r in results for pr in r['prefix_results']))
        psc = [np.mean([pr['psc_agreement_rate'] for r in results for pr in r['prefix_results']
               if abs(pr['fraction'] - f) < 0.01]) for f in fracs]
        ax.plot(fracs, psc, **_style(name), label=name)
    ax.axhline(0.75, color='#999', ls=':', linewidth=0.8)
    ax.set_xlabel('Prefix fraction')
    ax.set_ylabel('PSC agreement')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('(b) Prefix Self-Consistency', fontsize=12)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(0.05, 0.95)
    # Annotation: PSC high from start
    ax.annotate('70--92%\nfrom 10%',
                xy=(0.10, 0.88), fontsize=9, ha='center', color=PAL["GPT-OSS-120B"],
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=PAL["GPT-OSS-120B"],
                          alpha=0.9, lw=0.6))
    # theta line label
    ax.text(0.92, 0.76, '$\\theta$=0.75', fontsize=8, color='#999', ha='right')

    # ── (c) Commitment distribution ──
    ax = axes[2]
    names_list = list(MATH.keys())
    data_list = []
    for name in names_list:
        cfs = [r['commitment_fraction'] for r in MATH[name]
               if r['commitment_fraction'] is not None]
        data_list.append(cfs)

    bp = ax.boxplot(data_list, vert=False, patch_artist=True, widths=0.55,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    flierprops=dict(markersize=1.5, alpha=0.2))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(PAL[names_list[i]])
        patch.set_alpha(0.45)
    for i, (name, cfs) in enumerate(zip(names_list, data_list)):
        jitter = np.random.default_rng(42).normal(0, 0.08, len(cfs))
        ax.scatter(cfs, np.full(len(cfs), i + 1) + jitter, s=2, alpha=0.12,
                   color=PAL[name], zorder=0)

    ax.set_yticks(range(1, len(names_list) + 1))
    ax.set_yticklabels(names_list, fontsize=8)
    ax.set_xlabel('Commitment fraction')
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('(c) Where models recover', fontsize=12)
    ax.grid(False)
    # Annotation: Think early, NoThink late
    ax.annotate('Think:\nmedian 20--30%',
                xy=(0.22, 3.8), fontsize=8, color=PAL["32B-Think"], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=PAL["32B-Think"],
                          alpha=0.85, lw=0.6))
    ax.annotate('NoThink:\nmedian 20--50%',
                xy=(0.55, 1.3), fontsize=8, color=PAL["8B-NoThink"], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=PAL["8B-NoThink"],
                          alpha=0.85, lw=0.6))

    # ── (d) Commitment by difficulty ──
    ax = axes[3]
    levels = [1, 2, 3, 4, 5]
    for name, results in MATH.items():
        by_level = defaultdict(list)
        for r in results:
            cf = r.get('commitment_fraction')
            if cf is not None:
                by_level[r['level']].append(cf)
        means = [np.mean(by_level[l]) if l in by_level else 0 for l in levels]
        ax.plot(levels, means, **_style(name), label=name)

    ax.set_xlabel('Difficulty level')
    ax.set_ylabel('Mean commitment')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('(d) Harder $\\to$ later recovery', fontsize=12)
    ax.set_xticks(levels)
    ax.set_xticklabels(['L1', 'L2', 'L3', 'L4', 'L5'], fontsize=9)
    # Annotation: monotonic trend
    ax.annotate('L1: 13--32%\nL5: 32--63%',
                xy=(3.5, 0.20), fontsize=8.5, color='#555', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#999',
                          alpha=0.9, lw=0.6))

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT, f'fig2_main.{ext}'))
    plt.close(fig)
    print('Saved fig2_main')


if __name__ == '__main__':
    fig2_main()
