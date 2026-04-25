"""Generate HumanEval figures + update 4-benchmark bar chart."""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter

import paths
import plot_style

plot_style.apply_style()
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9})

from plot_style import PAL, MARKERS, LS, COL_GENUINE, COL_WRONG  # noqa: E402

COL_PRE = COL_GENUINE; COL_POST = "#fdae6b"

BASE = paths.PAPER_DIR
OUT = paths.FIGURES_DIR


def load(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]


def _style(n): return dict(color=PAL[n], marker=MARKERS[n], ls=LS[n], markersize=5, linewidth=1.8)


def _save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'))
    plt.close(fig); print(f'  Saved {name}')


HE = {
    "32B-Think":   load(paths.paper_data("humaneval_32b_think")),
    "32B-NoThink": load(paths.paper_data("humaneval_32b_nothink")),
    "8B-Think":    load(paths.paper_data("humaneval_8b_think")),
    "8B-NoThink":  load(paths.paper_data("humaneval_8b_nothink")),
}

def get_commits(results):
    return [r['commitment_fraction'] for r in results
            if r.get('commitment_fraction') is not None and r.get('n_correct_rollouts', 0) > 0]

def get_psc_efa(results):
    fracs = sorted(set(pr['fraction'] for r in results for pr in r.get('prefix_results', [])))
    psc, efa = [], []
    for f in fracs:
        ps = [pr['psc_agreement_rate'] for r in results for pr in r.get('prefix_results', [])
              if abs(pr['fraction'] - f) < 0.02]
        ef = [pr['efa_correct'] for r in results for pr in r.get('prefix_results', [])
              if abs(pr['fraction'] - f) < 0.02]
        psc.append(np.mean(ps) if ps else 0)
        efa.append(np.mean(ef) if ef else 0)
    return fracs, psc, efa


# ══════════════════════════════════════════════════════════════
# 1. HumanEval main figure (1x4)
# ══════════════════════════════════════════════════════════════
def fig_humaneval_main():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    fig.subplots_adjust(wspace=0.38, left=0.04, right=0.98, top=0.85, bottom=0.18)

    # (a) PSC curves
    ax = axes[0]
    for name, results in HE.items():
        fracs, psc, _ = get_psc_efa(results)
        ax.plot(fracs, psc, **_style(name), label=name)
    ax.axhline(0.75, color='#999', ls=':', linewidth=0.8)
    ax.set_xlabel('Prefix fraction')
    ax.set_ylabel('PSC agreement')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('(a) PSC — HumanEval', fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.5, loc='lower right')

    # (b) EFA curves — zoomed to actual data range; explicit call-out that EFA ≈ 0
    ax = axes[1]
    max_efa = 0.0
    for name, results in HE.items():
        fracs, _, efa = get_psc_efa(results)
        max_efa = max(max_efa, max(efa) if efa else 0.0)
        ax.plot(fracs, efa, **_style(name), label=name)
    ax.set_xlabel('Prefix fraction')
    ax.set_ylabel('EFA accuracy')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('(b) EFA — HumanEval', fontsize=11)
    # Zoom so the near-zero curves are readable; cap at max observed + headroom
    ax.set_ylim(0, max(0.08, max_efa * 1.4))
    ax.annotate('EFA $\\approx$ 0: code answers\nare not forceable',
                xy=(0.5, max(0.01, max_efa * 0.8)),
                xytext=(0.35, max(0.05, max_efa * 1.15)),
                fontsize=8, color='#b33', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='#d66', alpha=0.9, lw=0.6))

    # (c) Commitment distributions — use strip plot (boxplot collapses because
    # HumanEval commits are concentrated near 1.0 with almost no spread)
    ax = axes[2]
    names_list = list(HE.keys())
    data_list = [get_commits(HE[n]) for n in names_list]
    rng = np.random.default_rng(42)
    for i, (name, d) in enumerate(zip(names_list, data_list)):
        if not d:
            continue
        jitter = rng.normal(0, 0.12, len(d))
        ax.scatter(d, np.full(len(d), i + 1) + jitter, s=10, alpha=0.35,
                   color=PAL[name], edgecolors='none')
        # Median marker
        med = float(np.median(d))
        ax.plot([med, med], [i + 0.7, i + 1.3], color='black', linewidth=1.8, zorder=5)
        ax.text(med, i + 1.45, f'med {med:.0%}', fontsize=7, ha='center',
                color='#333')
    ax.set_yticks(range(1, len(names_list) + 1))
    ax.set_yticklabels(names_list, fontsize=8)
    ax.set_xlim(-0.02, 1.05)
    ax.set_xlabel('Commitment fraction')
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('(c) Commitment dist.', fontsize=11)
    ax.grid(False)

    # (d) Gap (PSC - EFA)
    ax = axes[3]
    for name, results in HE.items():
        fracs, psc, efa = get_psc_efa(results)
        gaps = [p - e for p, e in zip(psc, efa)]
        ax.plot(fracs, gaps, **_style(name), label=name)
    ax.axhline(0, color='#999', ls='-', linewidth=0.5)
    ax.set_xlabel('Prefix fraction')
    ax.set_ylabel('PSC $-$ EFA gap')
    ax.set_title('(d) Gap — HumanEval', fontsize=11)
    ax.set_ylim(-0.05, 0.7)
    ax.legend(fontsize=6.5, loc='upper right')

    _save(fig, 'fig_humaneval_main')


# ══════════════════════════════════════════════════════════════
# 2. 4-benchmark bar chart (MATH + GPQA + AIME + HumanEval)
# ══════════════════════════════════════════════════════════════
def fig_bars_four():
    # Load all benchmarks
    MATH = {n: load(paths.paper_data(p)) for n,p in [
        ("32B-Think","qwen3_32b_thinking_full500"),("32B-NoThink","qwen3_32b_no_thinking_full500"),
        ("8B-Think","qwen3_8b_thinking_full500"),("8B-NoThink","qwen3_8b_no_thinking_full500"),
        ("GPT-OSS-120B",  "gpt_oss_120b_full500")]}
    GPQA = {n: load(paths.paper_data(p)) for n,p in [
        ("32B-Think","gpqa_32b_think"),("32B-NoThink","gpqa_32b_nothink"),
        ("8B-Think","gpqa_8b_think"),("8B-NoThink","gpqa_8b_nothink"),
        ("GPT-OSS-120B","gpqa_gpt_oss_120b")]}
    AIME = {n: load(paths.paper_data(p)) for n,p in [
        ("32B-Think","aime24_32b_think"),("32B-NoThink","aime24_32b_nothink"),
        ("8B-Think","aime24_8b_think"),("8B-NoThink","aime24_8b_nothink"),
        ("GPT-OSS-120B","aime24_gpt_oss")]}

    PAL5 = {**PAL, "GPT-OSS-120B": "#238b45"}
    models = ["32B-Think", "32B-NoThink", "8B-Think", "8B-NoThink", "GPT-OSS-120B"]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(models))
    width = 0.20

    benchmarks = [
        ("MATH-500", MATH, 0.90, ''),
        ("GPQA", GPQA, 0.60, '///'),
        ("AIME", AIME, 0.40, '...'),
        ("HumanEval", {**HE, "GPT-OSS-120B": []}, 0.25, 'xxx'),
    ]

    for bi, (bname, data, alpha, hatch) in enumerate(benchmarks):
        vals = []
        for name in models:
            d = data.get(name, [])
            if not d:
                vals.append(0)
                continue
            commits = get_commits(d)
            vals.append(1 - np.mean(commits) if commits else 0)
        offset = (bi - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=bname,
                      color=[PAL5[n] for n in models], alpha=alpha,
                      edgecolor=[PAL5[n] for n in models] if bi > 0 else 'white',
                      linewidth=1.2 if bi > 0 else 0.5, hatch=hatch)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.0%}',
                        ha='center', va='bottom', fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Post-commitment fraction')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.0)
    ax.set_title('Post-Commitment Fraction Across Four Benchmarks')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    _save(fig, 'fig_bars_four_benchmarks')


if __name__ == '__main__':
    print("Generating HumanEval figures...")
    fig_humaneval_main()
    fig_bars_four()
    print("Done!")
