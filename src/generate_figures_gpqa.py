"""GPQA figures + combined MATH vs GPQA comparison figures."""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import PercentFormatter

import paths
import plot_style

plot_style.apply_style()

from plot_style import PAL, MARKERS, LS  # noqa: E402

MATH_PATHS = {
    "32B-Think":    paths.paper_data("qwen3_32b_thinking_full500"),
    "32B-NoThink":  paths.paper_data("qwen3_32b_no_thinking_full500"),
    "8B-Think":     paths.paper_data("qwen3_8b_thinking_full500"),
    "8B-NoThink":   paths.paper_data("qwen3_8b_no_thinking_full500"),
    "GPT-OSS-120B": paths.paper_data("gpt_oss_120b_full500"),
}

GPQA_PATHS = {
    "32B-Think":    paths.paper_data("gpqa_32b_think"),
    "32B-NoThink":  paths.paper_data("gpqa_32b_nothink"),
    "8B-Think":     paths.paper_data("gpqa_8b_think"),
    "8B-NoThink":   paths.paper_data("gpqa_8b_nothink"),
    "GPT-OSS-120B": paths.paper_data("gpqa_gpt_oss_120b"),
}

OUTPUT_DIR = paths.FIGURES_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load(path):
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def _save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  Saved {name}')


def _style(name):
    return dict(color=PAL[name], marker=MARKERS[name], ls=LS[name], markersize=5, linewidth=1.8)


def get_fracs(results):
    return sorted(set(
        pr['fraction'] for r in results for pr in r['prefix_results']
    ))


def get_psc_curve(results, solvable_only=True):
    if solvable_only:
        results = [r for r in results if r.get('n_correct_rollouts', 0) > 0]
    fracs = get_fracs(results)
    psc = []
    for f in fracs:
        vals = [pr['psc_agreement_rate'] for r in results for pr in r['prefix_results']
                if abs(pr['fraction'] - f) < 0.01 and pr['psc_agreement_rate'] is not None]
        psc.append(np.mean(vals) if vals else 0)
    return fracs, psc


def get_efa_curve(results):
    fracs = get_fracs(results)
    efa = []
    for f in fracs:
        vals = [pr['efa_correct'] for r in results for pr in r['prefix_results']
                if abs(pr['fraction'] - f) < 0.01]
        efa.append(np.mean(vals) if vals else 0)
    return fracs, efa


def get_commitment_fracs(results, theta=0.75):
    """Return list of commitment fractions for solvable problems."""
    commits = []
    for r in results:
        if r.get('n_correct_rollouts', 0) == 0:
            continue
        cf = r.get('commitment_fraction')
        if cf is not None:
            commits.append(cf)
        else:
            # Compute from prefix_results
            found = False
            for pr in r['prefix_results']:
                if (pr.get('psc_agreement_rate') or 0) >= theta:
                    commits.append(pr['fraction'])
                    found = True
                    break
            if not found:
                commits.append(1.0)
    return commits


# ═══════════════════════════════════════════════════════════════════════════
# Load all data
# ═══════════════════════════════════════════════════════════════════════════
print("Loading data...")
MATH_DATA = {name: load(path) for name, path in MATH_PATHS.items()}
GPQA_DATA = {name: load(path) for name, path in GPQA_PATHS.items()}
print(f"  MATH: {', '.join(f'{k}={len(v)}' for k,v in MATH_DATA.items())}")
print(f"  GPQA: {', '.join(f'{k}={len(v)}' for k,v in GPQA_DATA.items())}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE: GPQA Main Results (4 panels, same layout as fig2_main for MATH)
# ═══════════════════════════════════════════════════════════════════════════

def fig_gpqa_main():
    fig = plt.figure(figsize=(11, 8.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # (a) EFA accuracy
    ax = fig.add_subplot(gs[0, 0])
    for name, results in GPQA_DATA.items():
        fracs, efa = get_efa_curve(results)
        ax.plot(fracs, efa, **_style(name), label=name)
    ax.set_xlabel('Prefix fraction')
    ax.set_ylabel('EFA accuracy')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('(a) Early Forced Answering — GPQA')
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=7, loc='lower right')

    # (b) PSC agreement
    ax = fig.add_subplot(gs[0, 1])
    for name, results in GPQA_DATA.items():
        fracs, psc = get_psc_curve(results)
        ax.plot(fracs, psc, **_style(name), label=name)
    ax.axhline(0.75, color='#999999', ls=':', linewidth=1, label='$\\theta=0.75$')
    ax.set_xlabel('Prefix fraction')
    ax.set_ylabel('PSC agreement')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('(b) Prefix Self-Consistency — GPQA')
    ax.set_ylim(0.2, 1.02)
    ax.legend(fontsize=7, loc='lower right')

    # (c) Commitment distribution
    ax = fig.add_subplot(gs[1, 0])
    names_list = list(GPQA_DATA.keys())
    data_list = [get_commitment_fracs(GPQA_DATA[n]) for n in names_list]
    positions = list(range(len(names_list)))
    bp = ax.boxplot(data_list, positions=positions, vert=False, widths=0.6,
                    patch_artist=True, showfliers=False, medianprops=dict(color='black', linewidth=1.5))
    for patch, name in zip(bp['boxes'], names_list):
        patch.set_facecolor(PAL[name])
        patch.set_alpha(0.5)
    for i, (name, data) in enumerate(zip(names_list, data_list)):
        jitter = np.random.default_rng(42).normal(0, 0.12, len(data))
        ax.scatter(data, [i + j for j in jitter], color=PAL[name], alpha=0.3, s=8, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(names_list, fontsize=9)
    ax.set_xlabel('Commitment fraction')
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('(c) Commitment Distribution — GPQA')
    ax.set_xlim(0, 1.05)

    # (d) Detection-extraction gap
    ax = fig.add_subplot(gs[1, 1])
    for name, results in GPQA_DATA.items():
        fracs, psc = get_psc_curve(results, solvable_only=False)
        _, efa = get_efa_curve(results)
        gaps = [p - e for p, e in zip(psc, efa)]
        ax.plot(fracs, gaps, **_style(name), label=name)
    ax.axhline(0, color='#999999', ls='-', linewidth=0.5)
    ax.set_xlabel('Prefix fraction')
    ax.set_ylabel('PSC $-$ EFA gap')
    ax.set_title('(d) Detection–Extraction Gap — GPQA')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(-0.05, 0.65)

    fig.tight_layout(w_pad=3)
    _save(fig, 'fig_gpqa_main')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE: MATH vs GPQA side-by-side comparison (3×2 panels)
# ═══════════════════════════════════════════════════════════════════════════

def fig_combined_comparison():
    """6-panel figure: left column = MATH-500, right column = GPQA-Diamond.
    Row 1: EFA accuracy  (forced extractability comparison across benchmarks)
    Row 2: Commitment distributions
    Row 3: Detection-extraction gap
    NOTE: PSC curves are shown separately in fig_psc_monotonicity to avoid
    overlap with that figure."""
    fig = plt.figure(figsize=(11, 10))
    gs = gridspec.GridSpec(3, 2, hspace=0.40, wspace=0.30)

    for col, (bench_name, data) in enumerate([("MATH-500", MATH_DATA), ("GPQA-Diamond", GPQA_DATA)]):
        # Row 1: EFA accuracy curves (replaces PSC to avoid duplication with fig_psc_monotonicity)
        ax = fig.add_subplot(gs[0, col])
        for name, results in data.items():
            fracs, efa = get_efa_curve(results)
            ax.plot(fracs, efa, **_style(name), label=name)
        ax.set_xlabel('Prefix fraction')
        ax.set_ylabel('EFA accuracy')
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f'({chr(97+col)}) EFA Accuracy — {bench_name}')
        ax.set_ylim(0, 1.02)
        if col == 0:
            ax.legend(fontsize=6, loc='lower right')

        # Row 2: Commitment distributions
        ax = fig.add_subplot(gs[1, col])
        names_list = list(data.keys())
        data_list = [get_commitment_fracs(data[n]) for n in names_list]
        positions = list(range(len(names_list)))
        bp = ax.boxplot(data_list, positions=positions, vert=False, widths=0.6,
                        patch_artist=True, showfliers=False, medianprops=dict(color='black', linewidth=1.5))
        for patch, name in zip(bp['boxes'], names_list):
            patch.set_facecolor(PAL[name])
            patch.set_alpha(0.5)
        for i, (name, d) in enumerate(zip(names_list, data_list)):
            jitter = np.random.default_rng(42).normal(0, 0.12, len(d))
            ax.scatter(d, [i + j for j in jitter], color=PAL[name], alpha=0.25, s=6, zorder=3)
        ax.set_yticks(positions)
        ax.set_yticklabels(names_list, fontsize=8)
        ax.set_xlabel('Commitment fraction')
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f'({chr(99+col)}) Commitment — {bench_name}')
        ax.set_xlim(0, 1.05)

        # Row 3: Detection-extraction gap
        ax = fig.add_subplot(gs[2, col])
        for name, results in data.items():
            fracs, psc = get_psc_curve(results, solvable_only=False)
            _, efa = get_efa_curve(results)
            min_len = min(len(psc), len(efa))
            gaps = [psc[i] - efa[i] for i in range(min_len)]
            ax.plot(fracs[:min_len], gaps, **_style(name), label=name)
        ax.axhline(0, color='#999999', ls='-', linewidth=0.5)
        ax.set_xlabel('Prefix fraction')
        ax.set_ylabel('PSC $-$ EFA gap')
        ax.set_title(f'({chr(101+col)}) Gap — {bench_name}')
        ax.set_ylim(-0.05, 0.65)
        if col == 1:
            ax.legend(fontsize=6, loc='upper right')

    fig.tight_layout()
    _save(fig, 'fig_combined_math_gpqa')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE: PSC monotonicity comparison (MATH non-monotone vs GPQA monotone)
# ═══════════════════════════════════════════════════════════════════════════

def fig_psc_monotonicity():
    """2-panel: PSC curves on MATH (non-monotone) vs GPQA (monotone)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, bench_name, data in [(ax1, "MATH-500", MATH_DATA), (ax2, "GPQA-Diamond", GPQA_DATA)]:
        for name, results in data.items():
            fracs, psc = get_psc_curve(results)
            ax.plot(fracs, psc, **_style(name), label=name)
        ax.axhline(0.75, color='#999999', ls=':', linewidth=1, label='$\\theta=0.75$')
        ax.set_xlabel('Prefix fraction')
        ax.set_ylabel('PSC agreement (solvable)')
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.3, 1.02)
        ax.legend(fontsize=7, loc='lower right')

    ax1.set_title('(a) MATH-500 — Non-monotone PSC')
    ax2.set_title('(b) GPQA-Diamond — Monotone PSC')

    # Annotate the key difference
    ax1.annotate('PSC declines\nat late prefixes',
                 xy=(0.80, 0.68), xytext=(0.55, 0.45),
                 arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2),
                 fontsize=8, color='#666666', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#999999', alpha=0.9))
    ax2.annotate('PSC rises\nmonotonically',
                 xy=(0.70, 0.75), xytext=(0.35, 0.45),
                 arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2),
                 fontsize=8, color='#666666', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#999999', alpha=0.9))

    fig.tight_layout(w_pad=3)
    _save(fig, 'fig_psc_monotonicity')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE: Grouped bar chart — commitment by benchmark × model
# ═══════════════════════════════════════════════════════════════════════════

def fig_benchmark_comparison_bars():
    """Grouped bars: theater fraction per model, MATH vs GPQA side by side."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    models = list(MATH_PATHS.keys())
    x = np.arange(len(models))
    width = 0.35

    math_theater = []
    gpqa_theater = []
    for name in models:
        mc = get_commitment_fracs(MATH_DATA[name])
        gc = get_commitment_fracs(GPQA_DATA[name])
        math_theater.append(1 - np.mean(mc))
        gpqa_theater.append(1 - np.mean(gc))

    bars1 = ax.bar(x - width/2, math_theater, width, label='MATH-500',
                   color=[PAL[n] for n in models], alpha=0.7, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, gpqa_theater, width, label='GPQA-Diamond',
                   color=[PAL[n] for n in models], alpha=0.35, edgecolor=[PAL[n] for n in models],
                   linewidth=1.5, hatch='///')

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.0%}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Theater fraction')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 0.95)
    ax.set_title('Theater fraction: MATH-500 vs GPQA-Diamond')
    ax.legend(fontsize=9, loc='upper right')

    fig.tight_layout()
    _save(fig, 'fig_benchmark_bars')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE: GPQA theater map (32B-Think, like the MATH hero)
# ═══════════════════════════════════════════════════════════════════════════

def fig_gpqa_theater_map():
    """Theater map for 32B-Think on GPQA-Diamond."""
    results = GPQA_DATA["32B-Think"]

    # Split into solvable/unsolvable
    solvable = [r for r in results if r.get('n_correct_rollouts', 0) > 0]
    unsolvable = [r for r in results if r.get('n_correct_rollouts', 0) == 0]

    # Sort solvable by commitment fraction
    for r in solvable:
        if r.get('commitment_fraction') is None:
            r['commitment_fraction'] = 1.0
    solvable.sort(key=lambda r: r['commitment_fraction'])

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, r in enumerate(solvable):
        cf = r['commitment_fraction']
        # Blue = genuine (before commitment), Gold = theater (after)
        ax.barh(i, cf, color='#2171b5', height=1.0, linewidth=0)
        ax.barh(i, 1.0 - cf, left=cf, color='#ffc107', height=1.0, linewidth=0)

    # Unsolvable at bottom (gray)
    offset = len(solvable)
    for i, r in enumerate(unsolvable):
        ax.barh(offset + i, 1.0, color='#cccccc', height=1.0, linewidth=0)

    # Commitment boundary line
    boundary_y = list(range(len(solvable)))
    boundary_x = [r['commitment_fraction'] for r in solvable]
    ax.plot(boundary_x, boundary_y, color='black', linewidth=1.2, alpha=0.7)

    ax.set_xlabel('Fraction of CoT')
    ax.set_ylabel('Problems (sorted by commitment)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(results))
    ax.set_title('32B-Think GPQA-Diamond: Theater Map')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2171b5', label='Genuine reasoning'),
        Patch(facecolor='#ffc107', label='Post-commitment (theater)'),
        Patch(facecolor='#cccccc', label='Unsolvable'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

    fig.tight_layout()
    _save(fig, 'fig_gpqa_theater_map')


# ═══════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating GPQA figures...")
    fig_gpqa_main()
    print("Generating combined MATH vs GPQA comparison...")
    fig_combined_comparison()
    print("Generating PSC monotonicity comparison...")
    fig_psc_monotonicity()
    print("Generating benchmark comparison bars...")
    fig_benchmark_comparison_bars()
    print("Generating GPQA theater map...")
    fig_gpqa_theater_map()
    print("Done!")
