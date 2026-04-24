"""AIME-2024 figures + three-benchmark (MATH / GPQA / AIME) comparison."""

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

BASE = paths.PAPER_DIR

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

AIME_PATHS = {
    "32B-Think":    paths.paper_data("aime24_32b_think"),
    "32B-NoThink":  paths.paper_data("aime24_32b_nothink"),
    "8B-Think":     paths.paper_data("aime24_8b_think"),
    "8B-NoThink":   paths.paper_data("aime24_8b_nothink"),
    "GPT-OSS-120B": paths.paper_data("aime24_gpt_oss"),
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
    commits = []
    for r in results:
        if r.get('n_correct_rollouts', 0) == 0:
            continue
        cf = r.get('commitment_fraction')
        if cf is not None:
            commits.append(cf)
        else:
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
AIME_DATA = {name: load(path) for name, path in AIME_PATHS.items()}
print(f"  MATH: {', '.join(f'{k}={len(v)}' for k,v in MATH_DATA.items())}")
print(f"  GPQA: {', '.join(f'{k}={len(v)}' for k,v in GPQA_DATA.items())}")
print(f"  AIME: {', '.join(f'{k}={len(v)}' for k,v in AIME_DATA.items())}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE: AIME Main Results (4 panels, same layout as GPQA/MATH main)
# ═══════════════════════════════════════════════════════════════════════════

def fig_aime_main():
    fig = plt.figure(figsize=(11, 8.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # (a) EFA accuracy
    ax = fig.add_subplot(gs[0, 0])
    for name, results in AIME_DATA.items():
        fracs, efa = get_efa_curve(results)
        ax.plot(fracs, efa, **_style(name), label=name)
    ax.set_xlabel('Prefix fraction')
    ax.set_ylabel('EFA accuracy')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('(a) Early Forced Answering — AIME 2024')
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=7, loc='lower right')

    # (b) PSC agreement
    ax = fig.add_subplot(gs[0, 1])
    for name, results in AIME_DATA.items():
        fracs, psc = get_psc_curve(results)
        ax.plot(fracs, psc, **_style(name), label=name)
    ax.axhline(0.75, color='#999999', ls=':', linewidth=1, label='$\\theta=0.75$')
    ax.set_xlabel('Prefix fraction')
    ax.set_ylabel('PSC agreement')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('(b) Prefix Self-Consistency — AIME 2024')
    ax.set_ylim(0.2, 1.02)
    ax.legend(fontsize=7, loc='lower right')

    # (c) Commitment distribution
    ax = fig.add_subplot(gs[1, 0])
    names_list = list(AIME_DATA.keys())
    data_list = [get_commitment_fracs(AIME_DATA[n]) for n in names_list]
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
    ax.set_title('(c) Commitment Distribution — AIME 2024')
    ax.set_xlim(0, 1.05)

    # (d) Detection-extraction gap
    ax = fig.add_subplot(gs[1, 1])
    for name, results in AIME_DATA.items():
        fracs, psc = get_psc_curve(results, solvable_only=False)
        _, efa = get_efa_curve(results)
        min_len = min(len(psc), len(efa))
        gaps = [psc[i] - efa[i] for i in range(min_len)]
        ax.plot(fracs[:min_len], gaps, **_style(name), label=name)
    ax.axhline(0, color='#999999', ls='-', linewidth=0.5)
    ax.set_xlabel('Prefix fraction')
    ax.set_ylabel('PSC $-$ EFA gap')
    ax.set_title('(d) Detection–Extraction Gap — AIME 2024')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(-0.05, 0.65)

    fig.tight_layout(w_pad=3)
    _save(fig, 'fig_aime_main')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE: Three-benchmark comparison (3×3: MATH / GPQA / AIME × PSC / Commit / Gap)
# ═══════════════════════════════════════════════════════════════════════════

def fig_three_benchmark():
    """9-panel: columns = MATH, GPQA, AIME; rows = PSC, Commitment, Gap."""
    benchmarks = [
        ("MATH-500", MATH_DATA),
        ("GPQA-Diamond", GPQA_DATA),
        ("AIME 2024", AIME_DATA),
    ]
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.30)

    for col, (bench_name, data) in enumerate(benchmarks):
        # Row 1: PSC curves
        ax = fig.add_subplot(gs[0, col])
        for name, results in data.items():
            fracs, psc = get_psc_curve(results)
            ax.plot(fracs, psc, **_style(name), label=name)
        ax.axhline(0.75, color='#999999', ls=':', linewidth=1)
        ax.set_xlabel('Prefix fraction')
        ax.set_ylabel('PSC agreement')
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f'({chr(97+col)}) PSC — {bench_name}')
        ax.set_ylim(0.2, 1.02)
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
        ax.set_title(f'({chr(100+col)}) Commitment — {bench_name}')
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
        ax.set_title(f'({chr(103+col)}) Gap — {bench_name}')
        ax.set_ylim(-0.05, 0.65)
        if col == 2:
            ax.legend(fontsize=6, loc='upper right')

    fig.tight_layout()
    _save(fig, 'fig_three_benchmark')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE: Theater fraction bars — 3 benchmarks
# ═══════════════════════════════════════════════════════════════════════════

def fig_theater_bars_three():
    """Grouped bars: theater fraction per model, MATH / GPQA / AIME."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    models = list(MATH_PATHS.keys())
    x = np.arange(len(models))
    width = 0.25

    bench_data = [
        ("MATH-500", MATH_DATA, 0.85),
        ("GPQA-Diamond", GPQA_DATA, 0.50),
        ("AIME 2024", AIME_DATA, 0.30),
    ]
    hatches = ['', '///', '...']

    for bi, (bench_name, data, alpha) in enumerate(bench_data):
        theater = []
        for name in models:
            mc = get_commitment_fracs(data[name])
            theater.append(1 - np.mean(mc) if mc else 0)
        offset = (bi - 1) * width
        bars = ax.bar(x + offset, theater, width, label=bench_name,
                       color=[PAL[n] for n in models], alpha=alpha,
                       edgecolor=[PAL[n] for n in models] if bi > 0 else 'white',
                       linewidth=1.2 if bi > 0 else 0.5, hatch=hatches[bi])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.0%}',
                    ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Theater fraction')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 0.95)
    ax.set_title('Theater Fraction Across Benchmarks')
    ax.legend(fontsize=9, loc='upper right')

    fig.tight_layout()
    _save(fig, 'fig_theater_bars_three')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE: AIME theater map (32B-Think)
# ═══════════════════════════════════════════════════════════════════════════

def fig_aime_theater_map():
    """Theater map for 32B-Think on AIME 2024."""
    results = AIME_DATA["32B-Think"]

    solvable = [r for r in results if r.get('n_correct_rollouts', 0) > 0]
    unsolvable = [r for r in results if r.get('n_correct_rollouts', 0) == 0]

    for r in solvable:
        if r.get('commitment_fraction') is None:
            r['commitment_fraction'] = 1.0
    solvable.sort(key=lambda r: r['commitment_fraction'])

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, r in enumerate(solvable):
        cf = r['commitment_fraction']
        ax.barh(i, cf, color='#2171b5', height=1.0, linewidth=0)
        ax.barh(i, 1.0 - cf, left=cf, color='#ffc107', height=1.0, linewidth=0)

    offset = len(solvable)
    for i, r in enumerate(unsolvable):
        ax.barh(offset + i, 1.0, color='#cccccc', height=1.0, linewidth=0)

    boundary_y = list(range(len(solvable)))
    boundary_x = [r['commitment_fraction'] for r in solvable]
    ax.plot(boundary_x, boundary_y, color='black', linewidth=1.2, alpha=0.7)

    ax.set_xlabel('Fraction of CoT')
    ax.set_ylabel('Problems (sorted by commitment)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(results))
    ax.set_title('32B-Think AIME 2024: Theater Map')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2171b5', label='Genuine reasoning'),
        Patch(facecolor='#ffc107', label='Post-commitment (theater)'),
        Patch(facecolor='#cccccc', label='Unsolvable'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

    fig.tight_layout()
    _save(fig, 'fig_aime_theater_map')


# ═══════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating AIME main figure...")
    fig_aime_main()
    print("Generating three-benchmark comparison...")
    fig_three_benchmark()
    print("Generating theater bars (3 benchmarks)...")
    fig_theater_bars_three()
    print("Generating AIME theater map...")
    fig_aime_theater_map()
    print("Done!")
