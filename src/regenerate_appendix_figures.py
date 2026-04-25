"""Regenerate 4 appendix figures with updated terminology and style."""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter

import paths
import plot_style

plot_style.apply_style()

from plot_style import PAL, MARKERS, LS, COL_GENUINE, COL_WRONG  # noqa: E402

COL_PRE = COL_GENUINE
COL_POST = "#fdae6b"

BASE = paths.RESULTS_DIR
OUT = paths.FIGURES_DIR
def load(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]
def _style(n): return dict(color=PAL[n], marker=MARKERS[n], ls=LS[n], markersize=5, linewidth=1.8)
def _save(fig, name):
    for ext in ['pdf','png']:
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'))
    plt.close(fig); print(f'  Saved {name}')

MATH = {n: load(f"{BASE}/{p}/results.jsonl") for n,p in [
    ("32B-Think","qwen3_32b_thinking_full500"),("32B-NoThink","qwen3_32b_no_thinking_full500"),
    ("8B-Think","qwen3_8b_thinking_full500"),("8B-NoThink","qwen3_8b_no_thinking_full500"),
    ("GPT-OSS-120B","gpt_oss_120b_full500")]}
GPQA = {n: load(f"{BASE}/{p}/results.jsonl") for n,p in [
    ("32B-Think","gpqa_32b_think"),("32B-NoThink","gpqa_32b_nothink"),
    ("8B-Think","gpqa_8b_think"),("8B-NoThink","gpqa_8b_nothink"),
    ("GPT-OSS-120B","gpqa_gpt_oss_120b")]}
AIME = {n: load(f"{BASE}/{p}/results.jsonl") for n,p in [
    ("32B-Think","aime24_32b_think"),("32B-NoThink","aime24_32b_nothink"),
    ("8B-Think","aime24_8b_think"),("8B-NoThink","aime24_8b_nothink"),
    ("GPT-OSS-120B","aime24_gpt_oss")]}

def get_psc(results):
    fracs = sorted(set(pr['fraction'] for r in results for pr in r['prefix_results']))
    psc = [np.mean([pr['psc_agreement_rate'] for r in results for pr in r['prefix_results']
           if abs(pr['fraction']-f)<0.01 and r.get('n_correct_rollouts',0)>0]) for f in fracs]
    return fracs, psc

def get_commits(results):
    return [r['commitment_fraction'] for r in results if r.get('commitment_fraction') is not None]


# ═══════════════════════════════════════════════════════════════
# 1. theta_frontier — cleaner style
# ═══════════════════════════════════════════════════════════════
def fig_theta():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: NoThink FP-Savings
    for name in ["32B-NoThink", "8B-NoThink"]:
        results = MATH[name]
        thetas = np.arange(0.50, 1.001, 0.125)
        fps, savs = [], []
        wrong = [r for r in results if r.get('n_correct_rollouts', 0) == 0]
        for theta in thetas:
            fp_count = 0
            for r in wrong:
                for pr in r['prefix_results']:
                    if (pr.get('psc_agreement_rate') or 0) >= theta:
                        fp_count += 1; break
            fps.append(fp_count / len(wrong) * 100 if wrong else 0)
            sl = []
            for r in results:
                total = r['selected_rollout_len']
                for pr in sorted(r['prefix_results'], key=lambda x: x['fraction']):
                    if (pr.get('psc_agreement_rate') or 0) >= theta:
                        sl.append((total - pr['prefix_len']) / total * 100 if total > 0 else 0)
                        break
                else:
                    sl.append(0)
            savs.append(np.mean(sl))
        ax1.plot(savs, fps, color=PAL[name], marker=MARKERS[name], linewidth=2.0,
                 markersize=7, label=name)
        # Mark theta=0.875 (argmin to avoid float-equality lookup)
        idx = int(np.argmin(np.abs(thetas - 0.875)))
        ax1.scatter([savs[idx]], [fps[idx]], color=PAL[name], s=120,
                    edgecolors='black', linewidth=1.2, zorder=10)
        ax1.annotate(f'$\\theta$={thetas[idx]:.3f}',
                     xy=(savs[idx], fps[idx]),
                     xytext=(savs[idx] - 3.5, fps[idx] + 1.2),
                     fontsize=8, color=PAL[name],
                     bbox=dict(boxstyle='round,pad=0.18', fc='white',
                               ec=PAL[name], alpha=0.9, lw=0.6))

    ax1.set_xlabel('Mean token savings (%)')
    ax1.set_ylabel('Proxy FP rate (%)')
    ax1.set_title('(a) NoThink FP--savings frontier')
    ax1.legend(fontsize=9)

    # Panel B: Savings vs Theta
    for name, results in MATH.items():
        thetas = np.arange(0.125, 1.001, 0.125)
        savs = []
        for theta in thetas:
            sl = []
            for r in results:
                total = r['selected_rollout_len']
                for pr in sorted(r['prefix_results'], key=lambda x: x['fraction']):
                    if (pr.get('psc_agreement_rate') or 0) >= theta:
                        sl.append((total - pr['prefix_len']) / total * 100 if total > 0 else 0)
                        break
                else:
                    sl.append(0)
            savs.append(np.mean(sl))
        ax2.plot(thetas, savs, **_style(name), label=name)

    ax2.set_xlabel('PSC threshold $\\theta$')
    ax2.set_ylabel('Mean token savings (%)')
    ax2.set_title('(b) Savings vs $\\theta$')
    ax2.legend(fontsize=8, loc='upper right')

    fig.tight_layout(w_pad=3)
    _save(fig, 'theta_frontier')


# ═══════════════════════════════════════════════════════════════
# 2. fig_psc_monotonicity — updated style
# ═══════════════════════════════════════════════════════════════
def fig_mono():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, bname, data in [(ax1, "MATH-500", MATH), (ax2, "GPQA-Diamond", GPQA)]:
        for name, results in data.items():
            fracs, psc = get_psc(results)
            ax.plot(fracs, psc, **_style(name), label=name)
        ax.axhline(0.75, color='#999', ls=':', linewidth=0.8, label='$\\theta=0.75$')
        ax.set_xlabel('Prefix fraction')
        ax.set_ylabel('PSC agreement (solvable)')
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.30, 1.02)  # aligned across MATH/GPQA for honest comparison
        ax.legend(fontsize=7.5, loc='lower right')

    ax1.set_title('(a) MATH-500 — Non-monotone PSC')
    ax2.set_title('(b) GPQA-Diamond — Monotone PSC')

    fig.tight_layout(w_pad=3)
    _save(fig, 'fig_psc_monotonicity')


# ═══════════════════════════════════════════════════════════════
# 3. fig_aime_theater_map → commitment map, updated terminology
# ═══════════════════════════════════════════════════════════════
def fig_aime_map():
    results = AIME["32B-Think"]
    solvable, unsolvable = [], []
    for r in results:
        cf = r.get('commitment_fraction', 1.0) or 1.0
        if r.get('n_correct_rollouts', 0) > 0:
            solvable.append(cf)
        else:
            unsolvable.append(cf)
    solvable.sort()
    n_s, n_u = len(solvable), len(unsolvable)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, cf in enumerate(solvable):
        ax.barh(i, cf, height=1.0, color=COL_PRE, linewidth=0)
        ax.barh(i, 1.0-cf, height=1.0, color=COL_POST, linewidth=0, left=cf)
    for i in range(n_u):
        ax.barh(n_s+i, 1.0, height=1.0, color=COL_WRONG, linewidth=0)

    ax.plot(solvable, range(n_s), color='#222', linewidth=1.5, alpha=0.85)
    ax.axhline(n_s-0.5, color='#aa3333', linewidth=0.8, alpha=0.4)

    pc = np.mean([1-cf for cf in solvable])
    ax.annotate(f'Post-commit {pc:.0%}', xy=(0.72, n_s*0.25),
                fontsize=12, fontweight='bold', color='#7f4500', ha='center',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#e6a756', alpha=0.9))
    ax.annotate('Pre-commit', xy=(0.18, n_s*0.6),
                fontsize=11, fontweight='bold', color='#1a4d7a', ha='center',
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#6baed6', alpha=0.9))
    if n_u > 2:
        ax.text(0.5, n_s+n_u*0.5, f'Unsolvable ({n_u})', fontsize=9.5,
                color='#666', ha='center', va='center', fontstyle='italic')

    ax.set_xlim(0,1); ax.set_ylim(-0.5, n_s+n_u-0.5)
    ax.set_xlabel('Fraction of CoT'); ax.set_ylabel('Problems (sorted)')
    ax.set_yticks([]); ax.grid(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
    ax.set_title('32B-Think on AIME 2024', fontsize=13)
    ax.legend(handles=[
        mpatches.Patch(facecolor=COL_PRE, label='Pre-commitment'),
        mpatches.Patch(facecolor=COL_POST, label='Post-commitment'),
        mpatches.Patch(facecolor=COL_WRONG, label='Unsolvable'),
    ], fontsize=9, loc='upper left')
    fig.tight_layout()
    _save(fig, 'fig_aime_theater_map')


# ═══════════════════════════════════════════════════════════════
# 4. fig_theater_bars_three → post-commitment fraction bars
# ═══════════════════════════════════════════════════════════════
def fig_bars():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    models = list(MATH.keys())
    x = np.arange(len(models))
    width = 0.25

    bench_data = [
        ("MATH-500", MATH, 0.85, ''),
        ("GPQA-Diamond", GPQA, 0.50, '///'),
        ("AIME 2024", AIME, 0.30, '...'),
    ]

    for bi, (bname, data, alpha, hatch) in enumerate(bench_data):
        vals = []
        for name in models:
            commits = get_commits(data[name])
            vals.append(1 - np.mean(commits) if commits else 0)
        offset = (bi - 1) * width
        bars = ax.bar(x + offset, vals, width, label=bname,
                      color=[PAL[n] for n in models], alpha=alpha,
                      edgecolor=[PAL[n] for n in models] if bi > 0 else 'white',
                      linewidth=1.2 if bi > 0 else 0.5, hatch=hatch)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.0%}',
                    ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Post-commitment fraction')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 0.95)
    ax.set_title('Post-Commitment Fraction Across Benchmarks')
    ax.legend(fontsize=9.5, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    _save(fig, 'fig_theater_bars_three')


if __name__ == '__main__':
    print("Regenerating appendix figures...")
    fig_theta()
    fig_mono()
    fig_aime_map()
    fig_bars()
    print("Done!")
