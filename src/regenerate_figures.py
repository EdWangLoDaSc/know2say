"""Regenerate all figures that need updating: combined, gpqa_theater_map, entropy, overthinking."""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

import paths
import plot_style

plot_style.apply_style()

from plot_style import PAL, MARKERS, LS, COL_GENUINE, COL_WRONG  # noqa: E402

# Post-commit uses a warmer orange than the generic theater color
COL_POST = "#fdae6b"

BASE = paths.RESULTS_DIR
OUTPUT = paths.FIGURES_DIR

def load(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]

def _style(n): return dict(color=PAL[n], marker=MARKERS[n], ls=LS[n], markersize=5, linewidth=1.8)

def _save(fig, name):
    for ext in ['pdf','png']:
        fig.savefig(os.path.join(OUTPUT, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  Saved {name}')

MATH = {n: load(f"{BASE}/{p}/results.jsonl") for n,p in [
    ("32B-Think","qwen3_32b_thinking_full500"),("32B-NoThink","qwen3_32b_no_thinking_full500"),
    ("8B-Think","qwen3_8b_thinking_full500"),("8B-NoThink","qwen3_8b_no_thinking_full500"),
    ("GPT-OSS-120B","gpt_oss_120b_full500")]}
GPQA = {n: load(f"{BASE}/{p}/results.jsonl") for n,p in [
    ("32B-Think","gpqa_32b_think"),("32B-NoThink","gpqa_32b_nothink"),
    ("8B-Think","gpqa_8b_think"),("8B-NoThink","gpqa_8b_nothink"),
    ("GPT-OSS-120B","gpqa_gpt_oss_120b")]}

def get_curves(results):
    fracs = sorted(set(pr['fraction'] for r in results for pr in r['prefix_results']))
    psc, efa = [], []
    for f in fracs:
        psc.append(np.mean([pr['psc_agreement_rate'] for r in results for pr in r['prefix_results'] if abs(pr['fraction']-f)<0.01]))
        efa.append(np.mean([pr['efa_correct'] for r in results for pr in r['prefix_results'] if abs(pr['fraction']-f)<0.01]))
    return fracs, psc, efa

def get_commits(results):
    return [r['commitment_fraction'] for r in results if r.get('commitment_fraction') is not None]


# ══════════════════════════════════════════════════════════════
# 1. fig_combined_math_gpqa — simplified 2×2 (PSC + commitment only)
# ══════════════════════════════════════════════════════════════
def fig_combined():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    fig.subplots_adjust(wspace=0.38, left=0.04, right=0.98, top=0.78, bottom=0.17)

    panels = [
        ("MATH-500", MATH),
        ("GPQA-Diamond", GPQA),
    ]

    for col, (bname, data) in enumerate(panels):
        # PSC (solid) + EFA (dashed) curves
        ax = axes[col]
        for name, results in data.items():
            fracs, psc, efa = get_curves(results)
            ax.plot(fracs, psc, color=PAL[name], marker=MARKERS[name], ls='-',
                    markersize=4.5, linewidth=1.8,
                    label=f'{name} PSC' if col == 0 else None)
            ax.plot(fracs, efa, color=PAL[name], marker=MARKERS[name], ls='--',
                    markersize=3.5, linewidth=1.2, alpha=0.7,
                    markerfacecolor='white', markeredgewidth=1.0,
                    label=f'{name} EFA' if col == 0 else None)
        ax.axhline(0.75, color='#999', ls=':', linewidth=0.8)
        ax.set_xlabel('Prefix fraction')
        ax.set_ylabel('PSC (solid) / EFA (dashed)')
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f'({chr(97+col)}) {bname}', fontsize=12)
        ax.set_ylim(0.1, 1.02)
        ax.set_xlim(0.05, 0.95)

    # Single figure-level legend for model identities + line-style key
    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], color=PAL[n], marker=MARKERS[n], ls='-', lw=1.8,
               markersize=5, label=n) for n in panels[0][1]
    ]
    style_handles = [
        Line2D([0], [0], color='#555', ls='-',  lw=1.8, label='PSC (detection)'),
        Line2D([0], [0], color='#555', ls='--', lw=1.2, label='EFA (extraction)'),
    ]
    fig.legend(handles=model_handles + style_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.99), ncol=7, fontsize=9, frameon=False,
               handletextpad=0.4, columnspacing=1.5)

    for col, (bname, data) in enumerate(panels):
        # Commitment distributions
        ax = axes[2 + col]
        names_list = list(data.keys())
        data_list = [get_commits(data[n]) for n in names_list]
        bp = ax.boxplot(data_list, vert=False, widths=0.55, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8),
                        flierprops=dict(markersize=1.5, alpha=0.2))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(PAL[names_list[i]])
            patch.set_alpha(0.45)
        for i, (name, d) in enumerate(zip(names_list, data_list)):
            jitter = np.random.default_rng(42).normal(0, 0.08, len(d))
            ax.scatter(d, np.full(len(d), i+1)+jitter, s=2, alpha=0.12, color=PAL[name])
        ax.set_yticks(range(1, len(names_list)+1))
        ax.set_yticklabels(names_list, fontsize=7.5)
        ax.set_xlabel('Commitment fraction')
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f'({chr(99+col)}) Commit — {bname}', fontsize=11)
        ax.grid(False)

    _save(fig, 'fig_combined_math_gpqa')


# ══════════════════════════════════════════════════════════════
# 2. fig_gpqa_theater_map — updated style
# ══════════════════════════════════════════════════════════════
def fig_gpqa_map():
    results = GPQA["32B-Think"]
    solvable, unsolvable = [], []
    for r in results:
        cf = r.get('commitment_fraction', 1.0) or 1.0
        if r['selected_rollout_correct']:
            solvable.append(cf)
        else:
            unsolvable.append(cf)
    solvable.sort()
    n_s, n_u = len(solvable), len(unsolvable)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, cf in enumerate(solvable):
        ax.barh(i, cf, height=1.0, color=COL_GENUINE, linewidth=0)
        ax.barh(i, 1.0-cf, height=1.0, color=COL_POST, linewidth=0, left=cf)
    for i in range(n_u):
        ax.barh(n_s+i, 1.0, height=1.0, color=COL_WRONG, linewidth=0)

    ax.plot(solvable, range(n_s), color='#222222', linewidth=1.5, alpha=0.85)
    ax.axhline(n_s-0.5, color='#aa3333', linewidth=0.8, alpha=0.4)

    pc_frac = np.mean([1-cf for cf in solvable])
    ax.annotate(f'Post-commit {pc_frac:.0%}', xy=(0.75, n_s*0.25),
                fontsize=12, fontweight='bold', color='#7f4500', ha='center',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#e6a756', alpha=0.9))
    ax.annotate('Pre-commit', xy=(0.15, n_s*0.6),
                fontsize=11, fontweight='bold', color='#1a4d7a', ha='center',
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#6baed6', alpha=0.9))
    if n_u > 3:
        ax.text(0.5, n_s+n_u*0.5, f'Unsolvable ({n_u})', fontsize=9,
                color='#666', ha='center', va='center', fontstyle='italic')

    ax.set_xlim(0,1); ax.set_ylim(-0.5, n_s+n_u-0.5)
    ax.set_xlabel('Fraction of CoT'); ax.set_ylabel('Problems (sorted)')
    ax.set_yticks([]); ax.grid(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
    ax.set_title('32B-Think on GPQA-Diamond', fontsize=13)

    legend_elements = [
        mpatches.Patch(facecolor=COL_GENUINE, label='Pre-commitment'),
        mpatches.Patch(facecolor=COL_POST, label='Post-commitment'),
        mpatches.Patch(facecolor=COL_WRONG, label='Unsolvable'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')
    fig.tight_layout()
    _save(fig, 'fig_gpqa_theater_map')


# ══════════════════════════════════════════════════════════════
# 3. fig3_entropy — 1×3 but taller, bigger text
# ══════════════════════════════════════════════════════════════
def fig_entropy():
    fig = plt.figure(figsize=(14, 4.4))
    gs = gridspec.GridSpec(1, 3, wspace=0.38, left=0.06, right=0.97,
                           top=0.90, bottom=0.15)
    N_BINS = 80

    def _resample(ec, n):
        ec = np.array(ec, dtype=float)
        pos = np.linspace(0, len(ec)-1, n).astype(int)
        return ec[pos]

    # (a) Median entropy trajectory
    ax = fig.add_subplot(gs[0])
    for name, results in MATH.items():
        curves = [_resample(r['entropy_curve'], N_BINS) for r in results
                  if r.get('entropy_curve') and len(r['entropy_curve'])>=10 and r['selected_rollout_correct']]
        if not curves: continue
        x = np.linspace(0, 100, N_BINS)
        med = np.median(np.array(curves), axis=0)
        med = np.convolve(med, np.ones(5)/5, mode='same')
        ax.plot(x, med, color=PAL[name], ls=LS[name], linewidth=2.0, label=name)
    ax.set_xlabel('Position in CoT (%)')
    ax.set_ylabel('Median entropy (nats)')
    ax.set_title('(a) Entropy trajectory')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 0.12)

    # (b) 32B-Think correct vs wrong
    ax = fig.add_subplot(gs[1])
    for label, cv, ls, alpha in [('Correct', True, '-', 1.0), ('Wrong', False, '--', 0.8)]:
        curves = [_resample(r['entropy_curve'], N_BINS) for r in MATH["32B-Think"]
                  if r.get('entropy_curve') and len(r['entropy_curve'])>=10 and r['selected_rollout_correct']==cv]
        if not curves: continue
        x = np.linspace(0, 100, N_BINS)
        med = np.convolve(np.median(np.array(curves), axis=0), np.ones(5)/5, mode='same')
        ax.plot(x, med, color=PAL["32B-Think"], ls=ls, linewidth=2.0, alpha=alpha, label=label)
    cfs = [r['commitment_fraction'] for r in MATH["32B-Think"] if r.get('commitment_fraction')]
    ax.axvline(np.median(cfs)*100, color='#333', ls=':', linewidth=1.5)
    ax.annotate('median\ncommit', xy=(np.median(cfs)*100, 0.20), fontsize=9, ha='center',
                color='#333', fontweight='bold')
    ax.set_xlabel('Position in CoT (%)')
    ax.set_ylabel('Entropy (nats)')
    ax.set_title('(b) 32B-Think: correct vs wrong')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.92)
    ax.set_ylim(0, 0.25)

    # (c) Pre/post-commit entropy ratio — lollipop
    ax = fig.add_subplot(gs[2])
    ratios = {}
    for name, results in MATH.items():
        pre_l, post_l = [], []
        for r in results:
            ec = r.get('entropy_curve', [])
            cf = r.get('commitment_fraction')
            if not ec or cf is None or len(ec)<10: continue
            ec = np.array(ec, dtype=float)
            split = int(cf * len(ec))
            if split < 5 or split > len(ec)-5: continue
            pre_l.append(np.mean(ec[:split]))
            post_l.append(np.mean(ec[split:]))
        if pre_l:
            ratios[name] = np.mean(post_l) / np.mean(pre_l)

    names = list(ratios.keys())
    vals = [ratios[n] for n in names]
    colors = [PAL[n] for n in names]
    y_pos = range(len(names))

    ax.hlines(y_pos, 1.0, vals, colors=colors, linewidth=3.5, alpha=0.7)
    ax.scatter(vals, y_pos, color=colors, s=140, zorder=5, edgecolors='white', linewidth=1.5)
    ax.axvline(1.0, color='#999', ls='--', linewidth=1)

    # Unified right-aligned labels at a fixed x position so numbers line up as a column
    x_label = 1.60
    for i, (v, n) in enumerate(zip(vals, names)):
        ax.text(x_label, i, f'{v:.2f}x', va='center', ha='right', fontsize=10,
                fontweight='bold', color=colors[i])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Post / pre-commit entropy ratio')
    ax.set_title('(c) Entropy ratio')
    ax.set_xlim(0.5, 1.7)
    ax.axvspan(1.0, 1.7, color='#ff4444', alpha=0.03)
    ax.axvspan(0.5, 1.0, color='#44aa44', alpha=0.03)
    ax.grid(False)

    _save(fig, 'fig3_entropy')


# ══════════════════════════════════════════════════════════════
# 4. fig4_overthinking — taller, bigger text
# ══════════════════════════════════════════════════════════════
def fig_overthinking():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    thetas = np.arange(0.50, 1.001, 0.0625)
    model_data = {}
    for name, results in MATH.items():
        base_acc = np.mean([r['selected_rollout_correct'] for r in results])
        accs, savs = [], []
        for theta in thetas:
            cl, sl = [], []
            for r in results:
                total = r['selected_rollout_len']
                exited = False
                for pr in sorted(r['prefix_results'], key=lambda x: x['fraction']):
                    if pr['psc_agreement_rate'] >= theta:
                        cl.append(pr['psc_n_correct'] >= 5)
                        sl.append((total - pr['prefix_len']) / total if total > 0 else 0)
                        exited = True; break
                if not exited:
                    cl.append(r['selected_rollout_correct']); sl.append(0.0)
            accs.append(np.mean(cl)); savs.append(np.mean(sl))
        model_data[name] = {'base': base_acc, 'accs': accs, 'savs': savs}

    # (a) ΔAcc vs θ
    for name in MATH:
        d = model_data[name]
        deltas = [(a - d['base'])*100 for a in d['accs']]
        ax1.plot(thetas, deltas, **_style(name), label=name)
    ax1.axhline(0, color='#666', ls='-', linewidth=0.8)
    ax1.axvline(0.625, color='#333', ls=':', linewidth=1.5, alpha=0.5)
    ax1.annotate('$\\theta = 5/8$', xy=(0.625, -4.5), fontsize=10, ha='center',
                 color='#333', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#aaa', alpha=0.9))
    ax1.axhspan(-6, 0, color='#ff4444', alpha=0.04)
    ax1.axhspan(0, 8, color='#33aa33', alpha=0.04)
    ax1.set_xlabel('PSC threshold $\\theta$')
    ax1.set_ylabel('$\\Delta$ Accuracy vs full CoT (pp)')
    ax1.set_title('(a) Early exit corrects overthinking')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_ylim(-5.5, 7.5); ax1.set_xlim(0.48, 1.02)

    # (b) Pareto
    for name in MATH:
        d = model_data[name]
        ax2.plot([s*100 for s in d['savs']], [a*100 for a in d['accs']],
                 **_style(name), label=name)
        ax2.axhline(d['base']*100, color=PAL[name], ls=':', linewidth=0.6, alpha=0.3)
        idx = int(np.argmin(np.abs(thetas - 0.625)))
        ax2.scatter([d['savs'][idx]*100], [d['accs'][idx]*100],
                    color=PAL[name], s=100, zorder=10, edgecolors='black', linewidth=1.0)
    ax2.set_xlabel('Token savings (%)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('(b) Pareto frontier')
    ax2.legend(fontsize=8, loc='lower left')
    ax2.set_xlim(55, 92); ax2.set_ylim(74, 100)

    fig.tight_layout(w_pad=3)
    _save(fig, 'fig4_overthinking')


if __name__ == '__main__':
    print("Regenerating figures...")
    fig_combined()
    fig_gpqa_map()
    fig_entropy()
    fig_overthinking()
    print("Done!")
