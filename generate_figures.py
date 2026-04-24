"""DEER-inspired figures: heatmap, stacked bar, case study."""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.ticker import PercentFormatter

import paths
import plot_style

plot_style.apply_style()
# This figure uses no grid and tighter padding
plt.rcParams.update({'axes.grid': False, 'savefig.pad_inches': 0.02})

from plot_style import PAL, COL_GENUINE, COL_THEATER, COL_WRONG, COL_BOUNDARY  # noqa: E402

BASE = paths.PAPER_DIR
PATHS = {
    "32B-Think":    paths.paper_data("qwen3_32b_thinking_full500"),
    "32B-NoThink":  paths.paper_data("qwen3_32b_no_thinking_full500"),
    "8B-Think":     paths.paper_data("qwen3_8b_thinking_full500"),
    "8B-NoThink":   paths.paper_data("qwen3_8b_no_thinking_full500"),
    "GPT-OSS-120B": paths.paper_data("gpt_oss_120b_full500"),
}

OUTPUT_DIR = paths.FIGURES_DIR


def load(path):
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


ALL_DATA = {name: load(path) for name, path in PATHS.items()}


def _save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.{ext}'))
    plt.close(fig)
    print(f"  {name}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE A: "Theater Map" — each row = one problem, color = reasoning stage
# Inspired by DEER Figure 1a
# ═══════════════════════════════════════════════════════════════════════════

def fig_theater_map():
    """Heatmap: rows=problems (sorted by commitment), columns=normalized CoT position."""
    from matplotlib.patches import Patch, FancyArrowPatch
    from matplotlib.lines import Line2D

    C_GENUINE  = "#2b7bba"
    C_THEATER  = "#f07322"
    C_WRONG    = "#d0d0d0"
    C_BOUNDARY = "#111111"
    C_SEP      = "#cc2222"

    DISPLAY = {
        "32B-Think":  "Qwen3-32B  (thinking mode)",
        "8B-NoThink": "Qwen3-8B  (no-think mode)",
    }

    # Wide figure; generous wspace so right-side labels of left panel
    # never collide with the right panel
    fig, axes = plt.subplots(1, 2, figsize=(22, 11), facecolor='white')
    fig.subplots_adjust(left=0.07, right=0.95, top=0.88,
                        bottom=0.28, wspace=0.38)

    for ax_idx, name in enumerate(["32B-Think", "8B-NoThink"]):
        ax = axes[ax_idx]
        results = ALL_DATA[name]
        N_COLS = 100

        correct_rows, wrong_rows = [], []
        for r in results:
            cf = r.get('commitment_fraction')
            if cf is None:
                cf = 1.0
            correct = r['selected_rollout_correct']
            row = np.zeros(N_COLS)
            commit_col = min(int(cf * N_COLS), N_COLS - 1)
            if correct:
                row[:commit_col] = 1
                row[commit_col:] = 2
                correct_rows.append((cf, row))
            else:
                row[:] = 3
                wrong_rows.append((cf, row))

        correct_rows.sort(key=lambda x: x[0])
        wrong_rows.sort(key=lambda x: x[0])

        all_rows    = [r for _, r in correct_rows] + [r for _, r in wrong_rows]
        matrix      = np.array(all_rows)
        correct_cfs = [cf for cf, _ in correct_rows]
        n_correct   = len(correct_rows)
        n_wrong     = len(wrong_rows)
        n_total     = n_correct + n_wrong

        cmap = mcolors.ListedColormap(['#f0f0f0', C_GENUINE, C_THEATER, C_WRONG])
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm,
                  interpolation='nearest')

        # ── Commitment boundary ──────────────────────────────────────────────
        ax.plot([cf * N_COLS for cf in correct_cfs], range(n_correct),
                color=C_BOUNDARY, linewidth=2.5, alpha=0.95, zorder=5)

        # ── Correct / wrong separator ────────────────────────────────────────
        ax.axhline(n_correct - 0.5, color=C_SEP, linewidth=2.2,
                   linestyle='--', alpha=0.9, zorder=6)

        # ── Annotations (solid bg, white bold text) ──────────────────────────
        theater_frac = np.mean([1 - cf for cf, _ in correct_rows])

        ax.text(76, n_correct * 0.15,
                f'Theater\n{theater_frac:.0%}',
                fontsize=30, fontweight='bold', color='white',
                ha='center', va='center', zorder=10,
                bbox=dict(boxstyle='round,pad=0.65', fc=C_THEATER,
                          alpha=0.93, ec='none'))

        ax.text(10, n_correct * 0.60,
                'Genuine',
                fontsize=30, fontweight='bold', color='white',
                ha='center', va='center', zorder=10,
                bbox=dict(boxstyle='round,pad=0.62', fc=C_GENUINE,
                          alpha=0.93, ec='none'))

        if n_wrong > 5:
            ax.text(50, n_correct + n_wrong * 0.45,
                    f'Incorrect  ({n_wrong} problems)',
                    fontsize=26, color='#555555',
                    ha='center', va='center', fontweight='bold')

        # ── Axes styling ─────────────────────────────────────────────────────
        ax.set_xlabel('Position in Chain-of-Thought (%)',
                      fontsize=34, fontweight='bold', labelpad=16, color='black')
        if ax_idx == 0:
            ax.set_ylabel('Problems  (sorted by commitment point)',
                          fontsize=32, fontweight='bold', labelpad=16, color='black')

        acc_pct = n_correct / n_total * 100
        ax.set_title(
            f'({chr(97 + ax_idx)})  {DISPLAY[name]}',
            fontsize=34, fontweight='bold', pad=20, loc='left', color='black')

        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'],
                           fontsize=30, fontweight='bold', color='black')
        ax.tick_params(axis='x', pad=10, colors='black', length=7, width=1.5)
        ax.set_yticks([])
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.8)
        ax.spines['bottom'].set_color('black')

        # ── Right-side bracket labels ─────────────────────────────────────────
        # Draw a thin vertical bar separating labels from heatmap
        y_ok  = 1.0 - (n_correct / 2) / n_total
        y_bad = 1.0 - (n_correct + n_wrong / 2) / n_total
        y_sep = 1.0 - n_correct / n_total   # separator position in axes fraction

        # Thin bracket line on the right edge
        ax.axvline(x=N_COLS - 0.5, color='#cccccc', linewidth=1.0, zorder=1)

        ax.annotate(f'n = {n_correct}\ncorrect',
                    xy=(1.02, y_ok), xycoords='axes fraction',
                    fontsize=26, fontweight='bold',
                    color=C_GENUINE, va='center', ha='left',
                    annotation_clip=False)
        ax.annotate(f'n = {n_wrong}\nwrong',
                    xy=(1.02, y_bad), xycoords='axes fraction',
                    fontsize=26, fontweight='bold',
                    color='#999999', va='center', ha='left',
                    annotation_clip=False)

        # Thin colored divider tick on right side
        ax.annotate('', xy=(1.015, y_sep), xycoords='axes fraction',
                    xytext=(1.015, y_sep + 0.001),
                    arrowprops=dict(arrowstyle='-', color=C_SEP, lw=1.5),
                    annotation_clip=False)

    # ── Legend (2 × 2) ───────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor=C_GENUINE,  edgecolor='none',
              label='Pre-commitment  (genuine reasoning)'),
        Patch(facecolor=C_THEATER,  edgecolor='none',
              label='Post-commitment  (theater)'),
        Patch(facecolor=C_WRONG,    edgecolor='#aaaaaa',
              label='Incorrect problems'),
        Line2D([0], [0], color=C_BOUNDARY, linewidth=2.5,
               label='Commitment boundary'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=28, frameon=True, framealpha=0.97,
               edgecolor='#dddddd', bbox_to_anchor=(0.5, 0.01),
               handlelength=1.8, handleheight=1.3,
               columnspacing=2.5, handletextpad=0.8)

    _save(fig, 'fig_theater_map')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE B: BAEE outcome stacked bar (inspired by DEER Figure 5)
# ═══════════════════════════════════════════════════════════════════════════

def fig_baee_outcomes():
    """For each model at θ=0.625: categorize problems into 4 types."""

    fig, ax = plt.subplots(figsize=(10, 5))

    categories = {
        'Overthinking corrected': '#2ecc71',   # green - was wrong in full CoT, BAEE correct
        'BAEE harmed': '#e74c3c',               # red - was correct in full CoT, BAEE wrong
        'Always correct': '#3498db',             # blue
        'Always wrong': '#95a5a6',               # gray
    }

    theta = 0.625
    x_pos = np.arange(len(ALL_DATA))
    width = 0.6

    model_names = list(ALL_DATA.keys())
    stacked_data = {cat: [] for cat in categories}

    for name in model_names:
        results = ALL_DATA[name]
        counts = {cat: 0 for cat in categories}

        for r in results:
            full_correct = r['selected_rollout_correct']

            # BAEE result
            baee_correct = None
            for pr in sorted(r['prefix_results'], key=lambda x: x['fraction']):
                if pr['psc_agreement_rate'] >= theta:
                    baee_correct = pr['psc_n_correct'] >= 5
                    break
            if baee_correct is None:
                baee_correct = full_correct  # fallback to full CoT

            if full_correct and baee_correct:
                counts['Always correct'] += 1
            elif not full_correct and not baee_correct:
                counts['Always wrong'] += 1
            elif not full_correct and baee_correct:
                counts['Overthinking corrected'] += 1
            elif full_correct and not baee_correct:
                counts['BAEE harmed'] += 1

        n = len(results)
        for cat in categories:
            stacked_data[cat].append(counts[cat] / n)

    # Stack bars
    bottom = np.zeros(len(model_names))
    for cat, color in categories.items():
        vals = stacked_data[cat]
        bars = ax.bar(x_pos, vals, width, bottom=bottom, color=color, label=cat,
                      edgecolor='white', linewidth=0.5)
        # Annotate percentages
        for i, v in enumerate(vals):
            if v > 0.03:  # only label if visible
                ax.text(x_pos[i], bottom[i] + v/2, f'{v:.0%}',
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        color='white' if cat != 'Always wrong' else '#333333')
        bottom += vals

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel('Fraction of problems')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f'BAEE Outcome Breakdown ($\\theta = 0.625$, majority-vote)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    _save(fig, 'fig_baee_outcomes')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE C: Per-problem savings vs accuracy scatter (DEER Figure 1b style)
# ═══════════════════════════════════════════════════════════════════════════

def fig_savings_scatter():
    """Scatter plot: x=token savings, y=accuracy, each dot=one model config.
    Show vanilla (0% savings) vs BAEE at different θ."""

    fig, ax = plt.subplots(figsize=(7, 5.5))

    thetas = [0.50, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0]

    for name, results in ALL_DATA.items():
        base_acc = np.mean([r['selected_rollout_correct'] for r in results])

        # Plot vanilla point
        ax.scatter([0], [base_acc*100], color=PAL[name], marker='x', s=80, zorder=10,
                   linewidths=2)

        accs, savs = [], []
        for theta in thetas:
            correct_list, savings_list = [], []
            for r in results:
                total = r['selected_rollout_len']
                exited = False
                for pr in sorted(r['prefix_results'], key=lambda x: x['fraction']):
                    if pr['psc_agreement_rate'] >= theta:
                        correct_list.append(pr['psc_n_correct'] >= 5)
                        savings_list.append((total - pr['prefix_len']) / total if total > 0 else 0)
                        exited = True
                        break
                if not exited:
                    correct_list.append(r['selected_rollout_correct'])
                    savings_list.append(0.0)
            accs.append(np.mean(correct_list) * 100)
            savs.append(np.mean(savings_list) * 100)

        ax.plot(savs, accs, color=PAL[name], marker='o', markersize=4,
                linewidth=1.5, alpha=0.8, label=name)

        # Highlight θ=0.625 point
        idx = thetas.index(0.625)
        ax.scatter([savs[idx]], [accs[idx]], color=PAL[name], s=120,
                   edgecolors='black', linewidth=1.5, zorder=15)

        # Arrow from vanilla to θ=0.625
        ax.annotate('', xy=(savs[idx], accs[idx]), xytext=(2, base_acc*100),
                   arrowprops=dict(arrowstyle='->', color=PAL[name], lw=1.2, alpha=0.4))

    ax.set_xlabel('Token savings (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('PSC-BAEE: Savings vs Accuracy\n(x = vanilla, large dot = $\\theta=0.625$)')
    ax.legend(fontsize=8, loc='lower left')

    # Shade the "free lunch" region
    ax.axhspan(ax.get_ylim()[0], 100, xmin=0, xmax=1, color='#f0f0f0', zorder=0)

    ax.set_xlim(-3, 90)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    _save(fig, 'fig_savings_scatter')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE D: Hero figure — combines theater concept + key result
# ═══════════════════════════════════════════════════════════════════════════

def fig_hero():
    """2-panel hero: (a) smooth fill_betweenx theater map, (b) gap fill for best model."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe

    # ── Refined palette ──────────────────────────────────────────────────────
    C_GENUINE  = '#1a6fb5'   # saturated blue
    C_THEATER  = '#e8691e'   # rich warm orange
    C_WRONG    = '#c5c5c5'   # soft gray
    C_BOUNDARY = '#1a1a2e'   # near-black with slight blue
    C_SEP      = '#c0392b'   # crimson

    fig = plt.figure(figsize=(24, 10), facecolor='white')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.32,
                          left=0.06, right=0.95, top=0.88, bottom=0.12)

    # ── (a) Theater map — fill_betweenx: clean regions, smooth boundary ───────
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor('white')

    correct_cfs, wrong_count = [], 0
    for r in ALL_DATA["32B-Think"]:
        cf = r.get('commitment_fraction') or 1.0
        if r['selected_rollout_correct']:
            correct_cfs.append(cf)
        else:
            wrong_count += 1

    correct_cfs.sort()          # ascending: earliest commit at y=0
    n_c = len(correct_cfs)
    n_w = wrong_count
    cfs_pct = np.array(correct_cfs) * 100
    y_c = np.arange(n_c, dtype=float)

    # Solid filled regions – no pixel grid artefacts
    ax.fill_betweenx(y_c, 0,       cfs_pct, color=C_GENUINE, alpha=0.85, linewidth=0)
    ax.fill_betweenx(y_c, cfs_pct, 100,     color=C_THEATER, alpha=0.82, linewidth=0)

    # Wrong-problem band above correct section
    if n_w > 0:
        y_w = np.arange(n_c, n_c + n_w, dtype=float)
        ax.fill_betweenx(y_w, 0, 100, color=C_WRONG, alpha=0.50, linewidth=0)
        ax.text(50, n_c + n_w * 0.50, f'Incorrect  (n = {n_w})',
                fontsize=24, color='#555', ha='center', va='center',
                fontstyle='italic')

    # Commitment boundary curve with glow
    ax.plot(cfs_pct, y_c, color='white', linewidth=4.5, zorder=4,
            solid_capstyle='round', alpha=0.6)
    ax.plot(cfs_pct, y_c, color=C_BOUNDARY, linewidth=2.2, zorder=5,
            solid_capstyle='round')

    # Divider: correct / wrong
    ax.axhline(n_c - 0.5, color=C_SEP, linewidth=1.2, linestyle='--',
               alpha=0.65, zorder=4)

    theater_frac = float(np.mean(1.0 - np.array(correct_cfs)))

    # ── Annotation badges (larger, with shadow) ──────────────────────────────
    shadow_props = [pe.withStroke(linewidth=3, foreground='white')]

    ax.text(76, n_c * 0.17, f'Theater\n{theater_frac:.0%}',
            fontsize=34, fontweight='bold', color='white',
            ha='center', va='center', zorder=10,
            bbox=dict(boxstyle='round,pad=0.50', fc=C_THEATER,
                      alpha=0.92, ec='#c0550e', lw=1.4,
                      mutation_aspect=0.9))
    ax.text(14, n_c * 0.72, 'Genuine',
            fontsize=32, fontweight='bold', color='white',
            ha='center', va='center', zorder=10,
            bbox=dict(boxstyle='round,pad=0.45', fc=C_GENUINE,
                      alpha=0.92, ec='#0e4d80', lw=1.4,
                      mutation_aspect=0.9))

    # Right-side bracket labels
    y_ok_frac  = 1.0 - (n_c / 2) / (n_c + n_w)
    y_bad_frac = 1.0 - (n_c + n_w / 2) / (n_c + n_w)
    ax.annotate(f'n = {n_c}', xy=(1.015, y_ok_frac), xycoords='axes fraction',
                fontsize=24, fontweight='bold', color=C_GENUINE,
                va='center', ha='left', annotation_clip=False)
    ax.annotate(f'n = {n_w}', xy=(1.015, y_bad_frac), xycoords='axes fraction',
                fontsize=24, fontweight='bold', color='#999',
                va='center', ha='left', annotation_clip=False)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, n_c + n_w + 0.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0 %', '25 %', '50 %', '75 %', '100 %'],
                       fontsize=34, fontweight='bold', color='black')
    ax.set_yticks([])
    ax.set_xlabel('Position in reasoning trace',
                  fontsize=32, fontweight='bold', labelpad=14, color='black')
    ax.set_ylabel('Problems  (sorted by commitment point)',
                  fontsize=30, fontweight='bold', labelpad=12, color='black')
    for spine in ['left', 'top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='x', colors='black', length=7, width=1.5)
    ax.set_title('(a)  Qwen3-32B  Think — MATH-500',
                 fontsize=32, fontweight='bold', pad=18, loc='left',
                 color='black')

    # ── (b) PSC vs EFA trajectories — highlight best-gap model ───────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('white')

    fracs = sorted(set(
        pr['fraction']
        for r in list(ALL_DATA.values())[0]
        for pr in r['prefix_results']
    ))
    frac_pct = [int(round(f * 100)) for f in fracs]

    # Collect curves for all models
    curves = {}
    for mname, mresults in ALL_DATA.items():
        psc_c, efa_c = [], []
        for f in fracs:
            ps = [pr['psc_agreement_rate']
                  for r in mresults for pr in r['prefix_results']
                  if abs(pr['fraction'] - f) < 0.01]
            ef = [float(pr['efa_correct'])
                  for r in mresults for pr in r['prefix_results']
                  if abs(pr['fraction'] - f) < 0.01]
            psc_c.append(np.mean(ps) * 100 if ps else float('nan'))
            efa_c.append(np.mean(ef) * 100 if ef else float('nan'))
        curves[mname] = (np.array(psc_c), np.array(efa_c))

    # Pick model with the largest mean PSC–EFA gap
    best = max(curves, key=lambda m: float(np.nanmean(
        curves[m][0] - curves[m][1])))

    # Background models: thin, muted
    for mname in ALL_DATA:
        if mname == best:
            continue
        p, e = curves[mname]
        ax2.plot(frac_pct, p, color='#d5d5d5', linewidth=1.0, ls='-',  zorder=2)
        ax2.plot(frac_pct, e, color='#d5d5d5', linewidth=0.8, ls='--', zorder=2)

    # Highlighted model
    psc_hi, efa_hi = curves[best]

    # Filled gap region — more visible
    ax2.fill_between(frac_pct, efa_hi, psc_hi,
                     color='#2171b5', alpha=0.22, zorder=3,
                     label='Detection–Extraction Gap')
    # PSC: solid blue with white outline for pop; EFA: dashed red
    ax2.plot(frac_pct, psc_hi, color='#08519c', linewidth=2.8,
             ls='-', marker='o', ms=6, zorder=6,
             markeredgecolor='white', markeredgewidth=1.2,
             label=f'PSC  ({best})',
             path_effects=[pe.withStroke(linewidth=4.5, foreground='white')])
    ax2.plot(frac_pct, efa_hi, color='#d62728', linewidth=2.2,
             ls='--', marker='s', ms=5, zorder=6,
             markeredgecolor='white', markeredgewidth=1.0,
             label=f'EFA  ({best})',
             path_effects=[pe.withStroke(linewidth=4.0, foreground='white')])

    # ── Key analysis annotations ──────────────────────────────────────────────

    # (1) Double-headed arrow at f=10% with gap value
    g0 = float(psc_hi[0] - efa_hi[0])
    mid_y = float((psc_hi[0] + efa_hi[0]) / 2)
    ax2.annotate('', xy=(frac_pct[0], float(efa_hi[0]) + 1.5),
                 xytext=(frac_pct[0], float(psc_hi[0]) - 1.5),
                 arrowprops=dict(arrowstyle='<->', color='#333',
                                 lw=1.6, mutation_scale=14), zorder=7)
    ax2.text(frac_pct[0] + 4, mid_y,
             f'Δ = {g0:.0f} pp',
             fontsize=24, color='black', va='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.28', fc='white', alpha=0.85,
                       ec='none'))

    # (2) PSC plateau annotation — PSC stays flat ~71-74%
    psc_mean = float(np.nanmean(psc_hi))
    ax2.annotate('PSC plateaus early\n(answer already detectable)',
                 xy=(35, float(psc_hi[2])),
                 xytext=(48, 95),
                 fontsize=19, color='#08519c', fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.35', fc='#e8f0fa',
                           ec='#6baed6', lw=0.8, alpha=0.92),
                 arrowprops=dict(arrowstyle='->', color='#6baed6',
                                 lw=1.4, connectionstyle='arc3,rad=-0.2'),
                 zorder=8)

    # (3) EFA slow climb annotation — position above the curve, left side
    ax2.annotate('EFA lags: extraction\nrequires more context',
                 xy=(40, float(efa_hi[3])),
                 xytext=(28, 8),
                 fontsize=19, color='#d62728', fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.35', fc='#fde8e8',
                           ec='#e88a8a', lw=0.8, alpha=0.92),
                 arrowprops=dict(arrowstyle='->', color='#e88a8a',
                                 lw=1.4, connectionstyle='arc3,rad=0.15'),
                 zorder=8)

    # (4) Gap-never-closes annotation at 90%
    g_last = float(psc_hi[-1] - efa_hi[-1])
    if g_last > 3:
        ax2.annotate(f'Gap persists at 90%\n(Δ = {g_last:.0f} pp)',
                     xy=(frac_pct[-1] - 1, float((psc_hi[-1] + efa_hi[-1]) / 2)),
                     xytext=(72, 47),
                     fontsize=18, color='#555', fontstyle='italic',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.30', fc='#f5f5f5',
                               ec='#bbb', lw=0.7, alpha=0.90),
                     arrowprops=dict(arrowstyle='->', color='#999',
                                     lw=1.0, connectionstyle='arc3,rad=0.2'),
                     zorder=8)

    # Legend: highlighted lines + ghost entry for background models
    ghost = Line2D([0], [0], color='#d0d0d0', lw=1.2, ls='-',
                   label='Other models')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles + [ghost],
               fontsize=20, loc='lower right', framealpha=0.95,
               handlelength=2.4, borderpad=0.8, edgecolor='#ddd',
               fancybox=True, shadow=False)

    ax2.set_xlabel('Prefix fraction (%)',
                   fontsize=32, fontweight='bold', labelpad=14, color='black')
    ax2.set_ylabel('Accuracy (%)',
                   fontsize=32, fontweight='bold', labelpad=12, color='black')
    ax2.set_title('(b)  Detection–Extraction Gap',
                  fontsize=32, fontweight='bold', pad=18, loc='left',
                  color='black')
    ax2.set_xlim(6, 94)
    ax2.set_xticks(frac_pct)
    ax2.tick_params(axis='both', labelsize=30, colors='black', length=7, width=1.5)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.15, color='#bbb', linewidth=0.6)
    ax2.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['bottom'].set_linewidth(1.8)
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_linewidth(1.8)
    ax2.spines['left'].set_color('black')

    _save(fig, 'fig_hero')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE E: Cross-benchmark bar chart (MATH-500 + GPQA-Diamond)
# fig_theater_bars_two — replaces the old 3-benchmark version that included AIME
# ═══════════════════════════════════════════════════════════════════════════

def fig_theater_bars_two():
    """Grouped bar chart of post-commitment fraction for MATH-500 and GPQA-Diamond.

    Data are loaded from the main MATH-500 results files. GPQA numbers are
    hard-coded from the paper tables (Table 2 / tab:gpqa) since the GPQA
    result files are stored separately and may not be present.
    """

    # Post-commit fractions from Table 2 (tab:gpqa) in the paper
    gpqa_post_commit = {
        "32B-Think":    1 - 0.33,   # 67%
        "32B-NoThink":  1 - 0.34,   # 66%
        "8B-Think":     1 - 0.37,   # 63%
        "8B-NoThink":   1 - 0.38,   # 62%
        "GPT-OSS-120B": 1 - 0.39,   # 61%
    }

    model_names = list(ALL_DATA.keys())

    # Compute MATH-500 post-commit from data
    math_post_commit = {}
    for name, results in ALL_DATA.items():
        cfs = [r['commitment_fraction'] for r in results
               if r['selected_rollout_correct'] and r.get('commitment_fraction') is not None]
        math_post_commit[name] = 1 - np.mean(cfs) if cfs else 0.0

    benchmarks = ["MATH-500", "GPQA-Diamond"]
    bench_data = [math_post_commit, gpqa_post_commit]
    hatches = ['', '//']

    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    for b_idx, (bench, data, hatch) in enumerate(zip(benchmarks, bench_data, hatches)):
        vals = [data[m] for m in model_names]
        offset = (b_idx - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=bench,
                      color=[PAL[m] for m in model_names],
                      hatch=hatch, edgecolor='white', linewidth=0.6, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f'{v:.0%}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel('Post-commitment fraction')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    ax.set_title('Post-Commitment Fraction: MATH-500 vs GPQA-Diamond\n'
                 '(solid = MATH-500, hatched = GPQA-Diamond)')

    # Custom legend: benchmarks only
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#888888', edgecolor='white', label='MATH-500'),
        Patch(facecolor='#888888', edgecolor='white', hatch='//', label='GPQA-Diamond'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='upper right')

    fig.tight_layout()
    _save(fig, 'fig_theater_bars_two')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE F: Three-benchmark bar chart (MATH-500 + GPQA-Diamond + HumanEval)
# fig_bars_three_benchmarks — replaces the old four-benchmark version
# ═══════════════════════════════════════════════════════════════════════════

def fig_bars_three_benchmarks():
    """Grouped bar chart across three benchmarks.

    GPQA and HumanEval numbers are hard-coded from the paper tables since
    those result files may not be present in this directory.
    """

    # From Table 2 (GPQA) and Table tab:humaneval
    gpqa_post_commit = {
        "32B-Think":    0.67,
        "32B-NoThink":  0.66,
        "8B-Think":     0.63,
        "8B-NoThink":   0.62,
        "GPT-OSS-120B": 0.61,
    }
    humaneval_post_commit = {
        "32B-Think":    0.853,
        "32B-NoThink":  0.867,
        "8B-Think":     0.859,
        "8B-NoThink":   0.876,
        "GPT-OSS-120B": None,   # not evaluated
    }

    model_names = list(ALL_DATA.keys())

    math_post_commit = {}
    for name, results in ALL_DATA.items():
        cfs = [r['commitment_fraction'] for r in results
               if r['selected_rollout_correct'] and r.get('commitment_fraction') is not None]
        math_post_commit[name] = 1 - np.mean(cfs) if cfs else 0.0

    benchmarks  = ["MATH-500", "GPQA-Diamond", "HumanEval"]
    bench_data  = [math_post_commit, gpqa_post_commit, humaneval_post_commit]
    hatches     = ['', '//', 'xx']
    n_bench     = len(benchmarks)
    width       = 0.25
    x           = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(11, 5))

    for b_idx, (bench, data, hatch) in enumerate(zip(benchmarks, bench_data, hatches)):
        offset = (b_idx - (n_bench - 1) / 2) * width
        vals = []
        for m in model_names:
            v = data.get(m)
            vals.append(v if v is not None else 0.0)
        bars = ax.bar(x + offset, vals, width, label=bench,
                      color=[PAL[m] for m in model_names],
                      hatch=hatch, edgecolor='white', linewidth=0.6, alpha=0.85)
        for bar, m, v in zip(bars, model_names, vals):
            if data.get(m) is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f'{v:.0%}', ha='center', va='bottom', fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel('Post-commitment fraction')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.10)
    ax.set_title('Post-Commitment Fraction Across Three Benchmarks\n'
                 '(solid = MATH-500, hatched = GPQA-Diamond, crossed = HumanEval)')

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#888888', edgecolor='white', label='MATH-500'),
        Patch(facecolor='#888888', edgecolor='white', hatch='//', label='GPQA-Diamond'),
        Patch(facecolor='#888888', edgecolor='white', hatch='xx', label='HumanEval'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='upper left')

    # Note for missing GPT-OSS HumanEval bar
    gpt_x = list(model_names).index("GPT-OSS-120B")
    ax.text(gpt_x + width, 0.03, 'N/A', ha='center', fontsize=7, color='#777777',
            style='italic')

    fig.tight_layout()
    _save(fig, 'fig_bars_three_benchmarks')


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating figures...")
    fig_theater_map()
    fig_baee_outcomes()
    fig_savings_scatter()
    fig_hero()
    fig_theater_bars_two()
    fig_bars_three_benchmarks()
    print("Done!")
