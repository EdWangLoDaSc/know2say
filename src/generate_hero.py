"""Hero v4 (wrapfigure-friendly, portrait two-panel).

  (a) top — case study: "one problem, two verdicts"
  (b) bottom — aggregate PSC vs EFA across five models
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter

import paths
import plot_style

plot_style.apply_style(large=True)

from plot_style import PAL  # noqa: E402


# ── Palette ─────────────────────────────────────────────────────────────────
TEAL        = '#0d9488'
TEAL_DEEP   = '#115e59'
TEAL_LIGHT  = '#d1fae5'
CORAL       = '#dc2626'
CORAL_DEEP  = '#991b1b'
CORAL_LIGHT = '#fee2e2'
INK         = '#0f172a'
SLATE       = '#475569'
RIBBON_PRE  = '#a7f3d0'   # stronger teal so it reads at small size
RIBBON_POST = '#e5e7eb'


PATHS = {
    "32B-Think":    paths.paper_data("qwen3_32b_thinking_full500"),
    "32B-NoThink":  paths.paper_data("qwen3_32b_no_thinking_full500"),
    "8B-Think":     paths.paper_data("qwen3_8b_thinking_full500"),
    "8B-NoThink":   paths.paper_data("qwen3_8b_no_thinking_full500"),
    "GPT-OSS-120B": paths.paper_data("gpt_oss_120b_full500"),
}
OUTPUT_DIR = paths.FIGURES_DIR


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


ALL = {n: load(p) for n, p in PATHS.items()}


# ─── Panel (a): case study ──────────────────────────────────────────────────
def panel_a(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # ── Problem statement ──────────────────────────────────────────────────
    ax.text(5.0, 9.55,
            r'$1 - 2 + 3 - 4 + \cdots + 99 - 100$',
            fontsize=34, ha='center', va='center', color=INK,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.50', fc='white',
                      ec='#cbd5e1', lw=1.8))
    ax.text(5.0, 8.80,
            r'GT $\mathbf{-50}$  $\cdot$  2081-token CoT  $\cdot$  32B-Think',
            fontsize=19, ha='center', va='center', color=INK,
            fontweight='bold')

    # ── Rollout ribbon ─────────────────────────────────────────────────────
    y0, y1 = 6.80, 7.40
    pre_frac = 0.10
    rx0, rw = 0.60, 9.00

    # pre-commit band
    ax.add_patch(mpatches.Rectangle((rx0, y0),
                                    rw * pre_frac, y1 - y0,
                                    facecolor=RIBBON_PRE,
                                    edgecolor=TEAL_DEEP, linewidth=2.2,
                                    zorder=3))
    # post-commit band
    ax.add_patch(mpatches.Rectangle((rx0 + rw * pre_frac, y0),
                                    rw * (1 - pre_frac), y1 - y0,
                                    facecolor=RIBBON_POST,
                                    edgecolor='#94a3b8', linewidth=1.6,
                                    zorder=3))

    # 10 % tick — label ABOVE tick (outside the V arrow wedge below)
    xp = rx0 + rw * pre_frac
    ax.plot([xp, xp], [y0 - 0.20, y1 + 0.30], color=INK, lw=3.6, zorder=4)
    ax.text(xp, y1 + 0.40, '10 % prefix',
            fontsize=20, ha='center', va='bottom',
            color=INK, fontweight='bold')

    # ribbon text (kept short to fit at all sizes)
    ax.text(rx0 + rw * (pre_frac + (1 - pre_frac) / 2), (y0 + y1) / 2,
            '90 % more tokens — post-commitment',
            fontsize=19, ha='center', va='center',
            color='#475569', fontstyle='italic', fontweight='bold')

    # ── Two verdict cards, dropped from the tick via clean angled arrows ───
    card_top_y = 5.10
    card_bot_y = 1.30
    card_w = 4.15
    left_cx, right_cx = 2.55, 7.45

    for cx, color in [(left_cx, TEAL), (right_cx, CORAL)]:
        ax.annotate('',
                    xy=(cx, card_top_y + 0.05),
                    xytext=(xp, y0 - 0.15),
                    arrowprops=dict(arrowstyle='-|>', color=color,
                                    lw=3.0, mutation_scale=24,
                                    shrinkA=0, shrinkB=2),
                    zorder=2)

    def _card(cx, color, color_deep, header, subtext, big_answer,
              mark, tail):
        # body
        ax.add_patch(FancyBboxPatch(
            (cx - card_w / 2, card_bot_y),
            card_w, card_top_y - card_bot_y,
            boxstyle='round,pad=0.10,rounding_size=0.22',
            facecolor='white', edgecolor=color, linewidth=3.4, zorder=4))
        # header strip
        ax.add_patch(mpatches.Rectangle(
            (cx - card_w / 2 + 0.05, card_top_y - 1.05),
            card_w - 0.10, 0.98, facecolor=color, edgecolor='none', zorder=5))
        ax.text(cx, card_top_y - 0.55, header,
                fontsize=22, ha='center', va='center', color='white',
                fontweight='bold', zorder=6)
        ax.text(cx, card_top_y - 1.70, subtext,
                fontsize=18, ha='center', va='center', color=INK,
                fontweight='bold', zorder=6)
        # big answer
        ax.text(cx - 0.55, card_bot_y + 1.10, big_answer,
                fontsize=72, ha='center', va='center',
                color=color_deep, fontweight='bold', zorder=6,
                path_effects=[pe.withStroke(linewidth=4, foreground='white')])
        ax.text(cx + 1.25, card_bot_y + 1.10, mark,
                fontsize=52, ha='center', va='center', color=color,
                fontweight='bold', zorder=6)
        ax.text(cx, card_bot_y + 0.30, tail,
                fontsize=19, ha='center', va='center',
                color=color_deep, zorder=6, fontweight='bold')

    _card(left_cx, TEAL, TEAL_DEEP,
          'PSC  ·  FREE CONT.',
          '8 samples from prefix',
          r'$\mathbf{-50}$', r'$\checkmark$',
          '8 / 8 agree')

    _card(right_cx, CORAL, CORAL_DEEP,
          'EFA  ·  FORCED EXT.',
          'answer-suffix + greedy',
          r'$\mathbf{50}$', r'$\times$',
          'sign dropped')

    # tag line
    ax.text(5.0, 0.45, 'Same prefix.  Two verdicts.',
            fontsize=28, ha='center', va='center',
            color=INK, fontweight='bold',
            path_effects=[pe.withStroke(linewidth=5, foreground='white')])

    ax.set_title('(a)   One problem, two verdicts',
                 fontsize=32, loc='left', pad=12, color=INK,
                 fontweight='bold')


# ─── Panel (b): aggregate PSC vs EFA ────────────────────────────────────────
def panel_b(ax):
    fracs = sorted(set(pr['fraction']
                       for r in list(ALL.values())[0]
                       for pr in r['prefix_results']))
    frac_pct = [round(f * 100) for f in fracs]

    curves = {}
    for mname, mres in ALL.items():
        psc_c, efa_c = [], []
        for f in fracs:
            ps = [pr['psc_agreement_rate']
                  for r in mres for pr in r['prefix_results']
                  if abs(pr['fraction'] - f) < 0.01]
            ef = [float(pr['efa_correct'])
                  for r in mres for pr in r['prefix_results']
                  if abs(pr['fraction'] - f) < 0.01]
            psc_c.append(np.mean(ps) * 100 if ps else np.nan)
            efa_c.append(np.mean(ef) * 100 if ef else np.nan)
        curves[mname] = (np.array(psc_c), np.array(efa_c))

    hero = "32B-Think"

    # background models
    for mname in curves:
        if mname == hero:
            continue
        p, e = curves[mname]
        ax.plot(frac_pct, p, color='#d4d4d8', lw=1.6, ls='-',  zorder=2)
        ax.plot(frac_pct, e, color='#d4d4d8', lw=1.3, ls='--', zorder=2)

    p_hi, e_hi = curves[hero]
    # gap fill
    ax.fill_between(frac_pct, e_hi, p_hi, color=TEAL, alpha=0.18, zorder=3,
                    label='detection–extraction gap')
    # hero curves
    ax.plot(frac_pct, p_hi, color=TEAL_DEEP, lw=5.0, ls='-', marker='o',
            ms=14, markeredgecolor='white', markeredgewidth=2.2, zorder=6,
            label=f'PSC  ({hero})',
            path_effects=[pe.withStroke(linewidth=6.5, foreground='white')])
    ax.plot(frac_pct, e_hi, color=CORAL, lw=4.0, ls='--', marker='s',
            ms=13, markeredgecolor='white', markeredgewidth=2.0, zorder=6,
            label=f'EFA  ({hero})',
            path_effects=[pe.withStroke(linewidth=6.0, foreground='white')])

    # Δ bracket at f = 10 %
    g0 = float(p_hi[0] - e_hi[0])
    x0 = frac_pct[0]
    ax.annotate('', xy=(x0, float(e_hi[0]) + 1.5),
                xytext=(x0, float(p_hi[0]) - 1.5),
                arrowprops=dict(arrowstyle='<->', color=INK, lw=3.5,
                                mutation_scale=30), zorder=7)
    ax.add_patch(FancyBboxPatch(
        (x0 + 3.5, (p_hi[0] + e_hi[0]) / 2 - 7),
        26, 14, boxstyle='round,pad=0.38,rounding_size=2.8',
        facecolor='white', edgecolor=INK, linewidth=2.6, zorder=8))
    ax.text(x0 + 16.5, (p_hi[0] + e_hi[0]) / 2,
            f'$\\Delta$ = {g0:.0f} pp',
            fontsize=36, ha='center', va='center', color=INK,
            fontweight='bold', zorder=9)

    # ghost entry in legend
    ghost = Line2D([0], [0], color='#c7c7cc', lw=2.0, ls='-',
                   label='other models')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [ghost], fontsize=20, loc='lower right',
              framealpha=0.96, handlelength=2.6, borderpad=0.8,
              edgecolor='#d4d4d8',
              prop={'weight': 'bold', 'size': 19})

    ax.set_xlabel('Prefix fraction (%)', fontsize=30, fontweight='bold',
                  color=INK, labelpad=12)
    ax.set_ylabel('Accuracy / agreement', fontsize=30, fontweight='bold',
                  color=INK, labelpad=10)
    ax.set_title('(b)   The gap is universal across five models',
                 fontsize=32, loc='left', pad=12, color=INK,
                 fontweight='bold')
    ax.set_xlim(6, 94)
    ax.set_xticks(frac_pct)
    ax.tick_params(axis='both', labelsize=24, colors=INK,
                   length=8, width=1.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.22, color='#d4d4d8', linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('bottom', 'left'):
        ax.spines[sp].set_linewidth(1.8)
        ax.spines[sp].set_color(INK)


# ─── Assemble ───────────────────────────────────────────────────────────────
def fig_hero_v4():
    fig = plt.figure(figsize=(10, 13), facecolor='white')
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.10, 1.00],
        hspace=0.22, left=0.09, right=0.97, top=0.955, bottom=0.075,
    )

    panel_a(fig.add_subplot(gs[0]))
    panel_b(fig.add_subplot(gs[1]))

    out_base = os.path.join(OUTPUT_DIR, 'fig_hero')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_base}.{ext}')
    plt.close(fig)
    print('  Saved fig_hero (v4 portrait, wrap-friendly)')


if __name__ == '__main__':
    fig_hero_v4()
