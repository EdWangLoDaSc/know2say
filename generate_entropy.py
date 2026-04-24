"""Entropy figure v4: annotated with key findings."""

import json, os
import numpy as np
import matplotlib.pyplot as plt

import paths
import plot_style

plot_style.apply_style()
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9})

from plot_style import PAL, LS  # noqa: E402

OUT = paths.FIGURES_DIR


def load(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]


MATH = {n: load(paths.paper_data(p)) for n, p in [
    ("32B-Think", "qwen3_32b_thinking_full500"), ("32B-NoThink", "qwen3_32b_no_thinking_full500"),
    ("8B-Think", "qwen3_8b_thinking_full500"), ("8B-NoThink", "qwen3_8b_no_thinking_full500"),
    ("GPT-OSS-120B", "gpt_oss_120b_full500")]}

N_BINS = 80
def _resample(ec, n):
    ec = np.array(ec, dtype=float)
    return ec[np.linspace(0, len(ec)-1, n).astype(int)]
def _smooth(arr, w=5):
    return np.convolve(arr, np.ones(w)/w, mode='same')


def fig_entropy():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3.5),
                                         gridspec_kw={'width_ratios': [1, 1, 0.85],
                                                      'wspace': 0.32})

    # ── (a) Entropy trajectory — all models, with commit region ──
    x = np.linspace(0, 100, N_BINS)
    for name, results in MATH.items():
        curves = [_resample(r['entropy_curve'], N_BINS) for r in results
                  if r.get('entropy_curve') and len(r['entropy_curve'])>=10
                  and r['selected_rollout_correct']]
        if not curves: continue
        med = _smooth(np.median(np.array(curves), axis=0))
        ax1.plot(x, med, color=PAL[name], ls=LS[name], linewidth=2.0, label=name)

    # Shade the typical commitment zone (10-50%)
    ax1.axvspan(10, 48, color='#fdae6b', alpha=0.08, zorder=0)
    ax1.annotate('typical commit\nzone (10--48%)', xy=(29, 0.037),
                 fontsize=8, ha='center', color='#8B4500', fontstyle='italic',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#e6a756',
                           alpha=0.85, lw=0.6))

    # Annotate Think spike
    ax1.annotate('Think models:\nearly spike',
                 xy=(8, 0.028), xytext=(35, 0.033),
                 fontsize=8, color=PAL["32B-Think"], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=PAL["32B-Think"], lw=1.0),
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=PAL["32B-Think"],
                           alpha=0.85, lw=0.6))

    ax1.set_xlabel('Position in CoT (%)')
    ax1.set_ylabel('Median entropy (nats)')
    ax1.set_title('(a) Entropy trajectory')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_ylim(0, 0.04)
    ax1.set_xlim(0, 100)
    ax1.grid(True, alpha=0.25)

    # ── (b) 32B-Think: correct vs wrong ──
    for label, cv, ls, color in [
        ('Correct', True, '-', PAL["32B-Think"]),
        ('Wrong', False, '--', '#cc4444'),
    ]:
        curves = [_resample(r['entropy_curve'], N_BINS) for r in MATH["32B-Think"]
                  if r.get('entropy_curve') and len(r['entropy_curve'])>=10
                  and r['selected_rollout_correct']==cv]
        if not curves: continue
        arr = np.array(curves)
        med = _smooth(np.median(arr, axis=0))
        q25 = _smooth(np.percentile(arr, 25, axis=0))
        q75 = _smooth(np.percentile(arr, 75, axis=0))
        ax2.plot(x, med, color=color, ls=ls, linewidth=2.0, label=label)
        ax2.fill_between(x, q25, q75, color=color, alpha=0.06)

    # Commitment line
    cfs = [r['commitment_fraction'] for r in MATH["32B-Think"] if r.get('commitment_fraction')]
    med_cf = np.median(cfs) * 100
    ax2.axvline(med_cf, color='#333', ls=':', linewidth=1.5, alpha=0.6)
    ax2.annotate('commit', xy=(med_cf+1, 0.057), fontsize=8, color='#333',
                 fontweight='bold')

    # Annotate wrong > correct
    ax2.annotate('wrong: 1.5$\\times$\nhigher entropy',
                 xy=(65, 0.042), fontsize=8, color='#cc4444', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cc4444',
                           alpha=0.85, lw=0.6))

    ax2.set_xlabel('Position in CoT (%)')
    ax2.set_ylabel('Entropy (nats)')
    ax2.set_title('(b) 32B-Think: correct vs wrong')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.set_ylim(0, 0.06)
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.25)

    # ── (c) Post/pre-commit entropy ratio ──
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

    ax3.hlines(y_pos, 1.0, vals, colors=colors, linewidth=4.0, alpha=0.7)
    ax3.scatter(vals, y_pos, color=colors, s=160, zorder=5,
                edgecolors='white', linewidth=1.5)
    ax3.axvline(1.0, color='#999', ls='--', linewidth=1)

    for i, (v, n) in enumerate(zip(vals, names)):
        side = 0.04 if v > 1.0 else -0.04
        ha = 'left' if v > 1.0 else 'right'
        ax3.text(v+side, i, f'{v:.2f}x', va='center', ha=ha, fontsize=11,
                 fontweight='bold', color=colors[i])

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=10)
    ax3.set_xlabel('Post / pre-commit entropy ratio')
    ax3.set_title('(c) Entropy ratio')
    ax3.set_xlim(0.5, 1.7)

    # Shade + label regions
    ax3.axvspan(1.0, 1.7, color='#ff4444', alpha=0.03)
    ax3.axvspan(0.5, 1.0, color='#44aa44', alpha=0.03)
    ax3.text(1.50, 4.3, 'rises after\ncommit', fontsize=7.5, color='#cc3333',
             ha='center', fontstyle='italic')
    ax3.text(0.65, 4.3, 'falls after\ncommit', fontsize=7.5, color='#339933',
             ha='center', fontstyle='italic')
    ax3.grid(False)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT, f'fig3_entropy.{ext}'))
    plt.close(fig)
    print('Saved fig3_entropy')


if __name__ == '__main__':
    fig_entropy()
