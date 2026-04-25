"""Generate standalone entropy ratio panel for wrapfigure."""
import json, os
import numpy as np
import matplotlib.pyplot as plt

import paths
import plot_style

plot_style.apply_style()
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.grid': False})

from plot_style import PAL  # noqa: E402


def load(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]


MATH = {n: load(paths.paper_data(p)) for n, p in [
    ("32B-Think", "qwen3_32b_thinking_full500"), ("32B-NoThink", "qwen3_32b_no_thinking_full500"),
    ("8B-Think", "qwen3_8b_thinking_full500"), ("8B-NoThink", "qwen3_8b_no_thinking_full500"),
    ("GPT-OSS-120B", "gpt_oss_120b_full500")]}

ratios = {}
for name, results in MATH.items():
    pre_l, post_l = [], []
    for r in results:
        ec = r.get('entropy_curve', [])
        cf = r.get('commitment_fraction')
        if not ec or cf is None or len(ec) < 10: continue
        ec = np.array(ec, dtype=float)
        split = int(cf * len(ec))
        if split < 5 or split > len(ec) - 5: continue
        pre_l.append(np.mean(ec[:split]))
        post_l.append(np.mean(ec[split:]))
    if pre_l:
        ratios[name] = np.mean(post_l) / np.mean(pre_l)

fig, ax = plt.subplots(figsize=(4.5, 3.5))

names = list(ratios.keys())
vals = [ratios[n] for n in names]
colors = [PAL[n] for n in names]
y_pos = range(len(names))

ax.hlines(y_pos, 1.0, vals, colors=colors, linewidth=4, alpha=0.7)
ax.scatter(vals, y_pos, color=colors, s=160, zorder=5, edgecolors='white', linewidth=1.5)
ax.axvline(1.0, color='#999', ls='--', linewidth=1)

for i, (v, n) in enumerate(zip(vals, names)):
    side = 0.03 if v > 1.0 else -0.03
    ha = 'left' if v > 1.0 else 'right'
    ax.text(v + side, i, f'{v:.2f}x', va='center', ha=ha, fontsize=11,
            fontweight='bold', color=colors[i])

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=11)
ax.set_xlabel('Post / pre-commit entropy ratio')
ax.set_xlim(0.5, 1.65)
ax.axvspan(1.0, 1.65, color='#ff4444', alpha=0.04)
ax.axvspan(0.5, 1.0, color='#44aa44', alpha=0.04)
ax.grid(False)

fig.tight_layout()
os.makedirs(paths.FIGURES_DIR, exist_ok=True)
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(paths.FIGURES_DIR, f'fig_entropy_ratio.{ext}'))
plt.close(fig)
print('Saved fig_entropy_ratio')
