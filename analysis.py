"""
Publication-quality analysis for Know2Say experiments.

- Re-grades EFA answers (fixes trailing ``}.`` stripping bug).
- Produces a 6-panel figure with all models overlaid (PDF + PNG).
- Runs per-model BAEE savings simulation.
- Emits a LaTeX table and a grouped bar chart for commitment-by-level.

Usage:
    python analysis.py
    python analysis.py output_dir=/tmp/paper_figures
"""

import json
import os
from collections import defaultdict
from typing import Any

import chz
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RESULT_PATHS = {
    "32B-Think": "/tmp/tinker-examples/reasoning_theater/qwen3_32b_thinking/results.jsonl",
    "32B-NoThink": "/tmp/tinker-examples/reasoning_theater/qwen3_32b_no_thinking/results.jsonl",
    "8B-Think": "/tmp/tinker-examples/reasoning_theater/qwen3_8b_thinking/results.jsonl",
    "8B-NoThink": "/tmp/tinker-examples/reasoning_theater/qwen3_8b_no_thinking/results.jsonl",
    "GPT-OSS-120B": "/tmp/tinker-examples/reasoning_theater/gpt_oss_120b/results.jsonl",
}

MODEL_STYLES = {
    "32B-Think": {"color": "#2563eb", "marker": "o", "linestyle": "-"},
    "32B-NoThink": {"color": "#dc2626", "marker": "s", "linestyle": "--"},
    "8B-Think": {"color": "#7c3aed", "marker": "^", "linestyle": "-"},
    "8B-NoThink": {"color": "#ea580c", "marker": "D", "linestyle": "--"},
    "GPT-OSS-120B": {"color": "#059669", "marker": "P", "linestyle": "-."},
}

PREFIX_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 0.90]


@chz.chz
class AnalysisV2Config:
    result_paths: str | None = None  # JSON dict or use defaults
    output_dir: str = "/tmp/tinker-examples/reasoning_theater/paper_figures"
    save_pdf: bool = True
    save_png: bool = True
    show_plots: bool = False
    regrade_efa: bool = True  # fix trailing '}.' bug
    n_entropy_bins: int = 20


# ---------------------------------------------------------------------------
# Data loading and re-grading
# ---------------------------------------------------------------------------


def load_results(path: str) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def regrade_efa_answers(results: list[dict]) -> list[dict]:
    """Re-grade EFA answers with fixed stripping (rstrip('}.') instead of
    sequential rstrip('}').rstrip('.')).

    Modifies results in-place and recomputes commitment/theater fractions.
    Returns the same list for convenience.
    """
    try:
        from tinker_cookbook.recipes.math_rl.math_grading import grade_answer
    except ImportError:
        print("Warning: math_grading not available, skipping re-grading")
        return results

    n_flipped = 0
    for r in results:
        gt = r["ground_truth"]
        any_changed = False
        for pr in r["prefix_results"]:
            ans = pr["efa_answer"]
            if ans and not pr["efa_correct"]:
                cleaned = ans.strip().rstrip("}.").strip()
                if cleaned and cleaned != ans:
                    try:
                        if grade_answer(cleaned, gt):
                            pr["efa_correct"] = True
                            pr["efa_answer"] = cleaned
                            any_changed = True
                            n_flipped += 1
                    except Exception:
                        pass

        # Recompute commitment/theater if anything changed
        if any_changed and r["n_correct_rollouts"] > 0:
            commitment = None
            for pr in r["prefix_results"]:
                if pr["efa_correct"]:
                    commitment = pr["fraction"]
                    break
            r["commitment_fraction"] = commitment
            r["theater_fraction"] = (1.0 - commitment) if commitment is not None else None

    if n_flipped > 0:
        print(f"  Re-grading: {n_flipped} EFA answers flipped to correct")
    return results


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def analyze_model(results: list[dict]) -> dict[str, Any]:
    """Full analysis for one model's results."""
    n_total = len(results)
    correct_results = [r for r in results if r["n_correct_rollouts"] > 0]
    n_correct = len(correct_results)

    # Commitment
    committed = [r for r in correct_results if r["commitment_fraction"] is not None]
    commit_fracs = [r["commitment_fraction"] for r in committed]
    theater_fracs = [r["theater_fraction"] for r in committed]

    # By level
    by_level: dict[int, list[float]] = defaultdict(list)
    for r in committed:
        by_level[int(r["level"])].append(r["commitment_fraction"])

    # EFA curve
    efa_curve = {}
    for frac in PREFIX_FRACTIONS:
        n_efa_correct = 0
        n_efa_total = 0
        for r in correct_results:
            for pr in r["prefix_results"]:
                if abs(pr["fraction"] - frac) < 0.01:
                    n_efa_total += 1
                    if pr["efa_correct"]:
                        n_efa_correct += 1
        if n_efa_total > 0:
            efa_curve[frac] = n_efa_correct / n_efa_total

    # ATLT curve
    atlt_curve = {}
    for frac in PREFIX_FRACTIONS:
        lps = []
        for r in correct_results:
            for pr in r["prefix_results"]:
                if abs(pr["fraction"] - frac) < 0.01 and pr["atlt_logprob"] is not None:
                    lps.append(pr["atlt_logprob"])
        if lps:
            atlt_curve[frac] = {"mean": float(np.mean(lps)), "std": float(np.std(lps))}

    # PSC curve
    psc_curve = {}
    for frac in PREFIX_FRACTIONS:
        rates = []
        for r in correct_results:
            for pr in r["prefix_results"]:
                if abs(pr["fraction"] - frac) < 0.01:
                    rates.append(pr["psc_agreement_rate"])
        if rates:
            psc_curve[frac] = {"mean": float(np.mean(rates)), "std": float(np.std(rates))}

    # Entropy landscape
    all_curves = []
    for r in results:
        curve = r.get("entropy_curve", [])
        if curve and len(curve) >= 5:
            indices = np.linspace(0, len(curve) - 1, 20).astype(int)
            all_curves.append([curve[i] for i in indices])
    entropy = {}
    if all_curves:
        arr = np.array(all_curves)
        entropy = {
            "positions": np.linspace(0, 1, 20).tolist(),
            "median": np.median(arr, axis=0).tolist(),
            "p25": np.percentile(arr, 25, axis=0).tolist(),
            "p75": np.percentile(arr, 75, axis=0).tolist(),
        }

    # Theater distribution
    theater_dist = {}
    if theater_fracs:
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(theater_fracs, bins=bins)
        theater_dist = {
            "histogram_bins": bins,
            "histogram_counts": hist.tolist(),
            "mean": float(np.mean(theater_fracs)),
            "median": float(np.median(theater_fracs)),
            "frac_above_50": float(np.mean([t > 0.5 for t in theater_fracs])),
        }

    # BAEE simulation
    baee = simulate_baee(results)

    # Mean CoT length
    cot_lens = [r["selected_rollout_len"] for r in correct_results]
    mean_cot_len = float(np.mean(cot_lens)) if cot_lens else 0

    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total > 0 else 0,
        "n_committed": len(committed),
        "commitment_mean": float(np.mean(commit_fracs)) if commit_fracs else None,
        "commitment_median": float(np.median(commit_fracs)) if commit_fracs else None,
        "theater_mean": float(np.mean(theater_fracs)) if theater_fracs else None,
        "theater_median": float(np.median(theater_fracs)) if theater_fracs else None,
        "mean_cot_len": mean_cot_len,
        "by_level": {
            level: {
                "commitment_mean": float(np.mean(vals)),
                "commitment_median": float(np.median(vals)),
                "n": len(vals),
            }
            for level, vals in sorted(by_level.items())
        },
        "efa_curve": efa_curve,
        "atlt_curve": atlt_curve,
        "psc_curve": psc_curve,
        "entropy": entropy,
        "theater_dist": theater_dist,
        "baee": baee,
    }


# ---------------------------------------------------------------------------
# BAEE Simulation
# ---------------------------------------------------------------------------


def simulate_baee(results: list[dict]) -> dict[str, Any]:
    """Simulate BAEE (Black-box Adaptive Early Exit) savings.

    For each correctly-solved problem, walk prefix checkpoints in order.
    If EFA returns correct at fraction f, exit there. Savings = 1 - f.
    """
    savings_list = []
    exit_fractions = []
    n_early_exit = 0
    n_eligible = 0  # problems where rollout was correct
    n_false_exit = 0  # would have exited but answer was wrong (shouldn't happen by design)

    for r in results:
        if r["n_correct_rollouts"] == 0:
            continue
        if not r["selected_rollout_correct"]:
            continue
        n_eligible += 1

        exited = False
        for pr in r["prefix_results"]:
            if pr["efa_correct"]:
                # Would exit here
                savings = 1.0 - pr["fraction"]
                savings_list.append(savings)
                exit_fractions.append(pr["fraction"])
                n_early_exit += 1
                exited = True
                break

        if not exited:
            # No early exit — use full CoT
            savings_list.append(0.0)
            exit_fractions.append(1.0)

    if not savings_list:
        return {"n_eligible": 0}

    return {
        "n_eligible": n_eligible,
        "n_early_exit": n_early_exit,
        "early_exit_rate": n_early_exit / n_eligible if n_eligible > 0 else 0,
        "mean_savings": float(np.mean(savings_list)),
        "median_savings": float(np.median(savings_list)),
        "mean_exit_fraction": float(np.mean(exit_fractions)),
        "n_false_exit": n_false_exit,
    }


# ---------------------------------------------------------------------------
# LaTeX table export
# ---------------------------------------------------------------------------


def generate_latex_table(all_analysis: dict[str, dict]) -> str:
    """Generate a LaTeX table summarizing all models."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Reasoning theater metrics across Qwen3 model configurations. "
        r"Commitment is the earliest prefix fraction where EFA returns the correct answer. "
        r"Theater = 1 $-$ commitment. BAEE savings assume early exit at the commitment point.}",
        r"\label{tab:main-results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Acc. & Commit. & Theater & CoT len & BAEE sav. & Early exit \\",
        r"\midrule",
    ]

    for label, a in all_analysis.items():
        acc = f"{a['accuracy']:.0%}"
        commit = f"{a['commitment_mean']:.0%}" if a["commitment_mean"] is not None else "---"
        theater = f"{a['theater_mean']:.0%}" if a["theater_mean"] is not None else "---"
        cot_len = f"{a['mean_cot_len']:.0f}"
        baee_sav = f"{a['baee']['mean_savings']:.0%}" if a["baee"].get("mean_savings") else "---"
        early_exit = f"{a['baee']['early_exit_rate']:.0%}" if a["baee"].get("early_exit_rate") else "---"
        lines.append(
            f"Qwen3-{label} & {acc} & {commit} & {theater} & {cot_len} & {baee_sav} & {early_exit} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_level_latex_table(all_analysis: dict[str, dict]) -> str:
    """Generate LaTeX table of commitment by difficulty level."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean commitment fraction by MATH difficulty level (1=easiest, 5=hardest). "
        r"Lower values indicate earlier commitment and more reasoning theater.}",
        r"\label{tab:commitment-by-level}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Level 1 & Level 2 & Level 3 & Level 4 & Level 5 \\",
        r"\midrule",
    ]

    for label, a in all_analysis.items():
        by_level = a.get("by_level", {})
        vals = []
        for lvl in [1, 2, 3, 4, 5]:
            info = by_level.get(lvl, {})
            if info and info.get("n", 0) > 0:
                vals.append(f"{info['commitment_mean']:.0%} ({info['n']})")
            else:
                vals.append("---")
        lines.append(f"Qwen3-{label} & {' & '.join(vals)} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Publication figure
# ---------------------------------------------------------------------------


def make_publication_figure(
    all_analysis: dict[str, dict],
    output_dir: str,
    save_pdf: bool = True,
    save_png: bool = True,
    show: bool = False,
):
    """Generate 6-panel publication figure with all models overlaid."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        print("matplotlib not available -- skipping plots")
        return

    # NeurIPS-compatible styling
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
    })

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Black-box Logprob Probing Reveals Reasoning Theater in Chain-of-Thought Models",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # ── Panel A: EFA Curves ──
    ax = axes[0, 0]
    for label, a in all_analysis.items():
        style = MODEL_STYLES.get(label, {"color": "gray", "marker": "o", "linestyle": "-"})
        efa = a["efa_curve"]
        if efa:
            fracs = sorted(efa.keys())
            accs = [efa[f] for f in fracs]
            ax.plot(
                [f * 100 for f in fracs], accs,
                marker=style["marker"], linestyle=style["linestyle"],
                color=style["color"], linewidth=1.8, markersize=5, label=label,
            )
    ax.set_xlabel("CoT prefix (%)")
    ax.set_ylabel("P(EFA correct)")
    ax.set_title("A: Early Forced Answering")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)

    # ── Panel B: ATLT Curves ──
    ax = axes[0, 1]
    for label, a in all_analysis.items():
        style = MODEL_STYLES.get(label, {"color": "gray", "marker": "o", "linestyle": "-"})
        atlt = a["atlt_curve"]
        if atlt:
            fracs = sorted(atlt.keys())
            means = [atlt[f]["mean"] for f in fracs]
            ax.plot(
                [f * 100 for f in fracs], means,
                marker=style["marker"], linestyle=style["linestyle"],
                color=style["color"], linewidth=1.8, markersize=5, label=label,
            )
    ax.set_xlabel("CoT prefix (%)")
    ax.set_ylabel("Mean log P(answer)")
    ax.set_title("B: Answer Token Logprob")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)

    # ── Panel C: Entropy (median + IQR) ──
    ax = axes[0, 2]
    for label, a in all_analysis.items():
        style = MODEL_STYLES.get(label, {"color": "gray", "marker": "o", "linestyle": "-"})
        ent = a.get("entropy", {})
        if ent and "positions" in ent:
            pos = [p * 100 for p in ent["positions"]]
            ax.plot(
                pos, ent["median"],
                linestyle=style["linestyle"], color=style["color"],
                linewidth=1.5, label=label,
            )
            ax.fill_between(pos, ent["p25"], ent["p75"], alpha=0.1, color=style["color"])
    ax.set_xlabel("Relative CoT position (%)")
    ax.set_ylabel("Token entropy (nats)")
    ax.set_title("C: Entropy Dynamics")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    # ── Panel D: PSC Curves ──
    ax = axes[1, 0]
    for label, a in all_analysis.items():
        style = MODEL_STYLES.get(label, {"color": "gray", "marker": "o", "linestyle": "-"})
        psc = a["psc_curve"]
        if psc:
            fracs = sorted(psc.keys())
            means = [psc[f]["mean"] for f in fracs]
            ax.plot(
                [f * 100 for f in fracs], means,
                marker=style["marker"], linestyle=style["linestyle"],
                color=style["color"], linewidth=1.8, markersize=5, label=label,
            )
    ax.set_xlabel("CoT prefix (%)")
    ax.set_ylabel("Agreement rate")
    ax.set_title("D: Prefix Self-Consistency")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)

    # ── Panel E: Theater Distribution (overlaid step histograms) ──
    ax = axes[1, 1]
    for label, a in all_analysis.items():
        style = MODEL_STYLES.get(label, {"color": "gray", "linestyle": "-"})
        td = a.get("theater_dist", {})
        if td and "histogram_counts" in td:
            bins = td["histogram_bins"]
            counts = td["histogram_counts"]
            # Normalize to density for comparability
            total = sum(counts)
            density = [c / total if total > 0 else 0 for c in counts]
            ax.step(
                bins[:-1], density, where="post",
                color=style["color"], linestyle=style.get("linestyle", "-"),
                linewidth=1.8, label=label,
            )
            ax.fill_between(
                bins[:-1], density, step="post",
                alpha=0.08, color=style["color"],
            )
    ax.set_xlabel("Theater fraction")
    ax.set_ylabel("Density")
    ax.set_title("E: Theater Score Distribution")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")

    # ── Panel F: Commitment by Level (grouped bars) ──
    ax = axes[1, 2]
    levels = [1, 2, 3, 4, 5]
    n_models = len(all_analysis)
    bar_w = 0.8 / n_models
    for i, (label, a) in enumerate(all_analysis.items()):
        style = MODEL_STYLES.get(label, {"color": "gray"})
        by_level = a.get("by_level", {})
        means = []
        for lvl in levels:
            info = by_level.get(lvl, {})
            means.append(info.get("commitment_mean", 0) if info.get("n", 0) > 0 else 0)
        x = np.arange(len(levels))
        ax.bar(
            x + (i - n_models / 2 + 0.5) * bar_w, means,
            width=bar_w, color=style["color"], alpha=0.8, label=label,
        )
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_xlabel("MATH difficulty level")
    ax.set_ylabel("Mean commitment fraction")
    ax.set_title("F: Commitment by Difficulty")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    if save_png:
        path = os.path.join(output_dir, "reasoning_theater_6panel.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved PNG: {path}")
    if save_pdf:
        path = os.path.join(output_dir, "reasoning_theater_6panel.pdf")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved PDF: {path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_analysis(config: AnalysisV2Config):
    # Load result paths
    if config.result_paths:
        result_paths = json.loads(config.result_paths)
    else:
        result_paths = DEFAULT_RESULT_PATHS

    # Load and optionally re-grade
    all_results: dict[str, list[dict]] = {}
    for label, path in result_paths.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {label}")
            continue
        results = load_results(path)
        print(f"Loaded {len(results)} results for {label}")
        if config.regrade_efa:
            results = regrade_efa_answers(results)
        all_results[label] = results

    if not all_results:
        print("No results found.")
        return

    # Analyze each model
    all_analysis: dict[str, dict] = {}
    for label, results in all_results.items():
        analysis = analyze_model(results)
        all_analysis[label] = analysis

    # Print summary table
    print(f"\n{'='*80}")
    print("REASONING THEATER ANALYSIS (v2, re-graded)")
    print(f"{'='*80}")
    header = f"{'Model':<15} {'Acc':>5} {'Commit':>7} {'Theater':>8} {'CoT len':>8} {'BAEE sav':>9} {'Exit rate':>9}"
    print(header)
    print("-" * len(header))
    for label, a in all_analysis.items():
        acc = f"{a['accuracy']:.0%}"
        commit = f"{a['commitment_mean']:.0%}" if a["commitment_mean"] is not None else "---"
        theater = f"{a['theater_mean']:.0%}" if a["theater_mean"] is not None else "---"
        cot = f"{a['mean_cot_len']:.0f}"
        baee_s = f"{a['baee']['mean_savings']:.0%}" if a["baee"].get("mean_savings") else "---"
        baee_e = f"{a['baee']['early_exit_rate']:.0%}" if a["baee"].get("early_exit_rate") else "---"
        print(f"{label:<15} {acc:>5} {commit:>7} {theater:>8} {cot:>8} {baee_s:>9} {baee_e:>9}")

    # Print commitment by level
    print(f"\nCommitment by Level:")
    header2 = f"{'Model':<15} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8} {'L5':>8}"
    print(header2)
    print("-" * len(header2))
    for label, a in all_analysis.items():
        by_level = a.get("by_level", {})
        vals = []
        for lvl in [1, 2, 3, 4, 5]:
            info = by_level.get(lvl, {})
            if info and info.get("n", 0) > 0:
                vals.append(f"{info['commitment_mean']:.0%}({info['n']})")
            else:
                vals.append("---")
        print(f"{label:<15} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8}")

    # Save outputs
    os.makedirs(config.output_dir, exist_ok=True)

    # LaTeX tables
    latex_main = generate_latex_table(all_analysis)
    latex_level = generate_level_latex_table(all_analysis)
    latex_path = os.path.join(config.output_dir, "tables.tex")
    with open(latex_path, "w") as f:
        f.write("% Main results table\n")
        f.write(latex_main)
        f.write("\n\n% Commitment by level table\n")
        f.write(latex_level)
    print(f"\nLaTeX tables saved to {latex_path}")

    # Analysis JSON
    analysis_path = os.path.join(config.output_dir, "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(all_analysis, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"Analysis JSON saved to {analysis_path}")

    # Publication figure
    make_publication_figure(
        all_analysis, config.output_dir,
        save_pdf=config.save_pdf, save_png=config.save_png, show=config.show_plots,
    )


def main(config: AnalysisV2Config):
    run_analysis(config)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
