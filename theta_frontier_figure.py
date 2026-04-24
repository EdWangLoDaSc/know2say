"""
Aesthetically optimized theta-sweep figure for PSC-triggered BAEE.

This script produces:
  1) A publication-style 2-panel figure
     (A) Savings–proxy-FP frontier on NoThink models
     (B) Token savings across PSC thresholds for all models

  2) A JSON dump of the sweep results for traceability

Design goals:
- cleaner paper-style aesthetics
- clearer panel titles
- lighter grid / reduced dashboard feel
- consistent highlighting of chosen operating points
- safer annotation placement
- code comments aligned with actual plotted content
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required to run this script") from exc


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: str
    chosen_theta: float
    color: str
    marker: str


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        label="32B-Think",
        path="/tmp/tinker-examples/reasoning_theater/qwen3_32b_thinking/results.jsonl",
        chosen_theta=0.75,
        color="#35608D",   # muted blue
        marker="o",
    ),
    ModelSpec(
        label="32B-NoThink",
        path="/tmp/tinker-examples/reasoning_theater/qwen3_32b_no_thinking_full500/results.jsonl",
        chosen_theta=0.875,
        color="#A64B4B",   # muted red
        marker="s",
    ),
    ModelSpec(
        label="8B-Think",
        path="/tmp/tinker-examples/reasoning_theater/qwen3_8b_thinking_full500/results.jsonl",
        chosen_theta=0.75,
        color="#6C5B9A",   # muted purple
        marker="^",
    ),
    ModelSpec(
        label="8B-NoThink",
        path="/tmp/tinker-examples/reasoning_theater/qwen3_8b_no_thinking_full500/results.jsonl",
        chosen_theta=0.875,
        color="#C17C3A",   # muted orange
        marker="D",
    ),
    ModelSpec(
        label="GPT-OSS-120B",
        path="/tmp/tinker-examples/reasoning_theater/gpt_oss_120b_full500/results.jsonl",
        chosen_theta=0.75,
        color="#3E8E7E",   # muted teal
        marker="P",
    ),
]

# PSC-8 granularity
THRESHOLDS = [i / 8 for i in range(1, 9)]

# Output directory
OUT_DIR = "/tmp/tinker-examples/reasoning_theater/paper_figures_v4"

# Manual annotation offsets to reduce overlap.
# Keys are (panel_name, model_label)
ANNOTATION_OFFSETS: dict[tuple[str, str], tuple[int, int]] = {
    ("frontier", "32B-NoThink"): (6, 4),
    ("frontier", "8B-NoThink"): (6, -11),
    ("savings", "32B-Think"): (4, 6),
    ("savings", "32B-NoThink"): (4, -12),
    ("savings", "8B-Think"): (4, 6),
    ("savings", "8B-NoThink"): (4, -12),
    ("savings", "GPT-OSS-120B"): (4, 6),
}


# -----------------------------
# Data loading and sweep logic
# -----------------------------

def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sweep_metrics(results: list[dict], theta: float) -> dict:
    """
    Offline BAEE simulation under a threshold theta.

    Notes:
    - baseline_accuracy = accuracy of the selected full rollout
    - proxy FP uses the 0/4-correct subset as a proxy-defined wrong set
    - savings are measured relative to selected_rollout_len
    """
    n = len(results)
    if n == 0:
        raise ValueError("Empty results list")

    base_acc = sum(1 for r in results if r["selected_rollout_correct"]) / n

    n_correct = 0
    savings: list[float] = []

    wrong = [r for r in results if r["n_correct_rollouts"] == 0]
    fp = 0

    for r in results:
        full_len = r["selected_rollout_len"]
        exited = False

        for pr in r["prefix_results"]:
            if pr["psc_agreement_rate"] >= theta:
                exited = True

                # In this offline simulation, if the problem is solvable by any rollout,
                # we treat PSC-trigger as recovering a correct answer.
                if r["n_correct_rollouts"] > 0:
                    n_correct += 1

                if full_len > 0:
                    savings.append(max(0.0, 1.0 - pr["prefix_len"] / full_len))
                else:
                    savings.append(0.0)
                break

        if not exited:
            if r["selected_rollout_correct"]:
                n_correct += 1
            savings.append(0.0)

    for r in wrong:
        if any(pr["psc_agreement_rate"] >= theta for pr in r["prefix_results"]):
            fp += 1

    acc = n_correct / n
    fp_rate = (fp / len(wrong)) if wrong else 0.0

    return {
        "theta": theta,
        "accuracy": acc,
        "accuracy_delta": acc - base_acc,
        "mean_savings": float(np.mean(savings)) if savings else 0.0,
        "proxy_fp_rate": fp_rate,
        "proxy_fp_num": fp,
        "proxy_wrong_num": len(wrong),
        "baseline_accuracy": base_acc,
        "n_total": n,
    }


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_curve_point(curve: list[dict], theta: float) -> dict:
    for p in curve:
        if abs(p["theta"] - theta) < 1e-12:
            return p
    raise KeyError(f"Theta={theta} not found in curve")


# -----------------------------
# Plotting helpers
# -----------------------------

def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "grid.linewidth": 0.6,
        }
    )


def style_axis(ax) -> None:
    ax.grid(True, alpha=0.18)
    ax.set_axisbelow(True)


def annotate_theta(ax, x: float, y: float, theta: float, panel: str, label: str) -> None:
    dx, dy = ANNOTATION_OFFSETS.get((panel, label), (5, 4))
    ax.annotate(
        f"θ={theta:g}",
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        fontsize=7.5,
    )


def percent(values: Iterable[float]) -> list[float]:
    return [100.0 * v for v in values]


def plot_frontier_panel(ax, all_curves: dict[str, list[dict]]) -> None:
    """
    Panel A: Savings–proxy-FP trade-off for NoThink models.
    """
    for spec in MODEL_SPECS:
        if "NoThink" not in spec.label:
            continue

        curve = all_curves[spec.label]
        xs = percent(p["mean_savings"] for p in curve)
        ys = percent(p["proxy_fp_rate"] for p in curve)

        ax.plot(
            xs,
            ys,
            color=spec.color,
            marker=spec.marker,
            linewidth=1.6,
            markersize=4.2,
            label=spec.label,
        )

        chosen = get_curve_point(curve, spec.chosen_theta)
        x_chosen = 100.0 * chosen["mean_savings"]
        y_chosen = 100.0 * chosen["proxy_fp_rate"]

        ax.scatter(
            [x_chosen],
            [y_chosen],
            color=spec.color,
            s=42,
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )
        annotate_theta(ax, x_chosen, y_chosen, spec.chosen_theta, "frontier", spec.label)

    ax.set_title("A. Savings–FP trade-off on proxy-wrong subset")
    ax.set_xlabel("Mean token savings (%)")
    ax.set_ylabel("Proxy FP rate (%)")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left")

    # Small margin padding for cleaner appearance
    ax.margins(x=0.06, y=0.12)


def plot_savings_theta_panel(ax, all_curves: dict[str, list[dict]]) -> None:
    """
    Panel B: Savings vs theta for all models.
    """
    for spec in MODEL_SPECS:
        curve = all_curves[spec.label]
        xs = [p["theta"] for p in curve]
        ys = percent(p["mean_savings"] for p in curve)

        alpha = 1.0 if "NoThink" in spec.label else 0.92

        ax.plot(
            xs,
            ys,
            color=spec.color,
            marker=spec.marker,
            linewidth=1.5,
            markersize=3.8,
            alpha=alpha,
            label=spec.label,
        )

        chosen = get_curve_point(curve, spec.chosen_theta)
        x_chosen = chosen["theta"]
        y_chosen = 100.0 * chosen["mean_savings"]

        ax.scatter(
            [x_chosen],
            [y_chosen],
            color=spec.color,
            s=38,
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )
        annotate_theta(ax, x_chosen, y_chosen, spec.chosen_theta, "savings", spec.label)

    ax.set_title("B. Token savings across PSC thresholds")
    ax.set_xlabel("PSC threshold θ")
    ax.set_ylabel("Mean token savings (%)")
    ax.set_xticks(THRESHOLDS)
    ax.set_xticklabels(
        [f"{t:.3f}".rstrip("0").rstrip(".") for t in THRESHOLDS],
        rotation=25,
    )
    style_axis(ax)
    ax.legend(frameon=False, loc="lower left")
    ax.margins(x=0.03, y=0.10)


def save_json(all_curves: dict[str, list[dict]], out_dir: str) -> str:
    json_path = os.path.join(out_dir, "theta_frontier_sweep.json")
    ensure_parent(json_path)
    with open(json_path, "w") as f:
        json.dump(
            {
                "thresholds": THRESHOLDS,
                "models": all_curves,
                "model_specs": [spec.__dict__ for spec in MODEL_SPECS],
            },
            f,
            indent=2,
        )
    return json_path


def make_figure(all_curves: dict[str, list[dict]], out_dir: str) -> tuple[str, str]:
    configure_matplotlib()

    # Compact but paper-friendly aspect ratio
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    plot_frontier_panel(axes[0], all_curves)
    plot_savings_theta_panel(axes[1], all_curves)

    fig.suptitle("PSC Threshold Sweep", y=0.98, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = os.path.join(out_dir, "theta_frontier_optimized.png")
    pdf_path = os.path.join(out_dir, "theta_frontier_optimized.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    all_curves: dict[str, list[dict]] = {}
    for spec in MODEL_SPECS:
        if not os.path.exists(spec.path):
            raise FileNotFoundError(f"Missing results file for {spec.label}: {spec.path}")

        rows = load_results(spec.path)
        all_curves[spec.label] = [sweep_metrics(rows, th) for th in THRESHOLDS]

    json_path = save_json(all_curves, OUT_DIR)
    png_path, pdf_path = make_figure(all_curves, OUT_DIR)

    print(f"Saved sweep JSON: {json_path}")
    print(f"Saved figure PNG: {png_path}")
    print(f"Saved figure PDF: {pdf_path}")
    print("\nChosen operating points:")
    for spec in MODEL_SPECS:
        chosen = get_curve_point(all_curves[spec.label], spec.chosen_theta)
        print(
            f"  {spec.label:12s} "
            f"theta={spec.chosen_theta:.3f} | "
            f"acc={chosen['accuracy']:.3f} "
            f"(delta {chosen['accuracy_delta']:+.3f}) | "
            f"savings={chosen['mean_savings']:.3f} | "
            f"proxy_fp={chosen['proxy_fp_num']}/{chosen['proxy_wrong_num']}"
        )


if __name__ == "__main__":
    main()