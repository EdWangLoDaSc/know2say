"""Centralized path configuration for Know2Say experiments.

Layout (after the repo cleanup):

    <REPO>/
        src/         — all .py code (this file lives here)
        results/     — experiment-result jsonl folders + analysis.json
        data/        — raw benchmark inputs (HumanEval, GPQA, …)
        vendor/      — frozen copy of tinker_cookbook
        figures/     — generated figures (gitignored, recreated on demand)

Override defaults via environment variables:
  TINKER_COOKBOOK            — path to a tinker-cookbook checkout
                                (defaults to ``<repo>/vendor``)
  REASONING_THEATER_RESULTS  — root dir for *new* experiment outputs
                                (defaults to /tmp/tinker-examples/reasoning_theater)
"""

import os

# ── Directories ───────────────────────────────────────────────────────────────

# This file lives at <REPO>/src/paths.py, so the repo root is one level up.
SRC_DIR:    str = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR:  str = os.path.dirname(SRC_DIR)

# Where experiment-result JSONL folders are checked into the repo
DATA_DIR:    str = os.path.join(PAPER_DIR, "data")
RESULTS_DIR: str = os.path.join(PAPER_DIR, "results")
FIGURES_DIR: str = os.path.join(PAPER_DIR, "figures")
VENDOR_DIR:  str = os.path.join(PAPER_DIR, "vendor")

# Where *new* experiment outputs are written by the runner scripts
RESULTS_ROOT: str = os.environ.get(
    "REASONING_THEATER_RESULTS",
    "/tmp/tinker-examples/reasoning_theater",
)

# ── External package ──────────────────────────────────────────────────────────

# The repo ships a frozen copy of `tinker_cookbook` under `vendor/` so that
# `git clone` is enough to run the figure-regeneration scripts without
# additional setup.  Set TINKER_COOKBOOK if you'd rather use a system install.
TINKER_COOKBOOK: str = os.environ.get("TINKER_COOKBOOK", VENDOR_DIR)


# ── Helpers ───────────────────────────────────────────────────────────────────


def results(name: str) -> str:
    """Return path to a named experiment-output directory under RESULTS_ROOT.

    Used by runner scripts that write *new* results.
    """
    return os.path.join(RESULTS_ROOT, name)


def results_jsonl(name: str) -> str:
    """Return path to results.jsonl inside a named output directory."""
    return os.path.join(RESULTS_ROOT, name, "results.jsonl")


def paper_data(subdir: str) -> str:
    """Return path to a checked-in results.jsonl under ``results/<subdir>/``."""
    return os.path.join(RESULTS_DIR, subdir, "results.jsonl")


def setup_path() -> None:
    """Make ``tinker_cookbook`` importable, preferring the vendored copy."""
    import sys

    try:
        import tinker_cookbook  # noqa: F401
    except ImportError:
        sys.path.insert(0, TINKER_COOKBOOK)
