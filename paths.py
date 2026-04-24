"""Centralized path configuration for Reasoning Theater experiments.

Override defaults via environment variables:
  TINKER_COOKBOOK      — path to tinker-cookbook package
  REASONING_THEATER_RESULTS — root dir for experiment results (default: /tmp/...)
"""

import os

# ── External package ──────────────────────────────────────────────────────────

TINKER_COOKBOOK: str = os.environ.get(
    "TINKER_COOKBOOK",
    os.path.expanduser("~/tinker-cookbook"),
)

# ── Directories ───────────────────────────────────────────────────────────────

# Root of this repository (directory containing this file)
PAPER_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Where experiment results are written
RESULTS_ROOT: str = os.environ.get(
    "REASONING_THEATER_RESULTS",
    "/tmp/tinker-examples/reasoning_theater",
)

# Where generated figures are saved
FIGURES_DIR: str = os.path.join(PAPER_DIR, "figures")

# ── Helpers ───────────────────────────────────────────────────────────────────


def results(name: str) -> str:
    """Return path to a named results directory under RESULTS_ROOT."""
    return os.path.join(RESULTS_ROOT, name)


def results_jsonl(name: str) -> str:
    """Return path to results.jsonl inside a named results directory."""
    return os.path.join(RESULTS_ROOT, name, "results.jsonl")


def paper_data(subdir: str) -> str:
    """Return path to a local data directory inside the paper repo."""
    return os.path.join(PAPER_DIR, subdir, "results.jsonl")


def setup_path() -> None:
    """Ensure tinker-cookbook is importable, inserting it into sys.path if needed."""
    import sys

    try:
        import tinker_cookbook  # noqa: F401
    except ImportError:
        sys.path.insert(0, TINKER_COOKBOOK)
