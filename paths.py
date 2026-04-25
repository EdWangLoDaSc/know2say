"""Centralized path configuration for Know2Say experiments.

Override defaults via environment variables:
  TINKER_COOKBOOK      — path to a tinker-cookbook checkout (defaults to the
                          vendored copy under ``vendor/`` inside this repo).
  REASONING_THEATER_RESULTS — root dir for experiment results (default: /tmp/...)
"""

import os

# ── External package ──────────────────────────────────────────────────────────

# The repo ships a frozen copy of `tinker_cookbook` under `vendor/` so that
# `git clone` is enough to run the figure-regeneration scripts without
# additional setup.  Set TINKER_COOKBOOK if you'd rather use a system install.
TINKER_COOKBOOK: str = os.environ.get(
    "TINKER_COOKBOOK",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"),
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
