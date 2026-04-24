"""Run a named prefix-free baseline experiment preset.

Replaces: run_baselines_8b_think.py, run_baselines_8b_nothink.py,
          run_baselines_32b_think.py, run_baselines_32b_nothink.py,
          run_baselines_gpt_oss.py, run_baselines_gpqa_*.py

Usage:
    python run_baselines.py --preset 32b-think
    python run_baselines.py --list
"""

import argparse
import asyncio

import paths

paths.setup_path()

from tinker_cookbook.recipes.reasoning_theater.baseline_experiment import (  # noqa: E402
    BaselineConfig,
    _async_main,
)

PRESETS: dict[str, BaselineConfig] = {
    # ── MATH-500 ──────────────────────────────────────────────────────────────
    "32b-think": BaselineConfig(
        model_name="Qwen/Qwen3-32B",
        renderer_name="qwen3",
        n_problems=500,
        max_tokens=4096,
        main_results_path=paths.results_jsonl("qwen3_32b_thinking_full500"),
        output_dir=paths.results("baselines_32b_think"),
    ),
    "32b-nothink": BaselineConfig(
        model_name="Qwen/Qwen3-32B",
        renderer_name="qwen3_disable_thinking",
        n_problems=500,
        max_tokens=4096,
        main_results_path=paths.results_jsonl("qwen3_32b_no_thinking_full500"),
        output_dir=paths.results("baselines_32b_nothink"),
    ),
    "8b-think": BaselineConfig(
        model_name="Qwen/Qwen3-8B",
        renderer_name="qwen3",
        n_problems=500,
        max_tokens=4096,
        main_results_path=paths.results_jsonl("qwen3_8b_thinking_full500"),
        output_dir=paths.results("baselines_8b_think"),
    ),
    "8b-nothink": BaselineConfig(
        model_name="Qwen/Qwen3-8B",
        renderer_name="qwen3_disable_thinking",
        n_problems=500,
        max_tokens=4096,
        main_results_path=paths.results_jsonl("qwen3_8b_no_thinking_full500"),
        output_dir=paths.results("baselines_8b_nothink"),
    ),
    "gpt-oss": BaselineConfig(
        model_name="openai/gpt-oss-120b",
        renderer_name="gpt_oss_medium_reasoning",
        n_problems=500,
        max_tokens=4096,
        main_results_path=paths.results_jsonl("gpt_oss_120b_full500"),
        output_dir=paths.results("baselines_gpt_oss"),
    ),
    # ── GPQA-Diamond ──────────────────────────────────────────────────────────
    "gpqa-32b-think": BaselineConfig(
        model_name="Qwen/Qwen3-32B",
        renderer_name="qwen3",
        benchmark="gpqa-diamond",
        n_problems=198,
        max_tokens=8192,
        main_results_path=paths.results_jsonl("gpqa_32b_think"),
        output_dir=paths.results("baselines_gpqa_32b_think"),
    ),
    "gpqa-8b-think": BaselineConfig(
        model_name="Qwen/Qwen3-8B",
        renderer_name="qwen3",
        benchmark="gpqa-diamond",
        n_problems=198,
        max_tokens=8192,
        main_results_path=paths.results_jsonl("gpqa_8b_think"),
        output_dir=paths.results("baselines_gpqa_8b_think"),
    ),
    "gpqa-8b-nothink": BaselineConfig(
        model_name="Qwen/Qwen3-8B",
        renderer_name="qwen3_disable_thinking",
        benchmark="gpqa-diamond",
        n_problems=198,
        max_tokens=4096,
        main_results_path=paths.results_jsonl("gpqa_8b_nothink"),
        output_dir=paths.results("baselines_gpqa_8b_nothink"),
    ),
    "gpqa-gpt-oss": BaselineConfig(
        model_name="openai/gpt-oss-120b",
        renderer_name="gpt_oss_medium_reasoning",
        benchmark="gpqa-diamond",
        n_problems=198,
        max_tokens=8192,
        main_results_path=paths.results_jsonl("gpqa_gpt_oss_120b"),
        output_dir=paths.results("baselines_gpqa_gpt_oss"),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline experiment preset.")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Baseline preset name.")
    parser.add_argument("--list", action="store_true", help="List available presets and exit.")
    args = parser.parse_args()

    if args.list:
        print("Available presets:")
        for name in sorted(PRESETS):
            cfg = PRESETS[name]
            benchmark = getattr(cfg, "benchmark", "math-500")
            print(f"  {name:<24} benchmark={benchmark}, n={cfg.n_problems}")
        return

    if not args.preset:
        parser.error("--preset is required (or use --list)")

    asyncio.run(_async_main(PRESETS[args.preset]))


if __name__ == "__main__":
    main()
