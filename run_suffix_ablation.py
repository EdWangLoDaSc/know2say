"""Run a named EFA suffix ablation preset.

Replaces: run_suffix_ablation.py (32B), run_suffix_ablation_8b_nothink.py,
          run_suffix_ablation_gpt_oss.py

Usage:
    python run_suffix_ablation.py --preset 32b-think
    python run_suffix_ablation.py --list
"""

import argparse
import asyncio

import paths

paths.setup_path()

from tinker_cookbook.recipes.reasoning_theater.efa_suffix_ablation import (  # noqa: E402
    SuffixAblationConfig,
    _async_main,
)

PRESETS: dict[str, SuffixAblationConfig] = {
    "32b-think": SuffixAblationConfig(
        model_name="Qwen/Qwen3-32B",
        renderer_name="qwen3",
        n_problems=100,
        output_dir=paths.results("suffix_ablation_32b"),
    ),
    "8b-nothink": SuffixAblationConfig(
        model_name="Qwen/Qwen3-8B",
        renderer_name="qwen3_disable_thinking",
        n_problems=100,
        max_tokens=4096,
        output_dir=paths.results("suffix_ablation_8b_nothink"),
    ),
    "gpt-oss": SuffixAblationConfig(
        model_name="openai/gpt-oss-120b",
        renderer_name="gpt_oss_medium_reasoning",
        n_problems=100,
        max_tokens=4096,
        output_dir=paths.results("suffix_ablation_gpt_oss"),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an EFA suffix ablation preset.")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Preset name.")
    parser.add_argument("--list", action="store_true", help="List available presets and exit.")
    args = parser.parse_args()

    if args.list:
        print("Available presets:")
        for name in sorted(PRESETS):
            cfg = PRESETS[name]
            print(f"  {name:<14} model={cfg.model_name}, n={cfg.n_problems}")
        return

    if not args.preset:
        parser.error("--preset is required (or use --list)")

    asyncio.run(_async_main(PRESETS[args.preset]))


if __name__ == "__main__":
    main()
