"""Run a named Reasoning Theater experiment preset.

Replaces: run_32b_think.py, run_32b_nothink.py, run_8b_think.py,
          run_8b_nothink.py, run_gpt_oss_120b.py,
          run_gpqa_*.py, run_aime24_*.py, run_32b_think_shard450.py

Usage:
    python run_experiment.py --preset 32b-think
    python run_experiment.py --preset 32b-think --offset 450  # shard
    python run_experiment.py --list
"""

import argparse
import asyncio

import paths

paths.setup_path()

from experiment import (  # noqa: E402
    ExperimentConfig,
    _async_main,
)

_PF = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)

PRESETS: dict[str, ExperimentConfig] = {
    # ── MATH-500 ──────────────────────────────────────────────────────────────
    "32b-think": ExperimentConfig(
        model_name="Qwen/Qwen3-32B",
        renderer_name="qwen3",
        n_problems=500,
        prefix_fractions=_PF,
        output_dir=paths.results("qwen3_32b_thinking_full500"),
    ),
    "32b-nothink": ExperimentConfig(
        model_name="Qwen/Qwen3-32B",
        renderer_name="qwen3_disable_thinking",
        n_problems=500,
        prefix_fractions=_PF,
        output_dir=paths.results("qwen3_32b_no_thinking_full500"),
    ),
    "8b-think": ExperimentConfig(
        model_name="Qwen/Qwen3-8B",
        renderer_name="qwen3",
        n_problems=500,
        prefix_fractions=_PF,
        output_dir=paths.results("qwen3_8b_thinking_full500"),
    ),
    "8b-nothink": ExperimentConfig(
        model_name="Qwen/Qwen3-8B",
        renderer_name="qwen3_disable_thinking",
        n_problems=500,
        prefix_fractions=_PF,
        output_dir=paths.results("qwen3_8b_no_thinking_full500"),
    ),
    "gpt-oss": ExperimentConfig(
        model_name="openai/gpt-oss-120b",
        renderer_name="gpt_oss_medium_reasoning",
        n_problems=500,
        prefix_fractions=_PF,
        output_dir=paths.results("gpt_oss_120b_full500"),
    ),
    # ── GPQA-Diamond ──────────────────────────────────────────────────────────
    "gpqa-32b-think": ExperimentConfig(
        model_name="Qwen/Qwen3-32B",
        renderer_name="qwen3",
        benchmark="gpqa-diamond",
        n_problems=198,
        max_tokens=8192,
        prefix_fractions=_PF,
        output_dir=paths.results("gpqa_32b_think"),
    ),
    "gpqa-32b-nothink": ExperimentConfig(
        model_name="Qwen/Qwen3-32B",
        renderer_name="qwen3_disable_thinking",
        benchmark="gpqa-diamond",
        n_problems=198,
        max_tokens=4096,
        prefix_fractions=_PF,
        output_dir=paths.results("gpqa_32b_nothink"),
    ),
    "gpqa-8b-think": ExperimentConfig(
        model_name="Qwen/Qwen3-8B",
        renderer_name="qwen3",
        benchmark="gpqa-diamond",
        n_problems=198,
        max_tokens=4096,
        prefix_fractions=_PF,
        output_dir=paths.results("gpqa_8b_think"),
    ),
    "gpqa-8b-nothink": ExperimentConfig(
        model_name="Qwen/Qwen3-8B",
        renderer_name="qwen3_disable_thinking",
        benchmark="gpqa-diamond",
        n_problems=198,
        max_tokens=4096,
        prefix_fractions=_PF,
        output_dir=paths.results("gpqa_8b_nothink"),
    ),
    "gpqa-gpt-oss": ExperimentConfig(
        model_name="openai/gpt-oss-120b",
        renderer_name="gpt_oss_medium_reasoning",
        benchmark="gpqa-diamond",
        n_problems=198,
        max_tokens=8192,
        prefix_fractions=_PF,
        output_dir=paths.results("gpqa_gpt_oss_120b"),
    ),
    "gpqa-pilot": ExperimentConfig(
        model_name="Qwen/Qwen3-32B",
        renderer_name="qwen3",
        benchmark="gpqa-diamond",
        n_problems=5,
        max_tokens=8192,
        prefix_fractions=_PF,
        output_dir=paths.results("gpqa_pilot_32b_think"),
    ),
    # ── AIME-2024 ─────────────────────────────────────────────────────────────
    "aime-32b-think": ExperimentConfig(
        model_name="Qwen/Qwen3-32B",
        renderer_name="qwen3",
        benchmark="aime-2024",
        n_problems=30,
        max_tokens=8192,
        prefix_fractions=_PF,
        output_dir=paths.results("aime24_32b_think"),
    ),
    "aime-32b-nothink": ExperimentConfig(
        model_name="Qwen/Qwen3-32B",
        renderer_name="qwen3_disable_thinking",
        benchmark="aime-2024",
        n_problems=30,
        max_tokens=8192,
        prefix_fractions=_PF,
        output_dir=paths.results("aime24_32b_nothink"),
    ),
    "aime-8b-think": ExperimentConfig(
        model_name="Qwen/Qwen3-8B",
        renderer_name="qwen3",
        benchmark="aime-2024",
        n_problems=30,
        max_tokens=8192,
        prefix_fractions=_PF,
        output_dir=paths.results("aime24_8b_think"),
    ),
    "aime-8b-nothink": ExperimentConfig(
        model_name="Qwen/Qwen3-8B",
        renderer_name="qwen3_disable_thinking",
        benchmark="aime-2024",
        n_problems=30,
        max_tokens=8192,
        prefix_fractions=_PF,
        output_dir=paths.results("aime24_8b_nothink"),
    ),
    "aime-gpt-oss": ExperimentConfig(
        model_name="openai/gpt-oss-120b",
        renderer_name="gpt_oss_medium_reasoning",
        benchmark="aime-2024",
        n_problems=30,
        max_tokens=8192,
        prefix_fractions=_PF,
        output_dir=paths.results("aime24_gpt_oss"),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Reasoning Theater experiment preset.")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Experiment preset name.")
    parser.add_argument("--offset", type=int, default=0, help="Problem offset for sharding.")
    parser.add_argument("--list", action="store_true", help="List available presets and exit.")
    args = parser.parse_args()

    if args.list:
        print("Available presets:")
        for name in sorted(PRESETS):
            cfg = PRESETS[name]
            benchmark = getattr(cfg, "benchmark", "math-500")
            print(f"  {name:<22} benchmark={benchmark}, n={cfg.n_problems}")
        return

    if not args.preset:
        parser.error("--preset is required (or use --list)")

    config = PRESETS[args.preset]
    if args.offset:
        config = ExperimentConfig(
            **{**config.__dict__, "problem_offset": args.offset}
        )
    asyncio.run(_async_main(config))


if __name__ == "__main__":
    main()
