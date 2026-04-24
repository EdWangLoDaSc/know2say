"""
EFA Suffix Ablation Experiment.

Tests whether the detection--extraction gap is robust across different EFA forcing
templates.  For each prefix fraction we run EFA with 5 different suffixes and compare
extraction accuracy against PSC accuracy.

The key question: Is the gap between PSC accuracy (~92% at 10%) and EFA accuracy (~47%)
specific to the original suffix, or does it persist across all suffix choices?

If the gap is robust → supports distribution-shift / truncation-artifact explanation.
If only one suffix shows it → supports format-mismatch explanation.

Usage:
    python -m tinker_cookbook.recipes.reasoning_theater.efa_suffix_ablation
    python -m tinker_cookbook.recipes.reasoning_theater.efa_suffix_ablation \
        model_name=Qwen/Qwen3-32B n_problems=50
"""

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass, field

import chz
import tinker
import tinker.types as types

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import MathEnv
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed
from tinker_cookbook.recipes.reasoning_theater.experiment import (
    _load_math500,
    safe_grade,
    try_extract_and_grade,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EFA suffix definitions — 5 variants to test
# ---------------------------------------------------------------------------

EFA_SUFFIXES: dict[str, str] = {
    "original":  "\nTherefore, the final answer is \\boxed{",
    "natural":   "\nThe answer is \\boxed{",
    "soft":      "\nSo the answer is \\boxed{",
    "plain":     "\nAnswer: ",
    "direct":    "\n\\boxed{",
}

# Which stop token to use for each suffix
_SUFFIX_STOP: dict[str, list[str]] = {
    "original": ["}"],
    "natural":  ["}"],
    "soft":     ["}"],
    "plain":    ["\n"],
    "direct":   ["}"],
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@chz.chz
class SuffixAblationConfig:
    base_url: str | None = None
    model_name: str = "Qwen/Qwen3-32B"
    renderer_name: str | None = None
    n_problems: int = 100
    n_psc_samples: int = 8
    max_tokens: int = 4096
    prefix_fractions: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    efa_max_tokens: int = 64
    grader_timeout: float = 2.0
    output_dir: str = "/tmp/tinker-examples/reasoning_theater"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CheckpointResult:
    fraction: float
    suffix_name: str
    efa_answer: str | None
    efa_correct: bool


@dataclass
class ProblemAblationResult:
    problem_idx: int
    ground_truth: str
    rollout_correct: bool
    rollout_len: int
    psc_accuracy: dict[float, float]
    suffix_results: list[CheckpointResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EFA with arbitrary suffix
# ---------------------------------------------------------------------------


async def _run_efa_suffix(
    sampling_client: tinker.SamplingClient,
    prompt_mi: types.ModelInput,
    prefix_tokens: list[int],
    suffix_name: str,
    suffix_text: str,
    ground_truth: str,
    tokenizer,
    config: SuffixAblationConfig,
) -> tuple[str | None, bool]:
    suffix_toks = tokenizer.encode(suffix_text, add_special_tokens=False)
    combined = list(prefix_tokens) + list(suffix_toks)
    forced_mi = prompt_mi.append(types.EncodedTextChunk(tokens=combined))

    sp = types.SamplingParams(
        max_tokens=config.efa_max_tokens,
        temperature=0.0,
        stop=_SUFFIX_STOP[suffix_name],
    )
    result = await sampling_client.sample_async(prompt=forced_mi, num_samples=1, sampling_params=sp)
    raw = tokenizer.decode(result.sequences[0].tokens).strip().rstrip("}.").strip()
    if not raw:
        return None, False
    correct = safe_grade(raw, ground_truth, "sympy", config.grader_timeout)
    return raw, correct


# ---------------------------------------------------------------------------
# PSC at a single prefix
# ---------------------------------------------------------------------------


async def _run_psc(
    sampling_client: tinker.SamplingClient,
    prompt_mi: types.ModelInput,
    prefix_tokens: list[int],
    ground_truth: str,
    renderer,
    tokenizer,
    config: SuffixAblationConfig,
    total_cot_len: int | None = None,
) -> float:
    probe_mi = prompt_mi.append(types.EncodedTextChunk(tokens=list(prefix_tokens)))
    if total_cot_len is not None:
        remaining_actual = max(total_cot_len - len(prefix_tokens), 1)
        remaining = max(min(remaining_actual * 2, config.max_tokens), 256)
    else:
        remaining = max(config.max_tokens - len(prefix_tokens), 128)
    sp = types.SamplingParams(
        max_tokens=remaining,
        stop=renderer.get_stop_sequences(),
        temperature=1.0,
    )
    result = await sampling_client.sample_async(
        prompt=probe_mi,
        num_samples=config.n_psc_samples,
        sampling_params=sp,
    )
    n_correct = 0
    for seq in result.sequences:
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        content = renderers.get_text_content(parsed_msg)
        correct, _ = try_extract_and_grade(content, ground_truth, "sympy", config.grader_timeout)
        if correct:
            n_correct += 1
    return n_correct / config.n_psc_samples


# ---------------------------------------------------------------------------
# Per-problem processing
# ---------------------------------------------------------------------------


async def _process_problem(
    problem: dict,
    prompt_mi: types.ModelInput,
    sampling_client: tinker.SamplingClient,
    renderer,
    tokenizer,
    config: SuffixAblationConfig,
) -> ProblemAblationResult:
    ground_truth = problem["answer"]
    idx = problem["idx"]

    # Sample one full rollout to obtain CoT tokens
    sp = types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=1.0,
    )
    rollout = await sampling_client.sample_async(prompt=prompt_mi, num_samples=1, sampling_params=sp)
    cot_tokens = rollout.sequences[0].tokens
    parsed_msg, _ = renderer.parse_response(cot_tokens)
    content = renderers.get_text_content(parsed_msg)
    rollout_correct, _ = try_extract_and_grade(content, ground_truth, "sympy", config.grader_timeout)

    total_len = len(cot_tokens)
    logger.info(f"Problem {idx}: len={total_len}, correct={rollout_correct}")

    result = ProblemAblationResult(
        problem_idx=idx,
        ground_truth=ground_truth,
        rollout_correct=rollout_correct,
        rollout_len=total_len,
        psc_accuracy={},
    )

    if total_len < 10:
        return result

    async def _process_fraction(frac: float) -> tuple[float, float, list[CheckpointResult]]:
        prefix_len = max(1, min(int(frac * total_len), total_len))
        prefix_tokens = cot_tokens[:prefix_len]

        psc_coro = _run_psc(
            sampling_client, prompt_mi, prefix_tokens, ground_truth, renderer, tokenizer, config,
            total_cot_len=total_len,
        )
        efa_coros = [
            _run_efa_suffix(
                sampling_client, prompt_mi, prefix_tokens, sn, st, ground_truth, tokenizer, config
            )
            for sn, st in EFA_SUFFIXES.items()
        ]
        psc, *efa_results = await asyncio.gather(psc_coro, *efa_coros)

        crs = [
            CheckpointResult(fraction=frac, suffix_name=sn, efa_answer=ans, efa_correct=correct)
            for (sn, _), (ans, correct) in zip(EFA_SUFFIXES.items(), efa_results)
        ]
        efa_bits = "  ".join(
            f"{sn}={'✓' if cr.efa_correct else '✗'}" for sn, cr in zip(EFA_SUFFIXES, crs)
        )
        logger.info(f"  frac={frac:.0%}: PSC={psc:.2f}  {efa_bits}")
        return frac, psc, crs

    fraction_outputs = await asyncio.gather(*[_process_fraction(f) for f in config.prefix_fractions])

    for frac, psc, crs in fraction_outputs:
        result.psc_accuracy[frac] = psc
        result.suffix_results.extend(crs)

    return result


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def compute_report(results: list[ProblemAblationResult]) -> dict:
    fracs = sorted({frac for r in results for frac in r.psc_accuracy})
    snames = list(EFA_SUFFIXES.keys())

    report: dict = {
        "n_problems": len(results),
        "fractions": fracs,
        "psc_accuracy": {},
        "efa_accuracy": {sn: {} for sn in snames},
        "gap_vs_psc": {sn: {} for sn in snames},
    }

    for frac in fracs:
        psc_vals = [r.psc_accuracy[frac] for r in results if frac in r.psc_accuracy]
        report["psc_accuracy"][frac] = sum(psc_vals) / len(psc_vals) if psc_vals else float("nan")

        for sn in snames:
            flags = [
                sc.efa_correct
                for r in results
                for sc in r.suffix_results
                if sc.fraction == frac and sc.suffix_name == sn
            ]
            if flags:
                efa_acc = sum(flags) / len(flags)
                report["efa_accuracy"][sn][frac] = efa_acc
                report["gap_vs_psc"][sn][frac] = report["psc_accuracy"][frac] - efa_acc

    return report


def print_report(report: dict) -> None:
    fracs = report["fractions"]
    print("\n" + "=" * 72)
    print(f"EFA Suffix Ablation  (n={report['n_problems']} problems)")
    print("=" * 72)
    hdr = f"{'Metric':<32}" + "".join(f"{f:.0%}".rjust(8) for f in fracs)
    print(hdr)
    print("-" * 72)
    print(
        f"{'PSC accuracy':<32}"
        + "".join(
            f"{report['psc_accuracy'].get(f, float('nan')):.2f}".rjust(8) for f in fracs
        )
    )
    print()
    for sn in EFA_SUFFIXES:
        print(
            f"  EFA [{sn:<10}] acc    "
            + "".join(
                f"{report['efa_accuracy'][sn].get(f, float('nan')):.2f}".rjust(8)
                for f in fracs
            )
        )
        print(
            f"  EFA [{sn:<10}] gap    "
            + "".join(
                f"{report['gap_vs_psc'][sn].get(f, float('nan')):.2f}".rjust(8)
                for f in fracs
            )
        )
        print()
    print("=" * 72)
    print(
        "Key: gap > 0.15 for ALL suffixes at 10% => gap is suffix-robust (strong claim)."
    )
    print(
        "     gap only for some suffixes        => format-mismatch explanation more likely."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _async_main(config: SuffixAblationConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(config.output_dir, exist_ok=True)

    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model_name)
    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    service_client = tinker.ServiceClient(base_url=config.base_url)
    sampling_client = service_client.create_sampling_client(base_model=config.model_name)

    problems = _load_math500(config.n_problems)
    logger.info(f"Loaded {len(problems)} problems. Model={config.model_name}")

    fewshot = MathEnv.standard_fewshot_prefix()
    all_results: list[ProblemAblationResult] = []

    for pi, problem in enumerate(problems):
        question = problem["problem"] + MathEnv.question_suffix()
        convo = [*fewshot, {"role": "user", "content": question}]
        prompt_mi = renderer.build_generation_prompt(convo)

        logger.info(f"[{pi+1}/{len(problems)}] Problem {problem['idx']}")
        try:
            r = await _process_problem(
                problem, prompt_mi, sampling_client, renderer, tokenizer, config
            )
            all_results.append(r)
        except Exception as exc:
            logger.error(f"  FAILED: {exc}")

    report = compute_report(all_results)
    print_report(report)

    # Save full results
    out = os.path.join(config.output_dir, "suffix_ablation_results.json")

    def _default(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        raise TypeError(type(obj))

    with open(out, "w") as f:
        json.dump(
            {
                "config": {"model_name": config.model_name, "n_problems": config.n_problems},
                "report": report,
                "per_problem": [
                    {
                        "problem_idx": r.problem_idx,
                        "rollout_correct": r.rollout_correct,
                        "rollout_len": r.rollout_len,
                        "psc_accuracy": {str(k): v for k, v in r.psc_accuracy.items()},
                        "suffix_results": [
                            {
                                "fraction": sc.fraction,
                                "suffix_name": sc.suffix_name,
                                "efa_correct": sc.efa_correct,
                                "efa_answer": sc.efa_answer,
                            }
                            for sc in r.suffix_results
                        ],
                    }
                    for r in all_results
                ],
            },
            f,
            indent=2,
            default=_default,
        )
    logger.info(f"Results saved → {out}")


@chz.entrypoint
def main(config: SuffixAblationConfig) -> None:
    asyncio.run(_async_main(config))
