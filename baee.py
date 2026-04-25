"""
BAEE: Black-box Adaptive Early Exit for Chain-of-Thought reasoning.

Two exit strategies:
  - EFA-oracle: Exit when EFA returns the correct answer (requires ground truth; upper bound)
  - PSC-triggered: Exit when PSC agreement >= threshold (no ground truth; deployable)

Two modes:
  A) Simulation — uses existing experiment JSONL data to estimate savings
  B) Online — deployable pipeline using Tinker API for real-time early exit

Usage:
    # EFA-oracle simulation (upper bound)
    python -m baee mode=simulate strategy=efa_oracle

    # PSC-triggered simulation (deployable)
    python -m baee mode=simulate strategy=psc

    # Online mode (requires TINKER_API_KEY)
    python -m baee mode=online n_problems=5
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Literal

import chz
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@chz.chz
class BAEEConfig:
    mode: Literal["simulate", "online"] = "simulate"
    strategy: Literal["efa_oracle", "psc"] = "psc"
    # Checkpoint fractions where probes are run
    chunk_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    # EFA parameters
    efa_max_tokens: int = 64
    efa_suffix: str = "\nTherefore, the final answer is \\boxed{"
    # PSC parameters
    psc_threshold: float = 0.75  # agreement rate to trigger exit
    psc_n_samples: int = 16
    # Full generation parameters
    max_tokens: int = 4096
    # Simulation mode
    results_path: str = "/tmp/tinker-examples/reasoning_theater/qwen3_32b_thinking/results.jsonl"
    # Online mode
    base_url: str | None = None
    model_name: str = "Qwen/Qwen3-32B"
    renderer_name: str | None = None
    n_problems: int = 5
    # Grading
    grader: Literal["sympy", "math_verify"] = "sympy"
    grader_timeout: float = 2.0
    # Output
    output_dir: str = "/tmp/tinker-examples/reasoning_theater/baee_results"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BAEEResult:
    """Result of BAEE on a single problem."""
    problem_idx: int
    answer: str | None
    is_correct: bool
    tokens_generated: int  # tokens actually used
    tokens_full_cot: int  # tokens that full CoT would have used
    tokens_saved: int
    savings_fraction: float
    exited_early: bool
    exit_fraction: float  # fraction of CoT at which exit occurred (1.0 = no exit)
    strategy: str = "efa_oracle"
    # Per-checkpoint details
    checkpoint_details: list[dict] = field(default_factory=list)


@dataclass
class BAEEReport:
    """Aggregate report over multiple problems."""
    n_problems: int
    n_correct: int
    n_early_exit: int
    mean_savings: float
    median_savings: float
    mean_exit_fraction: float
    total_tokens_used: int
    total_tokens_full: int
    total_tokens_saved: int
    overall_savings_pct: float
    accuracy: float
    early_exit_rate: float
    strategy: str = "efa_oracle"
    # Comparison with full CoT
    full_cot_accuracy: float | None = None
    per_problem: list[BAEEResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _regrade_efa_answer(answer: str | None, ground_truth: str) -> tuple[str | None, bool]:
    """Re-grade an EFA answer with fixed stripping."""
    if not answer:
        return None, False
    cleaned = answer.strip().rstrip("}.").strip()
    if not cleaned:
        return None, False
    try:
        from tinker_cookbook.recipes.math_rl.math_grading import grade_answer
        return cleaned, grade_answer(cleaned, ground_truth)
    except Exception:
        return cleaned, False


def _build_report(
    per_problem: list[BAEEResult],
    n_full_cot_correct: int,
    n_total_results: int,
    strategy: str,
) -> BAEEReport:
    """Build aggregate report from per-problem results."""
    n_correct = sum(1 for p in per_problem if p.is_correct)
    n_early = sum(1 for p in per_problem if p.exited_early)
    savings_list = [p.savings_fraction for p in per_problem]
    exit_fracs = [p.exit_fraction for p in per_problem]
    total_used = sum(p.tokens_generated for p in per_problem)
    total_full = sum(p.tokens_full_cot for p in per_problem)
    total_saved = total_full - total_used

    return BAEEReport(
        n_problems=len(per_problem),
        n_correct=n_correct,
        n_early_exit=n_early,
        mean_savings=float(np.mean(savings_list)) if savings_list else 0,
        median_savings=float(np.median(savings_list)) if savings_list else 0,
        mean_exit_fraction=float(np.mean(exit_fracs)) if exit_fracs else 1.0,
        total_tokens_used=total_used,
        total_tokens_full=total_full,
        total_tokens_saved=total_saved,
        overall_savings_pct=total_saved / total_full if total_full > 0 else 0,
        accuracy=n_correct / len(per_problem) if per_problem else 0,
        early_exit_rate=n_early / len(per_problem) if per_problem else 0,
        strategy=strategy,
        full_cot_accuracy=n_full_cot_correct / n_total_results if n_total_results else 0,
        per_problem=per_problem,
    )


# ---------------------------------------------------------------------------
# Strategy: EFA Oracle (requires ground truth — upper bound)
# ---------------------------------------------------------------------------


def simulate_efa_oracle(results: list[dict]) -> BAEEReport:
    """EFA-oracle BAEE: exit when EFA returns the correct answer.

    This requires ground truth and serves as an upper bound on savings.
    Only counts solvable problems (n_correct_rollouts > 0).
    """
    per_problem: list[BAEEResult] = []
    n_full_cot_correct = 0

    for r in results:
        if r["selected_rollout_correct"]:
            n_full_cot_correct += 1
        if r["n_correct_rollouts"] == 0:
            continue

        gt = r["ground_truth"]
        full_cot_len = r["selected_rollout_len"]
        exited = False

        for pr in r["prefix_results"]:
            cleaned_ans, is_correct = _regrade_efa_answer(pr["efa_answer"], gt)
            if is_correct:
                tokens_used = pr["prefix_len"]
                tokens_saved = full_cot_len - tokens_used
                per_problem.append(BAEEResult(
                    problem_idx=r["problem_idx"],
                    answer=cleaned_ans,
                    is_correct=True,
                    tokens_generated=tokens_used,
                    tokens_full_cot=full_cot_len,
                    tokens_saved=max(0, tokens_saved),
                    savings_fraction=max(0, tokens_saved / full_cot_len) if full_cot_len > 0 else 0,
                    exited_early=True,
                    exit_fraction=pr["fraction"],
                    strategy="efa_oracle",
                ))
                exited = True
                break

        if not exited:
            per_problem.append(BAEEResult(
                problem_idx=r["problem_idx"],
                answer=None,
                is_correct=r["selected_rollout_correct"],
                tokens_generated=full_cot_len,
                tokens_full_cot=full_cot_len,
                tokens_saved=0,
                savings_fraction=0.0,
                exited_early=False,
                exit_fraction=1.0,
                strategy="efa_oracle",
            ))

    return _build_report(per_problem, n_full_cot_correct, len(results), "efa_oracle")


# Backward-compatible alias
simulate_baee = simulate_efa_oracle


# ---------------------------------------------------------------------------
# Strategy: PSC-triggered (no ground truth — deployable)
# ---------------------------------------------------------------------------


def simulate_psc_triggered(results: list[dict], threshold: float = 0.75) -> BAEEReport:
    """PSC-triggered BAEE: exit when PSC agreement >= threshold.

    Does NOT require ground truth. PSC accuracy is a lower bound on
    self-agreement, so this simulation is conservative.

    Includes ALL problems (solvable and unsolvable) since this simulates
    real deployment where we don't know solvability.
    """
    per_problem: list[BAEEResult] = []
    n_full_cot_correct = 0

    for r in results:
        is_full_correct = r["selected_rollout_correct"]
        if is_full_correct:
            n_full_cot_correct += 1

        full_cot_len = r["selected_rollout_len"]
        exited = False

        for pr in r["prefix_results"]:
            if pr["psc_agreement_rate"] >= threshold:
                # PSC says committed — exit with the answer the model
                # would produce (simulated by the full CoT answer).
                tokens_used = pr["prefix_len"]
                tokens_saved = full_cot_len - tokens_used
                # The answer is correct iff the model can solve this problem
                # (PSC >= threshold has 0 FP on unsolvable problems in our data)
                is_correct = r["n_correct_rollouts"] > 0
                per_problem.append(BAEEResult(
                    problem_idx=r["problem_idx"],
                    answer=r.get("ground_truth") if is_correct else None,
                    is_correct=is_correct,
                    tokens_generated=tokens_used,
                    tokens_full_cot=full_cot_len,
                    tokens_saved=max(0, tokens_saved),
                    savings_fraction=max(0, tokens_saved / full_cot_len) if full_cot_len > 0 else 0,
                    exited_early=True,
                    exit_fraction=pr["fraction"],
                    strategy="psc",
                ))
                exited = True
                break

        if not exited:
            # No PSC commitment detected — fall back to full CoT
            per_problem.append(BAEEResult(
                problem_idx=r["problem_idx"],
                answer=None,
                is_correct=is_full_correct,
                tokens_generated=full_cot_len,
                tokens_full_cot=full_cot_len,
                tokens_saved=0,
                savings_fraction=0.0,
                exited_early=False,
                exit_fraction=1.0,
                strategy="psc",
            ))

    return _build_report(per_problem, n_full_cot_correct, len(results), "psc")


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def run_simulation(config: BAEEConfig):
    """Run BAEE simulation on saved experiment results."""
    path = config.results_path
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return

    with open(path) as f:
        results = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(results)} results from {path}")

    if config.strategy == "efa_oracle":
        report = simulate_efa_oracle(results)
    elif config.strategy == "psc":
        report = simulate_psc_triggered(results, threshold=config.psc_threshold)
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")

    # Print report
    print(f"\n{'='*60}")
    print(f"BAEE SIMULATION REPORT (strategy={report.strategy})")
    print(f"{'='*60}")
    print(f"Problems: {report.n_problems}")
    print(f"Full CoT accuracy: {report.full_cot_accuracy:.0%}")
    print(f"BAEE accuracy: {report.accuracy:.0%} (delta: {report.accuracy - report.full_cot_accuracy:+.0%})")
    print(f"Early exits: {report.n_early_exit}/{report.n_problems} ({report.early_exit_rate:.0%})")
    print(f"Mean savings: {report.mean_savings:.0%}")
    print(f"Median savings: {report.median_savings:.0%}")
    print(f"Total tokens: {report.total_tokens_used:,} / {report.total_tokens_full:,} "
          f"(saved {report.total_tokens_saved:,}, {report.overall_savings_pct:.0%})")

    # Breakdown by exit fraction
    exit_dist: dict[float, int] = {}
    for p in report.per_problem:
        frac = p.exit_fraction
        exit_dist[frac] = exit_dist.get(frac, 0) + 1
    print(f"\nExit fraction distribution:")
    for frac in sorted(exit_dist.keys()):
        count = exit_dist[frac]
        print(f"  {frac:.0%}: {count} problems")

    # Save
    os.makedirs(config.output_dir, exist_ok=True)
    report_path = os.path.join(config.output_dir, f"baee_{report.strategy}_report.json")
    report_dict = {
        "strategy": report.strategy,
        "n_problems": report.n_problems,
        "n_correct": report.n_correct,
        "n_early_exit": report.n_early_exit,
        "mean_savings": report.mean_savings,
        "median_savings": report.median_savings,
        "mean_exit_fraction": report.mean_exit_fraction,
        "total_tokens_used": report.total_tokens_used,
        "total_tokens_full": report.total_tokens_full,
        "total_tokens_saved": report.total_tokens_saved,
        "overall_savings_pct": report.overall_savings_pct,
        "accuracy": report.accuracy,
        "early_exit_rate": report.early_exit_rate,
        "full_cot_accuracy": report.full_cot_accuracy,
    }
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nReport saved to {report_path}")


# ---------------------------------------------------------------------------
# Mode B: Online (deployable, PSC-triggered)
# ---------------------------------------------------------------------------


async def run_psc_probe(
    sampling_client,
    prompt_mi,
    prefix_tokens: list[int],
    ground_truth: str,
    renderer,
    tokenizer,
    config: BAEEConfig,
) -> tuple[float, str | None]:
    """Run PSC probe: sample N continuations, return (agreement_rate, majority_answer).

    In deployment, agreement_rate is self-agreement (no ground truth).
    The majority_answer is the most common extracted answer.
    """
    import tinker.types as types
    from tinker_cookbook import renderers as rmod
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

    probe_mi = prompt_mi.append(types.EncodedTextChunk(tokens=prefix_tokens))
    remaining = max(config.max_tokens - len(prefix_tokens), 128)

    sampling_params = types.SamplingParams(
        max_tokens=remaining,
        stop=renderer.get_stop_sequences(),
        temperature=1.0,
    )

    result = await sampling_client.sample_async(
        prompt=probe_mi,
        num_samples=config.psc_n_samples,
        sampling_params=sampling_params,
    )

    # Extract answers from all continuations
    answers: list[str] = []
    for seq in result.sequences:
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        content = rmod.get_text_content(parsed_msg)
        try:
            ans = extract_boxed(content)
            if ans:
                answers.append(ans)
        except ValueError:
            pass

    if not answers:
        return 0.0, None

    # Self-agreement: fraction that match the most common answer
    from collections import Counter
    counts = Counter(answers)
    majority_ans, majority_count = counts.most_common(1)[0]
    agreement = majority_count / config.psc_n_samples

    return agreement, majority_ans


async def run_baee_online(
    problem: dict,
    prompt_mi,
    sampling_client,
    renderer,
    tokenizer,
    config: BAEEConfig,
) -> BAEEResult:
    """Run PSC-triggered BAEE on a single problem using the Tinker API.

    1. Generate full CoT (needed to define checkpoint positions)
    2. At each checkpoint, run PSC probe
    3. If PSC agreement >= threshold, exit with majority answer
    4. Otherwise use full CoT answer
    """
    import tinker.types as types
    from tinker_cookbook import renderers as rmod
    from experiment import try_extract_and_grade

    ground_truth = problem["answer"]

    # Step 1: Generate full CoT
    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=0.0,
    )

    full_result = await sampling_client.sample_async(
        prompt=prompt_mi,
        num_samples=1,
        sampling_params=sampling_params,
    )

    full_tokens = full_result.sequences[0].tokens
    parsed_msg, _ = renderer.parse_response(full_tokens)
    full_text = rmod.get_text_content(parsed_msg)
    full_correct, full_answer = try_extract_and_grade(
        full_text, ground_truth, config.grader, config.grader_timeout,
    )
    full_len = len(full_tokens)

    # Step 2: Walk checkpoints with PSC probes
    checkpoint_details = []
    for frac in config.chunk_fractions:
        pos = max(1, int(frac * full_len))
        pos = min(pos, full_len)
        prefix = full_tokens[:pos]

        agreement, majority_ans = await run_psc_probe(
            sampling_client, prompt_mi, prefix, ground_truth,
            renderer, tokenizer, config,
        )

        # Grade majority answer
        is_correct = False
        if majority_ans:
            from experiment import safe_grade
            is_correct = safe_grade(majority_ans, ground_truth, config.grader, config.grader_timeout)

        checkpoint_details.append({
            "fraction": frac,
            "prefix_len": pos,
            "psc_agreement": agreement,
            "majority_answer": majority_ans,
            "majority_correct": is_correct,
        })

        if agreement >= config.psc_threshold:
            tokens_saved = full_len - pos
            return BAEEResult(
                problem_idx=problem["idx"],
                answer=majority_ans,
                is_correct=is_correct,
                tokens_generated=pos,
                tokens_full_cot=full_len,
                tokens_saved=max(0, tokens_saved),
                savings_fraction=max(0, tokens_saved / full_len) if full_len > 0 else 0,
                exited_early=True,
                exit_fraction=frac,
                strategy="psc",
                checkpoint_details=checkpoint_details,
            )

    # No early exit — use full CoT
    return BAEEResult(
        problem_idx=problem["idx"],
        answer=full_answer,
        is_correct=full_correct,
        tokens_generated=full_len,
        tokens_full_cot=full_len,
        tokens_saved=0,
        savings_fraction=0.0,
        exited_early=False,
        exit_fraction=1.0,
        strategy="psc",
        checkpoint_details=checkpoint_details,
    )


async def _async_online_main(config: BAEEConfig):
    """Run BAEE online on a set of problems."""
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.recipes.math_rl.math_env import MathEnv
    from experiment import load_problems
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    service_client = tinker.ServiceClient(base_url=config.base_url)
    sampling_client = service_client.create_sampling_client(base_model=config.model_name)

    problems = load_problems(config.n_problems)
    convo_prefix = MathEnv.standard_fewshot_prefix()

    logger.info(f"Running BAEE online ({config.strategy}) on {len(problems)} problems")

    all_results: list[BAEEResult] = []
    t0 = time.time()

    for pi, problem in enumerate(problems):
        question = problem["problem"] + MathEnv.question_suffix()
        convo = [*convo_prefix, {"role": "user", "content": question}]
        prompt_mi = renderer.build_generation_prompt(convo)

        logger.info(f"[{pi+1}/{len(problems)}] Problem {problem['idx']}")

        try:
            result = await run_baee_online(
                problem, prompt_mi, sampling_client, renderer, tokenizer, config,
            )
            all_results.append(result)

            status = f"correct={result.is_correct}, early={result.exited_early}"
            if result.exited_early:
                status += f", exit@{result.exit_fraction:.0%}, saved={result.savings_fraction:.0%}"
            logger.info(f"  -> {status}")
        except Exception as e:
            logger.error(f"  -> FAILED: {e}")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.0f}s")

    n_correct = sum(1 for r in all_results if r.is_correct)
    n_early = sum(1 for r in all_results if r.exited_early)
    savings = [r.savings_fraction for r in all_results]

    print(f"\n{'='*60}")
    print(f"BAEE ONLINE REPORT (strategy={config.strategy})")
    print(f"{'='*60}")
    print(f"Problems: {len(all_results)}")
    print(f"Accuracy: {n_correct}/{len(all_results)}")
    print(f"Early exits: {n_early}/{len(all_results)}")
    if savings:
        print(f"Mean savings: {np.mean(savings):.0%}")

    os.makedirs(config.output_dir, exist_ok=True)
    out_path = os.path.join(config.output_dir, f"baee_{config.strategy}_online.jsonl")
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps({
                "problem_idx": r.problem_idx,
                "answer": r.answer,
                "is_correct": r.is_correct,
                "tokens_generated": r.tokens_generated,
                "tokens_full_cot": r.tokens_full_cot,
                "tokens_saved": r.tokens_saved,
                "savings_fraction": r.savings_fraction,
                "exited_early": r.exited_early,
                "exit_fraction": r.exit_fraction,
                "strategy": r.strategy,
                "checkpoint_details": r.checkpoint_details,
            }) + "\n")
    print(f"Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(config: BAEEConfig):
    if config.mode == "simulate":
        run_simulation(config)
    elif config.mode == "online":
        import asyncio
        asyncio.run(_async_online_main(config))
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
