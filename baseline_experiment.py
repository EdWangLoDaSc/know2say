"""
Prefix-free baselines for isolating the contribution of prefix-conditioned state in PSC-BAEE.

Three baselines, all prefix-free:

  SC-8-full          8 full-length CoTs from prompt, majority vote
                     Tests: does majority voting alone explain PSC accuracy?

  SC-8-budget        8 CoTs from prompt, each capped at PSC per-call budget
                     Matches both call count AND per-call token budget of PSC-8 adaptive
                     Tests: does budget-matched ensembling explain PSC accuracy?

  Single-budget      1 CoT from prompt, capped at PSC per-call budget
                     Removes both prefix and ensembling
                     Tests: is budget reduction itself the bottleneck?

Budget definition:
    PSC at f=0.10 uses max_tokens = min(2 * 0.90 * cot_len, max_tokens_cap).
    We read cot_len from the main experiment results (per problem).
    If a problem is missing from main results, we fall back to the model-average CoT length.

Call count accounting:
    SC-8-full:   8 calls  (same as PSC-8 at one checkpoint)
    SC-8-budget: 8 calls  (matched to PSC-8 at one checkpoint)
    Single:      1 call

Token budget accounting:
    SC-8-full:   8 × max_tokens_cap
    SC-8-budget: 8 × psc_per_call_budget  (== PSC-8 total continuation budget at exit checkpoint)
    Single:      1 × psc_per_call_budget
"""

import asyncio
import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field

import chz
import numpy as np
import tinker
import tinker.types as types

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import MathEnv
from experiment import (
    ExperimentConfig,
    load_problems,
    try_extract_and_grade,
    safe_grade,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

PSC_TRIGGER_FRACTION = 0.10   # checkpoint where PSC typically exits
PSC_BUDGET_MULTIPLIER = 2.0   # PSC continuation max_tokens = 2 × remaining CoT tokens
N_SC_SAMPLES = 8              # must match n_prefix_probes in main experiment


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────


@chz.chz
class BaselineConfig:
    model_name: str = "Qwen/Qwen3-32B"
    renderer_name: str | None = None
    benchmark: str = "math-500"
    n_problems: int = 500
    max_tokens: int = 4096            # full-budget cap (SC-8-full)
    main_results_path: str = ""       # path to main experiment results.jsonl (for CoT lengths)
    output_dir: str = "/tmp/tinker-examples/reasoning_theater/baseline"
    seed: int = 42
    grader: str = "sympy"
    grader_timeout: float = 2.0


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def load_cot_lengths(main_results_path: str) -> dict[int, int]:
    """Return {problem_idx: selected_rollout_len} from main experiment results."""
    cot_lens: dict[int, int] = {}
    if not main_results_path or not os.path.exists(main_results_path):
        return cot_lens
    with open(main_results_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            idx = r.get("problem_idx", -1)
            cot_len = r.get("selected_rollout_len", 0)
            if cot_len > 0:
                cot_lens[idx] = cot_len
    return cot_lens


def psc_budget_for_cot(cot_len: int, max_tokens: int) -> int:
    """PSC per-call budget: 2 × tokens remaining after f=0.10 prefix, capped."""
    remaining = max(cot_len - int(PSC_TRIGGER_FRACTION * cot_len), 64)
    return min(int(PSC_BUDGET_MULTIPLIER * remaining), max_tokens)


def majority_vote_correct(
    answers: list[str | None],
    ground_truth: str,
    grader: str,
    timeout: float,
) -> tuple[bool, str | None]:
    """Return (correct, majority_answer) using majority vote over non-None answers."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return False, None
    counts = Counter(valid)
    majority_ans = counts.most_common(1)[0][0]
    from experiment import safe_grade
    correct = safe_grade(majority_ans, ground_truth, grader, timeout)
    return correct, majority_ans


# ──────────────────────────────────────────────────────────────
# Single-problem baseline runner
# ──────────────────────────────────────────────────────────────


async def run_baselines_for_problem(
    problem: dict,
    prompt_mi: types.ModelInput,
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    config: BaselineConfig,
    psc_budget: int,
    grader_override: str | None = None,
) -> dict:
    """Run all three prefix-free baselines for one problem. Returns result dict."""
    ground_truth = problem["answer"]
    grader = grader_override or config.grader
    grader_timeout = config.grader_timeout

    async def _sample_n(n: int, budget: int) -> list[str | None]:
        """Sample n continuations from the prompt with given max_tokens. Returns extracted answers."""
        sp = types.SamplingParams(
            max_tokens=budget,
            stop=renderer.get_stop_sequences(),
            temperature=1.0,
        )
        result = await sampling_client.sample_async(
            prompt=prompt_mi,
            num_samples=n,
            sampling_params=sp,
        )
        answers = []
        for seq in result.sequences:
            parsed_msg, _ = renderer.parse_response(seq.tokens)
            from tinker_cookbook.renderers import get_text_content
            content = get_text_content(parsed_msg)
            _, ans = try_extract_and_grade(content, ground_truth, grader, grader_timeout)
            answers.append(ans)
        return answers

    # Run all three baselines in parallel
    sc8_full_task = asyncio.create_task(_sample_n(N_SC_SAMPLES, config.max_tokens))
    sc8_budget_task = asyncio.create_task(_sample_n(N_SC_SAMPLES, psc_budget))
    single_task = asyncio.create_task(_sample_n(1, psc_budget))

    sc8_full_answers, sc8_budget_answers, single_answers = await asyncio.gather(
        sc8_full_task, sc8_budget_task, single_task
    )

    # Grade results — answers are already extracted, so use safe_grade directly
    def grade_answers(answers: list[str | None]) -> dict:
        n_total = len(answers)
        n_correct_individual = sum(
            1 for a in answers
            if a is not None and safe_grade(a, ground_truth, grader, grader_timeout)
        )
        majority_correct, majority_ans = majority_vote_correct(
            answers, ground_truth, grader, grader_timeout
        )
        return {
            "answers": answers,
            "n_correct_individual": n_correct_individual,
            "n_total": n_total,
            "majority_correct": majority_correct,
            "majority_answer": majority_ans,
            "pass_at_1": n_correct_individual / n_total if n_total > 0 else 0.0,
        }

    single_correct = (
        single_answers[0] is not None and
        safe_grade(single_answers[0], ground_truth, grader, grader_timeout)
    )

    return {
        "problem_idx": problem["idx"],
        "level": problem["level"],
        "subject": problem["subject"],
        "ground_truth": ground_truth,
        "psc_budget": psc_budget,
        "sc8_full": grade_answers(sc8_full_answers),
        "sc8_budget": grade_answers(sc8_budget_answers),
        "single_budget": {
            "answer": single_answers[0] if single_answers else None,
            "correct": single_correct,
        },
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


async def _async_main(config: BaselineConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.makedirs(config.output_dir, exist_ok=True)

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    logger.info(f"Model: {config.model_name}, Renderer: {renderer_name}")

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=config.model_name)

    problems = load_problems(config.n_problems, config.benchmark, seed=config.seed)
    logger.info(f"Loaded {len(problems)} problems from {config.benchmark}")

    # Auto-select grader for MC benchmarks
    from experiment import _is_mc_benchmark
    is_mc = _is_mc_benchmark(config.benchmark)
    effective_grader = "exact" if is_mc else config.grader
    if is_mc:
        logger.info(f"MC benchmark detected, using exact grader")

    # Load per-problem CoT lengths from main experiment
    cot_lens = load_cot_lengths(config.main_results_path)
    logger.info(f"Loaded CoT lengths for {len(cot_lens)} problems from main results")

    # Fallback: average CoT length from available data
    if cot_lens:
        avg_cot_len = int(np.mean(list(cot_lens.values())))
    else:
        avg_cot_len = 2000  # conservative fallback
    fallback_budget = psc_budget_for_cot(avg_cot_len, config.max_tokens)
    logger.info(f"Avg CoT len: {avg_cot_len}, fallback PSC budget: {fallback_budget}")

    convo_prefix = MathEnv.standard_fewshot_prefix() if not _is_mc_benchmark(config.benchmark) else []
    output_path = os.path.join(config.output_dir, "results.jsonl")

    # Resume support: skip already-processed problems
    done_idxs: set[int] = set()
    if os.path.exists(output_path):
        with open(output_path) as _f:
            for _line in _f:
                if _line.strip():
                    try:
                        done_idxs.add(json.loads(_line)["problem_idx"])
                    except Exception:
                        pass
        if done_idxs:
            logger.info(f"Resuming: {len(done_idxs)} problems already done, skipping.")

    t0 = time.time()
    n_processed = 0

    with open(output_path, "a") as f:
        for pi, problem in enumerate(problems):
            if problem["idx"] in done_idxs:
                continue
            if _is_mc_benchmark(config.benchmark):
                question = problem["problem"]
            else:
                question = problem["problem"] + MathEnv.question_suffix()
            convo = [*convo_prefix, {"role": "user", "content": question}]
            prompt_mi = renderer.build_generation_prompt(convo)

            # Determine per-call PSC budget for this specific problem
            cot_len = cot_lens.get(problem["idx"], avg_cot_len)
            budget = psc_budget_for_cot(cot_len, config.max_tokens)

            logger.info(
                f"[{pi+1}/{len(problems)}] idx={problem['idx']} L{problem['level']} "
                f"cot_len={cot_len} psc_budget={budget}"
            )

            try:
                result = await run_baselines_for_problem(
                    problem, prompt_mi, sampling_client, renderer, config, psc_budget=budget,
                    grader_override=effective_grader,
                )
                f.write(json.dumps(result) + "\n")
                f.flush()

                sc8f = result["sc8_full"]
                sc8b = result["sc8_budget"]
                sb = result["single_budget"]
                logger.info(
                    f"  sc8_full={sc8f['majority_correct']} ({sc8f['n_correct_individual']}/8) | "
                    f"sc8_budget={sc8b['majority_correct']} ({sc8b['n_correct_individual']}/8) | "
                    f"single={sb['correct']}"
                )
                n_processed += 1

            except Exception as e:
                logger.error(f"  FAILED: {e}")
                continue

    elapsed = time.time() - t0
    logger.info(f"Done: {n_processed}/{len(problems)} problems in {elapsed:.0f}s")

    # Save config
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump({
            "model_name": config.model_name,
            "renderer_name": renderer_name,
            "n_problems": config.n_problems,
            "max_tokens": config.max_tokens,
            "main_results_path": config.main_results_path,
            "n_sc_samples": N_SC_SAMPLES,
            "psc_trigger_fraction": PSC_TRIGGER_FRACTION,
            "psc_budget_multiplier": PSC_BUDGET_MULTIPLIER,
            "avg_cot_len_from_main": avg_cot_len,
        }, f, indent=2)

    # Print quick summary
    _print_summary(output_path)


def _print_summary(output_path: str):
    results = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    n = len(results)
    if n == 0:
        print("No results.")
        return

    sc8f_acc = sum(1 for r in results if r["sc8_full"]["majority_correct"]) / n
    sc8b_acc = sum(1 for r in results if r["sc8_budget"]["majority_correct"]) / n
    sb_acc = sum(1 for r in results if r["single_budget"]["correct"]) / n
    avg_budget = np.mean([r["psc_budget"] for r in results])

    print(f"\n{'='*50}")
    print(f"Baseline Summary (n={n})")
    print(f"{'='*50}")
    print(f"  Avg PSC per-call budget: {avg_budget:.0f} tokens")
    print(f"  SC-8-full   (8 calls, full budget):   {sc8f_acc:.1%}")
    print(f"  SC-8-budget (8 calls, PSC budget):    {sc8b_acc:.1%}")
    print(f"  Single-budget (1 call, PSC budget):   {sb_acc:.1%}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import chz
    config = chz.entrypoint(BaselineConfig)
    asyncio.run(_async_main(config))
