"""
Logprob Landscape of Reasoning Theater.

A probe-free, API-level behavioral study of "reasoning theater" — the phenomenon
where models already know the answer early in chain-of-thought but continue
generating performative tokens.

Uses four measurement protocols at prefix checkpoints through the CoT:
  A) Early Forced Answering (EFA) — append answer-forcing suffix, greedy decode
  B) Answer Token Logprob Trajectory (ATLT) — compute_logprobs on prefix + answer
  C) Entropy Dynamics (ED) — topk_prompt_logprobs over full CoT
  D) Prefix Self-Consistency (PSC) — sample N continuations from each prefix

References:
  - "Reasoning Theater" (2603.05488)
  - "Spike/Sparse/Sink" (2603.05498)

Usage:
    python -m experiment
    python -m experiment model_name=Qwen/Qwen3-32B n_problems=100
"""

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Literal

import chz
import numpy as np
import tinker
import tinker.types as types
from datasets import Dataset

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import MathEnv, _get_hendrycks_math_test
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────


@chz.chz
class ExperimentConfig:
    base_url: str | None = None
    model_name: str = "Qwen/Qwen3-32B"
    renderer_name: str | None = None  # auto-detect from model if None
    benchmark: Literal["math-500", "gsm8k", "arc-challenge", "aime-2024"] = "math-500"
    n_problems: int = 100
    n_full_rollouts: int = 4  # full CoT rollouts per problem
    max_tokens: int = 4096
    prefix_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    # Protocol D: self-consistency
    n_prefix_probes: int = 8
    # Protocol C: entropy
    topk_for_entropy: int = 20
    # EFA suffix
    efa_max_tokens: int = 64
    # Grading
    grader: Literal["sympy", "math_verify", "exact"] = "sympy"
    grader_timeout: float = 2.0
    # Output
    output_dir: str = "/tmp/tinker-examples/reasoning_theater"
    seed: int = 42


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────


@dataclass
class FullRollout:
    tokens: list[int]
    logprobs: list[float]
    text: str
    is_correct: bool
    extracted_answer: str | None


@dataclass
class PrefixResult:
    fraction: float
    prefix_len: int
    total_len: int
    # Protocol A: Early Forced Answering
    efa_answer: str | None = None
    efa_correct: bool = False
    # Protocol B: Answer Token Logprob Trajectory
    atlt_logprob: float | None = None
    # Protocol D: Prefix Self-Consistency
    psc_n_correct: int = 0
    psc_n_total: int = 0
    psc_agreement_rate: float = 0.0


@dataclass
class ProblemResult:
    problem_idx: int
    problem: str
    level: str
    subject: str
    ground_truth: str
    n_correct_rollouts: int
    n_total_rollouts: int
    selected_rollout_len: int
    selected_rollout_correct: bool
    prefix_results: list[PrefixResult] = field(default_factory=list)
    # Protocol C: Entropy dynamics (per-token for full CoT)
    entropy_curve: list[float] = field(default_factory=list)
    # Derived: commitment point (first prefix where EFA is correct)
    commitment_fraction: float | None = None
    theater_fraction: float | None = None  # 1 - commitment_fraction


# ─────────────────────────────────────────────
# Grading helpers
# ─────────────────────────────────────────────


def safe_grade(given: str, ground_truth: str, grader: str = "sympy", timeout: float = 2.0) -> bool:
    if grader == "exact":
        return given.strip().upper() == ground_truth.strip().upper()

    from tinker_cookbook.recipes.math_rl.math_grading import run_with_timeout_signal

    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        from tinker_cookbook.recipes.math_rl.math_grading import grade_answer_math_verify
        grader_func = grade_answer_math_verify
    else:
        raise ValueError(f"Invalid grader: {grader}")
    out = run_with_timeout_signal(grader_func, args=(given, ground_truth), timeout_seconds=int(math.ceil(timeout)))
    if out is None:
        return False
    return out


def try_extract_and_grade(text: str, ground_truth: str, grader: str, timeout: float) -> tuple[bool, str | None]:
    if grader == "exact":
        # For MC: extract the letter answer (A/B/C/D)
        answer = _extract_mc_answer(text)
        if answer:
            return answer.upper() == ground_truth.strip().upper(), answer
        return False, None
    try:
        answer = extract_boxed(text)
    except ValueError:
        return False, None
    return safe_grade(answer, ground_truth, grader, timeout), answer


def _extract_mc_answer(text: str) -> str | None:
    """Extract a multiple-choice answer (A/B/C/D) from text."""
    import re
    # Try common patterns: "The answer is (C)", "The answer is C", "Answer: C"
    patterns = [
        r"(?:the answer is|answer is|answer:)\s*\(?([A-Da-d])\)?",
        r"\\boxed\{([A-Da-d])\}",
        r"\b([A-D])\)\s*$",  # trailing "C)" at end
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    # Fallback: last standalone letter A-D
    match = re.findall(r"\b([A-D])\b", text)
    if match:
        return match[-1]
    return None


# ─────────────────────────────────────────────
# Protocol A: Early Forced Answering
# ─────────────────────────────────────────────


EFA_SUFFIX = "\nTherefore, the final answer is \\boxed{"
EFA_SUFFIX_MC = "\nTherefore, the answer is ("


def get_efa_suffix(benchmark: str) -> str:
    if benchmark == "arc-challenge":
        return EFA_SUFFIX_MC
    return EFA_SUFFIX


async def run_efa(
    sampling_client: tinker.SamplingClient,
    prompt_mi: types.ModelInput,
    prefix_tokens: list[int],
    ground_truth: str,
    renderer: renderers.Renderer,
    tokenizer: object,
    config: ExperimentConfig,
) -> tuple[str | None, bool]:
    """Force the model to answer from a prefix by appending an answer-forcing suffix."""
    efa_suffix = get_efa_suffix(config.benchmark)
    efa_suffix_tokens = tokenizer.encode(efa_suffix, add_special_tokens=False)

    combined_tokens = list(prefix_tokens) + list(efa_suffix_tokens)
    forced_mi = prompt_mi.append(types.EncodedTextChunk(tokens=combined_tokens))

    is_mc = config.benchmark == "arc-challenge"
    sampling_params = types.SamplingParams(
        max_tokens=config.efa_max_tokens,
        temperature=0.0,
        stop=[")"] if is_mc else ["}"],
    )

    result = await sampling_client.sample_async(
        prompt=forced_mi,
        num_samples=1,
        sampling_params=sampling_params,
    )

    answer_tokens = result.sequences[0].tokens
    answer_text = tokenizer.decode(answer_tokens)

    if is_mc:
        # Extract single letter A-D
        answer = answer_text.strip().rstrip(")").strip()
        if answer and len(answer) == 1 and answer.upper() in "ABCD":
            answer = answer.upper()
        else:
            # Try to find a letter in the response
            answer = _extract_mc_answer(answer_text)
        if not answer:
            return None, False
        correct = answer.upper() == ground_truth.strip().upper()
        return answer, correct
    else:
        boxed_answer = answer_text.strip().rstrip("}").rstrip(".").strip()
        if not boxed_answer:
            return None, False
        correct = safe_grade(boxed_answer, ground_truth, config.grader, config.grader_timeout)
        return boxed_answer, correct


# ─────────────────────────────────────────────
# Protocol B: Answer Token Logprob Trajectory
# ─────────────────────────────────────────────


async def run_atlt(
    sampling_client: tinker.SamplingClient,
    prompt_mi: types.ModelInput,
    prefix_tokens: list[int],
    answer_text: str,
    tokenizer: object,
) -> float | None:
    """Compute logprob of the correct answer given a CoT prefix.

    Constructs: prompt + prefix + answer_suffix, then uses compute_logprobs
    to get the logprob at the answer token positions.
    """
    answer_suffix = f"\nThe answer is \\boxed{{{answer_text}}}"
    answer_suffix_tokens = tokenizer.encode(answer_suffix, add_special_tokens=False)

    if not answer_suffix_tokens:
        return None

    combined_tokens = list(prefix_tokens) + list(answer_suffix_tokens)
    mi = prompt_mi.append(types.EncodedTextChunk(tokens=combined_tokens))

    logprobs = await sampling_client.compute_logprobs_async(prompt=mi)

    # Extract logprobs at answer suffix positions (the last len(answer_suffix_tokens) positions)
    n_answer = len(answer_suffix_tokens)
    answer_logprobs = logprobs[-n_answer:]

    # Mean logprob over answer tokens (skip None values)
    valid = [lp for lp in answer_logprobs if lp is not None]
    if not valid:
        return None
    return float(np.mean(valid))


# ─────────────────────────────────────────────
# Protocol C: Entropy Dynamics
# ─────────────────────────────────────────────


def compute_entropy_from_topk(topk_logprobs: list[tuple[int, float]]) -> float:
    """Approximate token entropy from top-k logprobs."""
    if not topk_logprobs:
        return 0.0

    probs = [math.exp(lp) for _, lp in topk_logprobs]
    remaining = max(0.0, 1.0 - sum(probs))

    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    if remaining > 0:
        entropy -= remaining * math.log(remaining)

    return entropy


async def run_entropy_dynamics(
    sampling_client: tinker.SamplingClient,
    prompt_mi: types.ModelInput,
    cot_tokens: list[int],
    config: ExperimentConfig,
) -> list[float]:
    """Get per-token entropy over the full CoT using topk_prompt_logprobs."""
    # Build prompt + full CoT
    full_mi = prompt_mi.append(types.EncodedTextChunk(tokens=cot_tokens))

    # We need topk logprobs; use sample with max_tokens=1 just to trigger prompt logprob computation
    result = await sampling_client.sample_async(
        prompt=full_mi,
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=config.topk_for_entropy,
    )

    topk_all = result.topk_prompt_logprobs
    if topk_all is None:
        return []

    # topk_all has entries for all prompt tokens. We want only the CoT portion.
    # The prompt has some number of tokens, then the CoT tokens start.
    # topk_all length = total prompt length = prompt_mi tokens + cot_tokens
    # We want the last len(cot_tokens) entries.
    n_cot = len(cot_tokens)
    cot_topk = topk_all[-n_cot:]

    entropies = []
    for entry in cot_topk:
        if entry is None:
            entropies.append(0.0)
        else:
            entropies.append(compute_entropy_from_topk(entry))

    return entropies


# ─────────────────────────────────────────────
# Protocol D: Prefix Self-Consistency
# ─────────────────────────────────────────────


async def run_psc(
    sampling_client: tinker.SamplingClient,
    prompt_mi: types.ModelInput,
    prefix_tokens: list[int],
    ground_truth: str,
    renderer: renderers.Renderer,
    tokenizer: object,
    config: ExperimentConfig,
    total_cot_len: int | None = None,
) -> tuple[int, int]:
    """Sample N continuations from a prefix and count how many give the correct answer.

    total_cot_len: if provided, caps PSC max_tokens to 2x the remaining CoT length,
    avoiding massive over-generation when CoTs are short relative to config.max_tokens.
    """
    probe_mi = prompt_mi.append(types.EncodedTextChunk(tokens=prefix_tokens))

    # Cap at 2× the actual remaining CoT length to avoid over-generation.
    # Floor at 256 tokens to handle edge cases.
    if total_cot_len is not None:
        remaining_actual = max(total_cot_len - len(prefix_tokens), 64)
        remaining_tokens = max(min(remaining_actual * 2, config.max_tokens), 256)
    else:
        remaining_tokens = max(config.max_tokens - len(prefix_tokens), 128)
    sampling_params = types.SamplingParams(
        max_tokens=remaining_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=1.0,
    )

    result = await sampling_client.sample_async(
        prompt=probe_mi,
        num_samples=config.n_prefix_probes,
        sampling_params=sampling_params,
    )

    n_correct = 0
    for seq in result.sequences:
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        content = renderers.get_text_content(parsed_msg)
        correct, _ = try_extract_and_grade(content, ground_truth, config.grader, config.grader_timeout)
        if correct:
            n_correct += 1

    return n_correct, config.n_prefix_probes


# ─────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────


def load_problems(n_problems: int, benchmark: str = "math-500") -> list[dict]:
    """Load problems from the specified benchmark."""
    if benchmark == "math-500":
        return _load_math500(n_problems)
    elif benchmark == "gsm8k":
        return _load_gsm8k(n_problems)
    elif benchmark == "arc-challenge":
        return _load_arc_challenge(n_problems)
    elif benchmark == "aime-2024":
        return _load_aime2024(n_problems)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def _load_math500(n_problems: int) -> list[dict]:
    ds = _get_hendrycks_math_test()
    problems = []
    for i, row in enumerate(ds):
        if i >= n_problems:
            break
        try:
            answer = extract_boxed(row["solution"])
        except ValueError:
            continue
        problems.append({
            "idx": i,
            "problem": row["problem"],
            "answer": answer,
            "level": row.get("level", "unknown"),
            "subject": row.get("subject", "unknown"),
        })
    return problems


def _load_gsm8k(n_problems: int) -> list[dict]:
    import re
    from datasets import load_dataset as hf_load
    ds = hf_load("openai/gsm8k", name="main", split="test")
    problems = []
    for i, row in enumerate(ds):
        if i >= n_problems:
            break
        match = re.search(r"####\s*(.*)", row["answer"])
        if not match:
            continue
        answer = match.group(1).strip().replace(",", "")
        problems.append({
            "idx": i,
            "problem": row["question"],
            "answer": answer,
            "level": "unknown",
            "subject": "Math",
        })
    return problems


def _format_arc_choices(choices: dict) -> str:
    """Format ARC multiple-choice options as A) ... B) ... etc."""
    parts = []
    for label, text in zip(choices["label"], choices["text"]):
        parts.append(f"{label}) {text}")
    return "\n".join(parts)


def _load_arc_challenge(n_problems: int) -> list[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("allenai/ai2_arc", name="ARC-Challenge", split="test")
    problems = []
    for i, row in enumerate(ds):
        if i >= n_problems:
            break
        question = row["question"] + "\n\n" + _format_arc_choices(row["choices"])
        problems.append({
            "idx": i,
            "problem": question,
            "answer": row["answerKey"],
            "level": "unknown",
            "subject": "Science",
        })
    return problems


def _load_aime2024(n_problems: int) -> list[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("Maxwell-Jia/AIME_2024", split="train")
    problems = []
    for i, row in enumerate(ds):
        if i >= n_problems:
            break
        problems.append({
            "idx": i,
            "problem": row["Problem"],
            "answer": str(row["Answer"]),
            "level": "AIME",
            "subject": "Math",
        })
    return problems


async def process_problem(
    problem: dict,
    prompt_mi: types.ModelInput,
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    tokenizer: object,
    config: ExperimentConfig,
) -> ProblemResult:
    """Run all four protocols on a single problem."""
    ground_truth = problem["answer"]

    # Step 1: Generate full CoT rollouts
    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=1.0,
    )

    sample_result = await sampling_client.sample_async(
        prompt=prompt_mi,
        num_samples=config.n_full_rollouts,
        sampling_params=sampling_params,
    )

    rollouts: list[FullRollout] = []
    for seq in sample_result.sequences:
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        content = renderers.get_text_content(parsed_msg)
        correct, answer = try_extract_and_grade(content, ground_truth, config.grader, config.grader_timeout)
        rollouts.append(FullRollout(
            tokens=seq.tokens,
            logprobs=seq.logprobs if seq.logprobs else [],
            text=content,
            is_correct=correct,
            extracted_answer=answer,
        ))

    n_correct = sum(1 for r in rollouts if r.is_correct)

    # Step 2: Select one rollout (first correct, or first if all wrong)
    selected = next((r for r in rollouts if r.is_correct), rollouts[0])
    cot_tokens = selected.tokens

    if len(cot_tokens) < 10:
        # CoT too short for meaningful analysis
        return ProblemResult(
            problem_idx=problem["idx"],
            problem=problem["problem"],
            level=problem["level"],
            subject=problem["subject"],
            ground_truth=ground_truth,
            n_correct_rollouts=n_correct,
            n_total_rollouts=len(rollouts),
            selected_rollout_len=len(cot_tokens),
            selected_rollout_correct=selected.is_correct,
        )

    # Step 3: Compute prefix checkpoints
    prefix_checkpoints = []
    for frac in config.prefix_fractions:
        pos = max(1, int(frac * len(cot_tokens)))
        pos = min(pos, len(cot_tokens))
        prefix_checkpoints.append((frac, pos))

    # Step 4: Run protocols A, B, D at ALL checkpoints in parallel.
    # Within each checkpoint, EFA / ATLT / PSC are also concurrent.
    async def _run_checkpoint(frac: float, pos: int) -> PrefixResult:
        prefix = cot_tokens[:pos]
        pr = PrefixResult(fraction=frac, prefix_len=pos, total_len=len(cot_tokens))

        async def _efa():
            try:
                ans, ok = await run_efa(
                    sampling_client, prompt_mi, prefix, ground_truth, renderer, tokenizer, config
                )
                pr.efa_answer = ans
                pr.efa_correct = ok
            except Exception as e:
                logger.warning(f"EFA failed at {frac:.0%}: {e}")

        async def _atlt():
            try:
                lp = await run_atlt(sampling_client, prompt_mi, prefix, ground_truth, tokenizer)
                pr.atlt_logprob = lp
            except Exception as e:
                logger.warning(f"ATLT failed at {frac:.0%}: {e}")

        async def _psc():
            try:
                n_corr, n_total = await run_psc(
                    sampling_client, prompt_mi, prefix, ground_truth, renderer, tokenizer, config,
                    total_cot_len=len(cot_tokens),
                )
                pr.psc_n_correct = n_corr
                pr.psc_n_total = n_total
                pr.psc_agreement_rate = n_corr / n_total if n_total > 0 else 0.0
            except Exception as e:
                logger.warning(f"PSC failed at {frac:.0%}: {e}")

        await asyncio.gather(_efa(), _atlt(), _psc())
        return pr

    # Step 5: Protocol C — run entropy in parallel with all checkpoints
    async def _entropy():
        try:
            return await run_entropy_dynamics(sampling_client, prompt_mi, cot_tokens, config)
        except Exception as e:
            logger.warning(f"Entropy dynamics failed: {e}")
            return []

    results_and_entropy = await asyncio.gather(
        *[_run_checkpoint(frac, pos) for frac, pos in prefix_checkpoints],
        _entropy(),
    )
    prefix_results: list[PrefixResult] = list(results_and_entropy[:-1])
    entropy_curve: list[float] = results_and_entropy[-1] or []

    # Step 6: Compute commitment point
    commitment_frac = None
    if selected.is_correct:
        for pr in prefix_results:
            if pr.efa_correct:
                commitment_frac = pr.fraction
                break

    theater_frac = (1.0 - commitment_frac) if commitment_frac is not None else None

    result = ProblemResult(
        problem_idx=problem["idx"],
        problem=problem["problem"],
        level=problem["level"],
        subject=problem["subject"],
        ground_truth=ground_truth,
        n_correct_rollouts=n_correct,
        n_total_rollouts=len(rollouts),
        selected_rollout_len=len(cot_tokens),
        selected_rollout_correct=selected.is_correct,
        prefix_results=prefix_results,
        entropy_curve=entropy_curve,
        commitment_fraction=commitment_frac,
        theater_fraction=theater_frac,
    )
    return result


def result_to_dict(r: ProblemResult) -> dict:
    """Serialize a ProblemResult to a JSON-safe dict."""
    return {
        "problem_idx": r.problem_idx,
        "problem": r.problem,
        "level": r.level,
        "subject": r.subject,
        "ground_truth": r.ground_truth,
        "n_correct_rollouts": r.n_correct_rollouts,
        "n_total_rollouts": r.n_total_rollouts,
        "selected_rollout_len": r.selected_rollout_len,
        "selected_rollout_correct": r.selected_rollout_correct,
        "commitment_fraction": r.commitment_fraction,
        "theater_fraction": r.theater_fraction,
        "prefix_results": [
            {
                "fraction": pr.fraction,
                "prefix_len": pr.prefix_len,
                "total_len": pr.total_len,
                "efa_answer": pr.efa_answer,
                "efa_correct": pr.efa_correct,
                "atlt_logprob": pr.atlt_logprob,
                "psc_n_correct": pr.psc_n_correct,
                "psc_n_total": pr.psc_n_total,
                "psc_agreement_rate": pr.psc_agreement_rate,
            }
            for pr in r.prefix_results
        ],
        "entropy_curve_summary": {
            "length": len(r.entropy_curve),
            "mean": float(np.mean(r.entropy_curve)) if r.entropy_curve else None,
            "std": float(np.std(r.entropy_curve)) if r.entropy_curve else None,
            # Store sampled points (every 10%) to keep JSONL manageable
            "sampled": [
                r.entropy_curve[int(f * len(r.entropy_curve))]
                for f in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                if r.entropy_curve and int(f * len(r.entropy_curve)) < len(r.entropy_curve)
            ],
        },
        # Full entropy curve stored separately to avoid bloating JSONL
        "entropy_curve": r.entropy_curve,
    }


async def _async_main(config: ExperimentConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.makedirs(config.output_dir, exist_ok=True)

    # Setup
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    logger.info(f"Model: {config.model_name}, Renderer: {renderer_name}")

    # Create sampling client (no training needed — we just sample from the base model)
    service_client = tinker.ServiceClient(base_url=config.base_url)
    sampling_client = service_client.create_sampling_client(base_model=config.model_name)

    # Load problems
    problems = load_problems(config.n_problems, config.benchmark)
    logger.info(f"Loaded {len(problems)} problems from {config.benchmark}")

    # Few-shot prefix (benchmark-specific)
    if config.benchmark == "arc-challenge":
        convo_prefix = []  # zero-shot for MC
    else:
        convo_prefix = MathEnv.standard_fewshot_prefix()

    # Output file
    output_path = os.path.join(config.output_dir, "results.jsonl")
    logger.info(f"Writing results to {output_path}")

    t0 = time.time()
    n_processed = 0

    with open(output_path, "w") as f:
        for pi, problem in enumerate(problems):
            # Build prompt
            if config.benchmark == "arc-challenge":
                question = problem["problem"] + "\n\nThink step by step, then give your answer."
            else:
                question = problem["problem"] + MathEnv.question_suffix()
            convo = [*convo_prefix, {"role": "user", "content": question}]
            prompt_mi = renderer.build_generation_prompt(convo)

            logger.info(
                f"[{pi+1}/{len(problems)}] Problem {problem['idx']} "
                f"(Level {problem['level']}, {problem['subject']})"
            )

            try:
                result = await process_problem(
                    problem, prompt_mi, sampling_client, renderer, tokenizer, config
                )
                result_dict = result_to_dict(result)
                f.write(json.dumps(result_dict) + "\n")
                f.flush()

                # Brief status
                status_parts = [
                    f"correct={result.n_correct_rollouts}/{result.n_total_rollouts}",
                    f"len={result.selected_rollout_len}",
                ]
                if result.commitment_fraction is not None:
                    status_parts.append(f"commit@{result.commitment_fraction:.0%}")
                    status_parts.append(f"theater={result.theater_fraction:.0%}")
                logger.info(f"  -> {', '.join(status_parts)}")

            except Exception as e:
                logger.error(f"  -> FAILED: {e}")
                continue

            n_processed += 1

    elapsed = time.time() - t0
    logger.info(f"Done: {n_processed}/{len(problems)} problems in {elapsed:.0f}s")

    # Save config
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_name": config.model_name,
            "renderer_name": renderer_name,
            "benchmark": config.benchmark,
            "n_problems": config.n_problems,
            "n_full_rollouts": config.n_full_rollouts,
            "max_tokens": config.max_tokens,
            "prefix_fractions": config.prefix_fractions,
            "n_prefix_probes": config.n_prefix_probes,
            "topk_for_entropy": config.topk_for_entropy,
            "efa_max_tokens": config.efa_max_tokens,
            "seed": config.seed,
        }, f, indent=2)

    # Print quick summary
    print_summary(output_path)


def print_summary(results_path: str):
    """Print a quick summary of experiment results."""
    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        print("No results found.")
        return

    print(f"\n{'='*70}")
    print("REASONING THEATER EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Total problems: {len(results)}")

    # Accuracy
    correct_problems = [r for r in results if r["n_correct_rollouts"] > 0]
    print(f"Problems with >= 1 correct rollout: {len(correct_problems)}/{len(results)} "
          f"({100*len(correct_problems)/len(results):.1f}%)")

    # Commitment analysis (only for problems with correct rollouts)
    committed = [r for r in correct_problems if r["commitment_fraction"] is not None]
    if committed:
        fracs = [r["commitment_fraction"] for r in committed]
        theater = [r["theater_fraction"] for r in committed]
        print(f"\nCommitment Analysis ({len(committed)} problems with correct rollouts):")
        print(f"  Mean commitment point: {np.mean(fracs):.1%} through CoT")
        print(f"  Median commitment point: {np.median(fracs):.1%}")
        print(f"  Mean theater fraction: {np.mean(theater):.1%} of CoT is 'theater'")

        # By level
        by_level: dict[str, list[float]] = {}
        for r in committed:
            by_level.setdefault(r["level"], []).append(r["commitment_fraction"])
        print(f"\n  Commitment by difficulty level:")
        for level in sorted(by_level.keys()):
            vals = by_level[level]
            print(f"    {level}: mean={np.mean(vals):.1%}, n={len(vals)}")

    # EFA accuracy curve
    print(f"\nProtocol A — Early Forced Answering (correct problems only):")
    for frac_idx, frac in enumerate([0.10, 0.25, 0.50, 0.75, 0.90]):
        n_efa_correct = 0
        n_total = 0
        for r in correct_problems:
            if frac_idx < len(r["prefix_results"]):
                pr = r["prefix_results"][frac_idx]
                if pr["efa_correct"]:
                    n_efa_correct += 1
                n_total += 1
        if n_total > 0:
            print(f"  {frac:.0%} prefix: {n_efa_correct}/{n_total} correct ({100*n_efa_correct/n_total:.1f}%)")

    # ATLT curve
    print(f"\nProtocol B — Answer Token Logprob Trajectory:")
    for frac_idx, frac in enumerate([0.10, 0.25, 0.50, 0.75, 0.90]):
        lps = []
        for r in correct_problems:
            if frac_idx < len(r["prefix_results"]):
                lp = r["prefix_results"][frac_idx]["atlt_logprob"]
                if lp is not None:
                    lps.append(lp)
        if lps:
            print(f"  {frac:.0%} prefix: mean logprob = {np.mean(lps):.3f} (n={len(lps)})")

    # PSC curve
    print(f"\nProtocol D — Prefix Self-Consistency:")
    for frac_idx, frac in enumerate([0.10, 0.25, 0.50, 0.75, 0.90]):
        rates = []
        for r in correct_problems:
            if frac_idx < len(r["prefix_results"]):
                rate = r["prefix_results"][frac_idx]["psc_agreement_rate"]
                rates.append(rate)
        if rates:
            print(f"  {frac:.0%} prefix: mean agreement = {np.mean(rates):.1%} (n={len(rates)})")

    print(f"{'='*70}\n")


def main(config: ExperimentConfig):
    import asyncio
    asyncio.run(_async_main(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
