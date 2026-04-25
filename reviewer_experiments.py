"""
Reviewer response experiments:

1. Fine-grained checkpoints (W4): Run PSC/EFA at 2%-5% intervals on 50 problems.
2. PSC raw answers + self-agreement (W5): Store raw continuation answers,
   analyze self-agreement on wrong problems.

Usage:
    python -m reviewer_experiments \
        --experiment fine_grained --n_problems 50
    python -m reviewer_experiments \
        --experiment psc_raw --n_problems 100
"""

import asyncio
import json
import logging
import os
import time
from collections import Counter

import numpy as np
import tinker
import tinker.types as types

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from experiment import (
    ExperimentConfig,
    MathEnv,
    load_problems,
    run_efa,
    try_extract_and_grade,
    safe_grade,
    _is_mc_benchmark,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


# ─────────────────────────────────────────────
# Experiment 1: Fine-grained checkpoints (W4)
# ─────────────────────────────────────────────

FINE_FRACTIONS = [0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


async def run_fine_grained(
    model_name: str = "Qwen/Qwen3-32B",
    renderer_name: str = "qwen3",
    n_problems: int = 50,
    output_dir: str = "/tmp/tinker-examples/reasoning_theater/fine_grained_checkpoints",
    n_psc_samples: int = 8,
    max_tokens: int = 4096,
):
    """Run PSC + EFA at fine-grained checkpoints."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.jsonl")

    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    problems = load_problems(n_problems, "math-500")
    logger.info(f"Loaded {len(problems)} problems")

    convo_prefix = MathEnv.standard_fewshot_prefix()

    # Resume support
    done_idxs: set[int] = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        done_idxs.add(json.loads(line)["problem_idx"])
                    except Exception:
                        pass
        if done_idxs:
            logger.info(f"Resuming: {len(done_idxs)} done")

    config = ExperimentConfig(
        model_name=model_name,
        max_tokens=max_tokens,
        n_prefix_probes=n_psc_samples,
    )

    with open(output_path, "a") as f:
        for pi, problem in enumerate(problems):
            if problem["idx"] in done_idxs:
                continue

            question = problem["problem"] + MathEnv.question_suffix()
            convo = [*convo_prefix, {"role": "user", "content": question}]
            prompt_mi = renderer.build_generation_prompt(convo)
            ground_truth = problem["answer"]

            # Generate full CoT
            try:
                sample_result = await sampling_client.sample_async(
                    prompt=prompt_mi,
                    num_samples=4,
                    sampling_params=types.SamplingParams(
                        max_tokens=max_tokens,
                        stop=renderer.get_stop_sequences(),
                        temperature=1.0,
                    ),
                )
            except Exception as e:
                logger.error(f"[{pi+1}] idx={problem['idx']} rollout failed: {e}")
                continue

            rollouts = []
            for seq in sample_result.sequences:
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = renderers.get_text_content(parsed_msg)
                correct, answer = try_extract_and_grade(content, ground_truth, "sympy", 2.0)
                rollouts.append({"tokens": seq.tokens, "correct": correct, "answer": answer})

            n_correct = sum(1 for r in rollouts if r["correct"])
            selected = next((r for r in rollouts if r["correct"]), rollouts[0])
            cot_tokens = selected["tokens"]

            if len(cot_tokens) < 20:
                logger.warning(f"[{pi+1}] idx={problem['idx']} CoT too short ({len(cot_tokens)})")
                continue

            # Run PSC + EFA at each fine-grained checkpoint
            checkpoint_results = []

            async def run_checkpoint(frac):
                pos = max(1, int(frac * len(cot_tokens)))
                pos = min(pos, len(cot_tokens) - 1)
                prefix = cot_tokens[:pos]
                probe_mi = prompt_mi.append(types.EncodedTextChunk(tokens=prefix))

                remaining = max(len(cot_tokens) - pos, 64)
                psc_max = max(min(remaining * 2, max_tokens), 256)

                # PSC
                psc_result = await sampling_client.sample_async(
                    prompt=probe_mi,
                    num_samples=n_psc_samples,
                    sampling_params=types.SamplingParams(
                        max_tokens=psc_max,
                        stop=renderer.get_stop_sequences(),
                        temperature=1.0,
                    ),
                )
                psc_correct = 0
                for seq in psc_result.sequences:
                    parsed_msg, _ = renderer.parse_response(seq.tokens)
                    content = renderers.get_text_content(parsed_msg)
                    ok, _ = try_extract_and_grade(content, ground_truth, "sympy", 2.0)
                    if ok:
                        psc_correct += 1

                # EFA
                try:
                    efa_ans, efa_ok = await run_efa(
                        sampling_client, prompt_mi, prefix, ground_truth, renderer, tokenizer, config
                    )
                except Exception:
                    efa_ans, efa_ok = None, False

                return {
                    "fraction": frac,
                    "prefix_len": pos,
                    "total_len": len(cot_tokens),
                    "psc_correct": psc_correct,
                    "psc_total": n_psc_samples,
                    "psc_rate": psc_correct / n_psc_samples,
                    "efa_correct": efa_ok,
                    "efa_answer": str(efa_ans)[:100] if efa_ans else None,
                }

            try:
                results = await asyncio.gather(*[run_checkpoint(frac) for frac in FINE_FRACTIONS])
            except Exception as e:
                logger.error(f"[{pi+1}] idx={problem['idx']} checkpoint failed: {e}")
                continue
            checkpoint_results = sorted(results, key=lambda x: x["fraction"])

            record = {
                "problem_idx": problem["idx"],
                "level": problem["level"],
                "subject": problem["subject"],
                "ground_truth": ground_truth,
                "n_correct_rollouts": n_correct,
                "cot_len": len(cot_tokens),
                "selected_correct": selected["correct"],
                "checkpoints": checkpoint_results,
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

            # Log
            psc_at_5 = next((c for c in checkpoint_results if c["fraction"] == 0.05), None)
            psc_at_10 = next((c for c in checkpoint_results if c["fraction"] == 0.10), None)
            logger.info(
                f"[{pi+1}/{len(problems)}] idx={problem['idx']} L{problem['level']} "
                f"cot={len(cot_tokens)} correct={n_correct}/4 "
                f"PSC@5%={psc_at_5['psc_rate']:.0%} PSC@10%={psc_at_10['psc_rate']:.0%}"
            )

    logger.info("Fine-grained experiment complete")
    _summarize_fine_grained(output_path)


def _summarize_fine_grained(path: str):
    results = [json.loads(l) for l in open(path) if l.strip()]
    solvable = [r for r in results if r["n_correct_rollouts"] > 0]
    print(f"\nFine-grained summary (n={len(results)}, solvable={len(solvable)})")
    print(f"{'frac':>6} {'PSC_mean':>9} {'EFA_acc':>8} {'gap':>6}")
    for frac in FINE_FRACTIONS:
        psc_vals = []
        efa_correct = 0
        for r in solvable:
            cp = next((c for c in r["checkpoints"] if c["fraction"] == frac), None)
            if cp:
                psc_vals.append(cp["psc_rate"])
                if cp["efa_correct"]:
                    efa_correct += 1
        if psc_vals:
            mean_psc = np.mean(psc_vals) * 100
            efa_rate = efa_correct / len(solvable) * 100
            print(f"  {frac:>5.0%} {mean_psc:>8.1f}% {efa_rate:>7.1f}% {mean_psc-efa_rate:>5.1f}pp")


# ─────────────────────────────────────────────
# Experiment 2: PSC raw answers (W5)
# ─────────────────────────────────────────────

async def run_psc_raw(
    model_name: str = "Qwen/Qwen3-32B",
    renderer_name: str = "qwen3",
    n_problems: int = 100,
    output_dir: str = "/tmp/tinker-examples/reasoning_theater/psc_raw_answers",
    n_psc_samples: int = 8,
    max_tokens: int = 4096,
    benchmark: str = "math-500",
):
    """Run PSC storing raw continuation answers for self-agreement analysis."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.jsonl")

    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    is_mc = _is_mc_benchmark(benchmark)
    grader = "exact" if is_mc else "sympy"

    problems = load_problems(n_problems, benchmark)
    logger.info(f"Loaded {len(problems)} problems from {benchmark}")

    convo_prefix = MathEnv.standard_fewshot_prefix() if not is_mc else []

    done_idxs: set[int] = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        done_idxs.add(json.loads(line)["problem_idx"])
                    except Exception:
                        pass
        if done_idxs:
            logger.info(f"Resuming: {len(done_idxs)} done")

    # Checkpoints: focus on early ones where it matters
    fractions = [0.10, 0.20, 0.30, 0.50]

    with open(output_path, "a") as f:
        for pi, problem in enumerate(problems):
            if problem["idx"] in done_idxs:
                continue

            question = problem["problem"] if is_mc else problem["problem"] + MathEnv.question_suffix()
            convo = [*convo_prefix, {"role": "user", "content": question}]
            prompt_mi = renderer.build_generation_prompt(convo)
            ground_truth = problem["answer"]

            try:
                # Generate full CoT (4 rollouts)
                sample_result = await sampling_client.sample_async(
                    prompt=prompt_mi,
                    num_samples=4,
                    sampling_params=types.SamplingParams(
                        max_tokens=max_tokens,
                        stop=renderer.get_stop_sequences(),
                        temperature=1.0,
                    ),
                )
            except Exception as e:
                logger.error(f"[{pi+1}/{len(problems)}] idx={problem['idx']} rollout failed: {e}")
                continue

            rollouts = []
            for seq in sample_result.sequences:
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = renderers.get_text_content(parsed_msg)
                correct, answer = try_extract_and_grade(content, ground_truth, grader, 2.0)
                rollouts.append({"tokens": seq.tokens, "correct": correct, "answer": answer})

            n_correct = sum(1 for r in rollouts if r["correct"])
            selected = next((r for r in rollouts if r["correct"]), rollouts[0])
            cot_tokens = selected["tokens"]

            if len(cot_tokens) < 20:
                continue

            # Run PSC at each checkpoint, store raw answers
            checkpoint_results = []

            async def run_checkpoint_raw(frac):
                pos = max(1, int(frac * len(cot_tokens)))
                pos = min(pos, len(cot_tokens) - 1)
                prefix = cot_tokens[:pos]
                probe_mi = prompt_mi.append(types.EncodedTextChunk(tokens=prefix))

                remaining = max(len(cot_tokens) - pos, 64)
                psc_max = max(min(remaining * 2, max_tokens), 256)

                psc_result = await sampling_client.sample_async(
                    prompt=probe_mi,
                    num_samples=n_psc_samples,
                    sampling_params=types.SamplingParams(
                        max_tokens=psc_max,
                        stop=renderer.get_stop_sequences(),
                        temperature=1.0,
                    ),
                )

                answers = []
                correct_count = 0
                continuation_lens = []
                for seq in psc_result.sequences:
                    parsed_msg, _ = renderer.parse_response(seq.tokens)
                    content = renderers.get_text_content(parsed_msg)
                    ok, extracted = try_extract_and_grade(content, ground_truth, grader, 2.0)
                    answers.append(extracted)
                    continuation_lens.append(len(seq.tokens))
                    if ok:
                        correct_count += 1

                # Self-agreement: fraction of pairs that agree
                valid_answers = [a for a in answers if a is not None]
                if len(valid_answers) >= 2:
                    n_pairs = 0
                    n_agree = 0
                    for i in range(len(valid_answers)):
                        for j in range(i + 1, len(valid_answers)):
                            n_pairs += 1
                            if safe_grade(valid_answers[i], valid_answers[j], grader, 2.0):
                                n_agree += 1
                    self_agreement = n_agree / n_pairs if n_pairs > 0 else 0
                else:
                    self_agreement = None

                # Most common answer
                if valid_answers:
                    majority = Counter(valid_answers).most_common(1)[0]
                    majority_answer = majority[0]
                    majority_count = majority[1]
                else:
                    majority_answer = None
                    majority_count = 0

                return {
                    "fraction": frac,
                    "prefix_len": pos,
                    "total_len": len(cot_tokens),
                    "psc_correct": correct_count,
                    "psc_total": n_psc_samples,
                    "psc_accuracy": correct_count / n_psc_samples,
                    "raw_answers": [str(a)[:200] if a else None for a in answers],
                    "continuation_lens": continuation_lens,
                    "mean_continuation_len": int(np.mean(continuation_lens)),
                    "self_agreement": self_agreement,
                    "majority_answer": str(majority_answer)[:200] if majority_answer else None,
                    "majority_count": majority_count,
                    "majority_correct": safe_grade(majority_answer, ground_truth, grader, 2.0) if majority_answer else False,
                }

            try:
                results = await asyncio.gather(*[run_checkpoint_raw(frac) for frac in fractions])
            except Exception as e:
                logger.error(f"[{pi+1}/{len(problems)}] idx={problem['idx']} checkpoint failed: {e}")
                continue
            checkpoint_results = sorted(results, key=lambda x: x["fraction"])

            record = {
                "problem_idx": problem["idx"],
                "level": problem["level"],
                "ground_truth": ground_truth,
                "n_correct_rollouts": n_correct,
                "cot_len": len(cot_tokens),
                "selected_correct": selected["correct"],
                "checkpoints": checkpoint_results,
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

            solvable = "✓" if n_correct > 0 else "✗"
            cp10 = next((c for c in checkpoint_results if c["fraction"] == 0.10), None)
            if cp10:
                sa = cp10['self_agreement']
                sa_str = f"{sa:.2f}" if sa is not None else "N/A"
                logger.info(
                    f"[{pi+1}/{len(problems)}] idx={problem['idx']} {solvable} "
                    f"PSC@10%={cp10['psc_accuracy']:.0%} self_agree={sa_str} "
                    f"cont_len={cp10['mean_continuation_len']}"
                )

    logger.info("PSC raw answers experiment complete")
    _summarize_psc_raw(output_path)


def _summarize_psc_raw(path: str):
    results = [json.loads(l) for l in open(path) if l.strip()]
    solvable = [r for r in results if r["n_correct_rollouts"] > 0]
    wrong = [r for r in results if r["n_correct_rollouts"] == 0]

    print(f"\nPSC Raw Answers Summary (n={len(results)}, solvable={len(solvable)}, wrong={len(wrong)})")

    for frac in [0.10, 0.20, 0.30, 0.50]:
        print(f"\n  === f={frac:.0%} ===")

        # Solvable
        if solvable:
            psc_accs = [next(c for c in r["checkpoints"] if c["fraction"] == frac)["psc_accuracy"] for r in solvable]
            self_agrees = [next(c for c in r["checkpoints"] if c["fraction"] == frac)["self_agreement"] for r in solvable if next(c for c in r["checkpoints"] if c["fraction"] == frac)["self_agreement"] is not None]
            cont_lens = [next(c for c in r["checkpoints"] if c["fraction"] == frac)["mean_continuation_len"] for r in solvable]
            print(f"    Solvable: PSC_acc={np.mean(psc_accs):.1%} self_agree={np.mean(self_agrees):.1%} cont_len={np.mean(cont_lens):.0f}")

        # Wrong problems — key for W5
        if wrong:
            wrong_agrees = []
            wrong_majority_wrong = 0
            for r in wrong:
                cp = next(c for c in r["checkpoints"] if c["fraction"] == frac)
                if cp["self_agreement"] is not None:
                    wrong_agrees.append(cp["self_agreement"])
                if cp["majority_count"] >= 6:  # high agreement on majority
                    wrong_majority_wrong += 1

            if wrong_agrees:
                print(f"    Wrong:    self_agree={np.mean(wrong_agrees):.1%} "
                      f"high_agree(>=6/8)={wrong_majority_wrong}/{len(wrong)} "
                      f"max_agree={max(wrong_agrees):.1%}")

    # Actual continuation token counts (Q2)
    print(f"\n  === Actual continuation token counts (Q2) ===")
    for frac in [0.10, 0.20, 0.50]:
        lens = []
        for r in solvable:
            cp = next((c for c in r["checkpoints"] if c["fraction"] == frac), None)
            if cp:
                lens.append(cp["mean_continuation_len"])
        if lens:
            cot_lens = [r["cot_len"] for r in solvable]
            mean_remaining = np.mean([c * (1 - frac) for c in cot_lens])
            print(f"    f={frac:.0%}: actual_cont={np.mean(lens):.0f} vs remaining_cot={mean_remaining:.0f} "
                  f"(ratio={np.mean(lens)/mean_remaining:.2f})")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["fine_grained", "psc_raw"], required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen3-32B")
    parser.add_argument("--renderer_name", default="qwen3")
    parser.add_argument("--n_problems", type=int, default=50)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.experiment == "fine_grained":
        output = args.output_dir or "/tmp/tinker-examples/reasoning_theater/fine_grained_checkpoints"
        asyncio.run(run_fine_grained(
            model_name=args.model_name,
            renderer_name=args.renderer_name,
            n_problems=args.n_problems,
            output_dir=output,
        ))
    elif args.experiment == "psc_raw":
        output = args.output_dir or "/tmp/tinker-examples/reasoning_theater/psc_raw_answers"
        asyncio.run(run_psc_raw(
            model_name=args.model_name,
            renderer_name=args.renderer_name,
            n_problems=args.n_problems,
            output_dir=output,
        ))
