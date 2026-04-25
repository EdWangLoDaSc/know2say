"""Reasoning theater experiment on HumanEval (code generation).

PSC for code: sample N continuations from a prefix, execute each, check if all
test cases pass. "Correct" = passes all unit tests.

EFA for code: append "```python\n" suffix to force immediate code output.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import subprocess

import paths

paths.setup_path()

import numpy as np
import tinker
import tinker.types as types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


# ── HumanEval grading ──────────────────────────────────────────────────

def extract_code_block(text: str) -> str:
    """Extract code from model output. Handles ```python blocks and raw code."""
    import re
    # Try to find ```python ... ``` block
    match = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try ``` ... ``` block
    match = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If text starts with def/class, take it as code
    lines = text.strip().split('\n')
    code_lines = []
    started = False
    for line in lines:
        if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
            started = True
        if started:
            code_lines.append(line)
    if code_lines:
        return '\n'.join(code_lines)
    return text.strip()


def grade_humaneval(completion: str, problem: dict, timeout: float = 5.0) -> bool:
    """Execute completion against HumanEval test cases. Returns True if all pass."""
    code = extract_code_block(completion)
    # Build full program: prompt (function signature) + completion + test
    full_code = problem['prompt'] + code + "\n" + problem['test'] + f"\ncheck({problem['entry_point']})\n"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, timeout=timeout, text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
        finally:
            os.unlink(f.name)


# ── Main experiment ────────────────────────────────────────────────────

async def run_experiment(
    model_name: str,
    renderer_name: str,
    output_dir: str,
    n_problems: int = 164,
    max_tokens: int = 4096,
    n_rollouts: int = 4,
    n_psc: int = 8,
    prefix_fractions: tuple = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90),
    problem_start: int = 0,
    problem_end: int = 164,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(output_dir, exist_ok=True)

    # Load HumanEval
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = []
    for i, row in enumerate(ds):
        if i >= n_problems:
            break
        problems.append({
            "idx": i,
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "test": row["test"],
            "entry_point": row["entry_point"],
            "canonical_solution": row["canonical_solution"],
        })
    # Filter to shard range
    problems = [p for p in problems if problem_start <= p["idx"] < problem_end]
    logger.info(f"Loaded {len(problems)} HumanEval problems (idx {problem_start}--{problem_end-1})")

    # Setup
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    logger.info(f"Model: {model_name}, Renderer: {renderer_name}")

    # EFA suffix for code
    efa_suffix = "\n```python\n"

    output_path = os.path.join(output_dir, "results.jsonl")

    # Resume
    done_idxs = set()
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

    t0 = time.time()

    # System prompt for code generation
    system_prompt = "You are an expert Python programmer. Solve the given coding problem step by step, then provide your solution."

    with open(output_path, "a") as f:
        for pi, problem in enumerate(problems):
            if problem["idx"] in done_idxs:
                continue

            # Build prompt
            question = (
                f"Complete the following Python function:\n\n"
                f"```python\n{problem['prompt']}```\n\n"
                f"Think step by step about the approach, then write the complete function body."
            )
            convo = [{"role": "user", "content": question}]
            prompt_mi = renderer.build_generation_prompt(convo)

            logger.info(f"[{pi+1}/{len(problems)}] {problem['task_id']}")

            try:
                # Step 1: Generate full rollouts
                sp = types.SamplingParams(
                    max_tokens=max_tokens,
                    stop=renderer.get_stop_sequences(),
                    temperature=1.0,
                )
                sample_result = await sampling_client.sample_async(
                    prompt=prompt_mi, num_samples=n_rollouts, sampling_params=sp
                )

                rollout_results = []
                for seq in sample_result.sequences:
                    text = tokenizer.decode(seq.tokens)
                    correct = grade_humaneval(text, problem)
                    rollout_results.append({"correct": correct, "len": len(seq.tokens)})

                n_correct = sum(r["correct"] for r in rollout_results)

                # Pick the first rollout for prefix analysis
                selected = sample_result.sequences[0]
                cot_tokens = list(selected.tokens)
                selected_correct = rollout_results[0]["correct"]

                if len(cot_tokens) < 20:
                    logger.warning(f"  Too short ({len(cot_tokens)} tokens), skip")
                    continue

                # Step 2: Prefix analysis
                prefix_results = []
                commitment_fraction = None

                for frac in prefix_fractions:
                    prefix_len = max(1, int(frac * len(cot_tokens)))
                    prefix = cot_tokens[:prefix_len]

                    # PSC: sample N continuations from prefix
                    psc_mi = prompt_mi.append(types.EncodedTextChunk(tokens=prefix))
                    remaining = len(cot_tokens) - prefix_len
                    cont_budget = min(2 * remaining, max_tokens)

                    psc_sp = types.SamplingParams(
                        max_tokens=cont_budget,
                        stop=renderer.get_stop_sequences(),
                        temperature=1.0,
                    )
                    psc_result = await sampling_client.sample_async(
                        prompt=psc_mi, num_samples=n_psc, sampling_params=psc_sp
                    )

                    psc_correct = 0
                    for seq in psc_result.sequences:
                        # Combine prefix text + continuation for grading
                        prefix_text = tokenizer.decode(prefix)
                        full_text = prefix_text + tokenizer.decode(seq.tokens)
                        if grade_humaneval(full_text, problem):
                            psc_correct += 1

                    psc_rate = psc_correct / n_psc

                    # EFA: force code output
                    efa_tokens = tokenizer.encode(efa_suffix, add_special_tokens=False)
                    efa_combined = list(prefix) + list(efa_tokens)
                    efa_mi = prompt_mi.append(types.EncodedTextChunk(tokens=efa_combined))
                    efa_sp = types.SamplingParams(
                        max_tokens=512, stop=["```", "\n\n\n"], temperature=0.0,
                    )
                    try:
                        efa_result = await sampling_client.sample_async(
                            prompt=efa_mi, num_samples=1, sampling_params=efa_sp
                        )
                        efa_text = efa_result.sequences[0].text
                        efa_correct = grade_humaneval(efa_text, problem)
                    except Exception:
                        efa_correct = False

                    # Track commitment
                    if commitment_fraction is None and psc_rate >= 0.75:
                        commitment_fraction = frac

                    prefix_results.append({
                        "fraction": frac,
                        "prefix_len": prefix_len,
                        "psc_n_correct": psc_correct,
                        "psc_agreement_rate": psc_rate,
                        "efa_correct": efa_correct,
                    })

                result = {
                    "problem_idx": problem["idx"],
                    "task_id": problem["task_id"],
                    "n_correct_rollouts": n_correct,
                    "n_total_rollouts": n_rollouts,
                    "selected_rollout_correct": selected_correct,
                    "selected_rollout_len": len(cot_tokens),
                    "commitment_fraction": commitment_fraction,
                    "theater_fraction": (1 - commitment_fraction) if commitment_fraction else None,
                    "prefix_results": prefix_results,
                    "level": "code",
                    "subject": "Code",
                }

                f.write(json.dumps(result) + "\n")
                f.flush()

                status = f"correct={n_correct}/{n_rollouts}, len={len(cot_tokens)}"
                if commitment_fraction is not None:
                    status += f", commit@{commitment_fraction:.0%}, theater={1-commitment_fraction:.0%}"
                logger.info(f"  -> {status}")

            except Exception as e:
                logger.error(f"  -> FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.0f}s")

    # Summary
    results = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if results:
        solvable = [r for r in results if r['n_correct_rollouts'] > 0]
        commits = [r['commitment_fraction'] for r in solvable if r['commitment_fraction'] is not None]
        print(f"\n{'='*60}")
        print(f"HumanEval Summary: {model_name}")
        print(f"  Problems: {len(results)}, Solvable: {len(solvable)}")
        print(f"  Accuracy: {sum(r['selected_rollout_correct'] for r in results)/len(results):.1%}")
        if commits:
            print(f"  Mean commitment: {np.mean(commits):.1%}")
            print(f"  Mean theater: {1-np.mean(commits):.1%}")
        print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--renderer", default="qwen3")
    parser.add_argument("--output-dir", default="/tmp/tinker-examples/reasoning_theater/humaneval_32b_think")
    parser.add_argument("--n-problems", type=int, default=164)
    parser.add_argument("--start", type=int, default=0, help="Start problem index (inclusive)")
    parser.add_argument("--end", type=int, default=164, help="End problem index (exclusive)")
    args = parser.parse_args()

    # Override output dir with shard suffix if sharded
    output_dir = args.output_dir
    if args.start > 0 or args.end < 164:
        output_dir = f"{args.output_dir}_shard_{args.start}_{args.end}"

    asyncio.run(run_experiment(
        model_name=args.model,
        renderer_name=args.renderer,
        output_dir=output_dir,
        n_problems=args.n_problems,
        problem_start=args.start,
        problem_end=args.end,
    ))
