"""Prefix perturbation at f=0.50: longer prefixes should show more perturbation sensitivity."""

import asyncio
import json
import logging
import os
import random
import sys
import time

import numpy as np

import paths

paths.setup_path()

import tinker
import tinker.types as types
from tinker_cookbook import model_info, renderers
from experiment import (
    ExperimentConfig,
    load_problems,
    run_psc,
    safe_grade,
)
from tinker_cookbook.recipes.math_rl.math_env import MathEnv
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

# Use the same 50 problem indices as f=0.10 experiment
SAME_PROBLEM_IDXS = None  # Will load from f=0.10 results


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_name = "Qwen/Qwen3-32B"
    renderer_name = "qwen3"
    main_results_path = paths.results_jsonl("qwen3_32b_thinking_full500")
    f10_perturbation_path = paths.results_jsonl("prefix_perturbation")
    output_dir = paths.results("prefix_perturbation_f50")
    os.makedirs(output_dir, exist_ok=True)

    # Load the same 50 problem indices used in f=0.10 experiment
    f10_idxs = []
    with open(f10_perturbation_path) as f:
        for line in f:
            if line.strip():
                f10_idxs.append(json.loads(line)['problem_idx'])
    logger.info(f"Using same {len(f10_idxs)} problem indices as f=0.10 experiment")

    # Load main results
    main_results = {}
    with open(main_results_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                main_results[r['problem_idx']] = r

    # Setup
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    config = ExperimentConfig(
        model_name=model_name,
        renderer_name=renderer_name,
        n_problems=500,
        max_tokens=4096,
        prefix_fractions=(0.50,),
    )

    problems_map = {p['idx']: p for p in load_problems(500, "math-500")}
    convo_prefix = MathEnv.standard_fewshot_prefix()
    rng = random.Random(42)
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 150000

    output_path = os.path.join(output_dir, "results.jsonl")

    # Resume support
    done_idxs = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        done_idxs.add(json.loads(line)['problem_idx'])
                    except Exception:
                        pass
        if done_idxs:
            logger.info(f"Resuming: {len(done_idxs)} already done")

    t0 = time.time()

    with open(output_path, "a") as f:
        for i, idx in enumerate(f10_idxs):
            if idx in done_idxs:
                continue

            problem = problems_map.get(idx)
            if problem is None:
                continue

            question = problem["problem"] + MathEnv.question_suffix()
            convo = [*convo_prefix, {"role": "user", "content": question}]
            prompt_mi = renderer.build_generation_prompt(convo)
            ground_truth = problem["answer"]

            # Generate one rollout to get tokens
            sp = types.SamplingParams(
                max_tokens=config.max_tokens,
                stop=renderer.get_stop_sequences(),
                temperature=0.0,
            )
            sample_result = await sampling_client.sample_async(
                prompt=prompt_mi, num_samples=1, sampling_params=sp
            )
            cot_tokens = list(sample_result.sequences[0].tokens)

            if len(cot_tokens) < 20:
                logger.warning(f"[{i+1}] idx={idx} CoT too short ({len(cot_tokens)}), skip")
                continue

            # f=0.50 prefix
            prefix_len = max(1, int(0.50 * len(cot_tokens)))
            prefix = cot_tokens[:prefix_len]

            def _truncate(tokens, frac=0.20):
                cut = max(1, int(len(tokens) * (1 - frac)))
                return tokens[:cut]

            def _shuffle_tail(tokens, frac=0.30):
                cut = max(1, int(len(tokens) * (1 - frac)))
                head = tokens[:cut]
                tail = list(tokens[cut:])
                rng.shuffle(tail)
                return head + tail

            def _random_tail(tokens, frac=0.30):
                cut = max(1, int(len(tokens) * (1 - frac)))
                head = tokens[:cut]
                rand_tail = [rng.randint(100, vocab_size - 1) for _ in range(len(tokens) - cut)]
                return head + rand_tail

            perturbations = {
                "intact": prefix,
                "truncate_20pct": _truncate(prefix, 0.20),
                "shuffle_30pct": _shuffle_tail(prefix, 0.30),
                "random_30pct": _random_tail(prefix, 0.30),
            }

            result = {"problem_idx": idx, "prefix_len": prefix_len, "cot_len": len(cot_tokens), "fraction": 0.50}

            for pname, ptokens in perturbations.items():
                try:
                    n_corr, n_total = await run_psc(
                        sampling_client, prompt_mi, ptokens, ground_truth,
                        renderer, tokenizer, config,
                        total_cot_len=len(cot_tokens),
                        grader="sympy",
                    )
                    result[pname] = {
                        "psc_n_correct": n_corr,
                        "psc_n_total": n_total,
                        "psc_rate": n_corr / n_total if n_total > 0 else 0,
                        "prefix_len": len(ptokens),
                    }
                except Exception as e:
                    logger.error(f"  {pname} failed: {e}")
                    result[pname] = {"error": str(e)}

            f.write(json.dumps(result) + "\n")
            f.flush()

            intact_rate = result.get("intact", {}).get("psc_rate", 0)
            trunc_rate = result.get("truncate_20pct", {}).get("psc_rate", 0)
            shuf_rate = result.get("shuffle_30pct", {}).get("psc_rate", 0)
            rand_rate = result.get("random_30pct", {}).get("psc_rate", 0)

            logger.info(
                f"[{i+1}/{len(f10_idxs)}] idx={idx} "
                f"intact={intact_rate:.0%} trunc={trunc_rate:.0%} "
                f"shuf={shuf_rate:.0%} rand={rand_rate:.0%}"
            )

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.0f}s")

    # Summary
    results = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    if results:
        for pname in ["intact", "truncate_20pct", "shuffle_30pct", "random_30pct"]:
            rates = [r[pname]["psc_rate"] for r in results if pname in r and "psc_rate" in r[pname]]
            if rates:
                print(f"{pname}: mean PSC = {np.mean(rates):.1%} (n={len(rates)})")


if __name__ == "__main__":
    asyncio.run(main())
