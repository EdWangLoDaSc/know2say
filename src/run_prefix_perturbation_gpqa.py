"""Prefix perturbation on GPQA-Diamond with GPT-OSS-120B, f=0.10 and f=0.50."""

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
from tinker_cookbook import renderers
from experiment import (
    ExperimentConfig,
    load_problems,
    run_psc,
    _is_mc_benchmark,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_name = "openai/gpt-oss-120b"
    renderer_name = "gpt_oss_medium_reasoning"
    main_results_path = paths.paper_data("gpqa_gpt_oss_120b")
    output_dir = paths.results("prefix_perturbation_gpqa")
    os.makedirs(output_dir, exist_ok=True)

    # Load main results, filter committed+correct at f=0.10
    main_results = []
    with open(main_results_path) as f:
        for line in f:
            if line.strip():
                main_results.append(json.loads(line))

    committed = []
    for r in main_results:
        for pr in r['prefix_results']:
            if abs(pr['fraction'] - 0.10) < 0.01 and pr['psc_agreement_rate'] >= 0.75:
                if r['selected_rollout_correct']:
                    committed.append(r)
                    break

    logger.info(f"Found {len(committed)} committed+correct problems at f=0.10")

    rng = random.Random(42)
    rng.shuffle(committed)
    subset = committed[:50]
    logger.info(f"Using {len(subset)} problems")

    # Setup
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    config = ExperimentConfig(
        model_name=model_name,
        renderer_name=renderer_name,
        benchmark="gpqa-diamond",
        n_problems=198,
        max_tokens=8192,
        prefix_fractions=(0.10, 0.50),
    )

    problems_map = {p['idx']: p for p in load_problems(198, "gpqa-diamond")}
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 150000

    output_path = os.path.join(output_dir, "results.jsonl")

    # Resume
    done_keys = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        done_keys.add((r['problem_idx'], r['fraction']))
                    except Exception:
                        pass
        if done_keys:
            logger.info(f"Resuming: {len(done_keys)} already done")

    t0 = time.time()

    with open(output_path, "a") as f:
        for i, r in enumerate(subset):
            idx = r['problem_idx']
            problem = problems_map.get(idx)
            if problem is None:
                continue

            question = problem["problem"]
            convo = [{"role": "user", "content": question}]
            prompt_mi = renderer.build_generation_prompt(convo)
            ground_truth = problem["answer"]

            # Generate one rollout
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
                logger.warning(f"[{i+1}] idx={idx} CoT too short, skip")
                continue

            for frac in [0.10, 0.50]:
                if (idx, frac) in done_keys:
                    continue

                prefix_len = max(1, int(frac * len(cot_tokens)))
                prefix = cot_tokens[:prefix_len]

                def _truncate(tokens, pct=0.20):
                    cut = max(1, int(len(tokens) * (1 - pct)))
                    return tokens[:cut]

                def _shuffle_tail(tokens, pct=0.30):
                    cut = max(1, int(len(tokens) * (1 - pct)))
                    head = tokens[:cut]
                    tail = list(tokens[cut:])
                    rng.shuffle(tail)
                    return head + tail

                def _random_tail(tokens, pct=0.30):
                    cut = max(1, int(len(tokens) * (1 - pct)))
                    head = tokens[:cut]
                    rand_tail = [rng.randint(100, vocab_size - 1) for _ in range(len(tokens) - cut)]
                    return head + rand_tail

                perturbations = {
                    "intact": prefix,
                    "truncate_20pct": _truncate(prefix),
                    "shuffle_30pct": _shuffle_tail(prefix),
                    "random_30pct": _random_tail(prefix),
                }

                result = {"problem_idx": idx, "prefix_len": prefix_len, "cot_len": len(cot_tokens), "fraction": frac}

                for pname, ptokens in perturbations.items():
                    try:
                        n_corr, n_total = await run_psc(
                            sampling_client, prompt_mi, ptokens, ground_truth,
                            renderer, tokenizer, config,
                            total_cot_len=len(cot_tokens),
                            grader="exact",
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
                rand_rate = result.get("random_30pct", {}).get("psc_rate", 0)
                logger.info(
                    f"[{i+1}/{len(subset)}] idx={idx} f={frac:.2f} "
                    f"intact={intact_rate:.0%} rand={rand_rate:.0%}"
                )

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.0f}s")

    # Summary
    results = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    for frac in [0.10, 0.50]:
        frac_results = [r for r in results if abs(r['fraction'] - frac) < 0.01]
        if frac_results:
            print(f"\nf={frac:.2f} (n={len(frac_results)}):")
            for p in ['intact', 'truncate_20pct', 'shuffle_30pct', 'random_30pct']:
                rates = [r[p]['psc_rate'] for r in frac_results if p in r and 'psc_rate' in r[p]]
                if rates:
                    print(f"  {p}: mean={np.mean(rates):.1%}")


if __name__ == "__main__":
    asyncio.run(main())
