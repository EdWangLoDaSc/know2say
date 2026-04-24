"""Latency benchmark: Full CoT vs BAEE with actual parallel execution."""

import asyncio
import json
import logging
import os
import sys
import time

import paths

paths.setup_path()

import numpy as np
import tinker
import tinker.types as types
from tinker_cookbook import renderers
from tinker_cookbook.recipes.reasoning_theater.experiment import (
    ExperimentConfig, load_problems, try_extract_and_grade,
)
from tinker_cookbook.recipes.math_rl.math_env import MathEnv
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


async def benchmark_latency(
    model_name: str,
    renderer_name: str,
    n_problems: int = 50,
    n_psc: int = 8,
    theta: float = 0.75,
    max_tokens: int = 4096,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    problems = load_problems(n_problems, "math-500", seed=42)
    convo_prefix = MathEnv.standard_fewshot_prefix()

    results = []

    for pi, problem in enumerate(problems):
        question = problem["problem"] + MathEnv.question_suffix()
        convo = [*convo_prefix, {"role": "user", "content": question}]
        prompt_mi = renderer.build_generation_prompt(convo)
        ground_truth = problem["answer"]

        # ── Measure Full CoT latency ──
        sp_full = types.SamplingParams(
            max_tokens=max_tokens,
            stop=renderer.get_stop_sequences(),
            temperature=1.0,
        )
        t0 = time.perf_counter()
        full_result = await sampling_client.sample_async(
            prompt=prompt_mi, num_samples=1, sampling_params=sp_full
        )
        full_latency = time.perf_counter() - t0
        full_tokens = list(full_result.sequences[0].tokens)
        full_len = len(full_tokens)

        # Check correctness
        full_text = tokenizer.decode(full_tokens)
        full_correct, _ = try_extract_and_grade(full_text, ground_truth, "sympy", 2.0)

        if full_len < 50:
            logger.warning(f"[{pi+1}] Too short ({full_len}), skip")
            continue

        # ── Measure BAEE latency (parallel PSC at f=0.10) ──
        prefix_frac = 0.10
        prefix_len = max(1, int(prefix_frac * full_len))
        prefix_tokens = full_tokens[:prefix_len]

        # Step 1: "Generate" prefix (in practice, stream up to 10% then pause)
        # We simulate by measuring the time to generate prefix_len tokens
        sp_prefix = types.SamplingParams(
            max_tokens=prefix_len + 5,  # slightly over to ensure we get prefix_len
            stop=renderer.get_stop_sequences(),
            temperature=1.0,
        )
        t0 = time.perf_counter()
        await sampling_client.sample_async(
            prompt=prompt_mi, num_samples=1, sampling_params=sp_prefix
        )
        prefix_latency = time.perf_counter() - t0

        # Step 2: Parallel PSC probes (N=8 concurrent calls)
        psc_mi = prompt_mi.append(types.EncodedTextChunk(tokens=prefix_tokens))
        remaining = full_len - prefix_len
        cont_budget = min(2 * remaining, max_tokens)
        sp_cont = types.SamplingParams(
            max_tokens=cont_budget,
            stop=renderer.get_stop_sequences(),
            temperature=1.0,
        )

        t0 = time.perf_counter()
        # Single API call with num_samples=N (true server-side parallelism)
        psc_batch_result = await sampling_client.sample_async(
            prompt=psc_mi, num_samples=n_psc, sampling_params=sp_cont
        )
        parallel_latency = time.perf_counter() - t0

        # Check PSC correctness
        psc_correct = 0
        for seq in psc_batch_result.sequences:
            text = tokenizer.decode(list(seq.tokens))
            c, _ = try_extract_and_grade(text, ground_truth, "sympy", 2.0)
            if c:
                psc_correct += 1

        baee_latency = prefix_latency + parallel_latency
        psc_rate = psc_correct / n_psc
        triggered = psc_rate >= theta

        speedup = full_latency / baee_latency if baee_latency > 0 else 0

        result = {
            "idx": problem["idx"],
            "full_latency": full_latency,
            "full_len": full_len,
            "full_correct": full_correct,
            "prefix_latency": prefix_latency,
            "parallel_latency": parallel_latency,
            "baee_latency": baee_latency,
            "psc_rate": psc_rate,
            "triggered": triggered,
            "speedup": speedup,
        }
        results.append(result)

        status = (f"full={full_latency:.1f}s ({full_len}tok) | "
                  f"baee={baee_latency:.1f}s (prefix={prefix_latency:.1f}s + parallel={parallel_latency:.1f}s) | "
                  f"speedup={speedup:.2f}x | psc={psc_rate:.0%} | trigger={'Y' if triggered else 'N'}")
        logger.info(f"[{pi+1}/{n_problems}] {status}")

    # Summary
    if results:
        triggered_results = [r for r in results if r['triggered']]
        all_full = [r['full_latency'] for r in results]
        all_baee = [r['baee_latency'] for r in results]
        trig_full = [r['full_latency'] for r in triggered_results]
        trig_baee = [r['baee_latency'] for r in triggered_results]
        trig_speedup = [r['speedup'] for r in triggered_results]

        print(f"\n{'='*70}")
        print(f"Latency Benchmark: {model_name} ({len(results)} problems)")
        print(f"{'='*70}")
        print(f"All problems:")
        print(f"  Full CoT:  mean={np.mean(all_full):.1f}s, median={np.median(all_full):.1f}s")
        print(f"  BAEE:      mean={np.mean(all_baee):.1f}s, median={np.median(all_baee):.1f}s")
        print(f"Triggered ({len(triggered_results)}/{len(results)} = {len(triggered_results)/len(results):.0%}):")
        if triggered_results:
            print(f"  Full CoT:  mean={np.mean(trig_full):.1f}s, median={np.median(trig_full):.1f}s")
            print(f"  BAEE:      mean={np.mean(trig_baee):.1f}s, median={np.median(trig_baee):.1f}s")
            print(f"  Speedup:   mean={np.mean(trig_speedup):.2f}x, median={np.median(trig_speedup):.2f}x")
            print(f"  Latency reduction: {1 - np.mean(trig_baee)/np.mean(trig_full):.0%}")
        print(f"{'='*70}")

    # Save
    out_dir = "/tmp/tinker-examples/reasoning_theater/latency_benchmark"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"latency_{model_name.replace('/', '_')}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--renderer", default="qwen3")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--theta", type=float, default=0.75)
    args = p.parse_args()
    asyncio.run(benchmark_latency(args.model, args.renderer, args.n, theta=args.theta))
