"""EFA suffix ablation on GPQA-Diamond with Qwen3-32B-Think."""
import asyncio
import json
import logging
import os
import sys

import paths

paths.setup_path()

import tinker
import tinker.types as types
from tinker_cookbook import renderers
from experiment import (
    ExperimentConfig, load_problems, run_psc, safe_grade, _is_mc_benchmark,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

EFA_SUFFIXES = {
    "original":  "\nTherefore, the final answer is \\boxed{",
    "natural":   "\nThe answer is \\boxed{",
    "soft":      "\nSo the answer is \\boxed{",
    "plain":     "\nAnswer: ",
    "direct":    "\n\\boxed{",
}
SUFFIX_STOP = {
    "original": ["}"], "natural": ["}"], "soft": ["}"],
    "plain": ["\n"], "direct": ["}"],
}


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_name = "Qwen/Qwen3-32B"
    renderer_name = "qwen3"
    benchmark = "gpqa-diamond"
    n_problems = 100
    output_dir = paths.results("suffix_ablation_gpqa")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    config = ExperimentConfig(
        model_name=model_name, renderer_name=renderer_name,
        benchmark=benchmark, n_problems=198, max_tokens=8192,
    )

    problems = load_problems(n_problems, benchmark, seed=42)
    logger.info(f"Loaded {len(problems)} problems")

    output_path = os.path.join(output_dir, "results.jsonl")

    # Resume support
    done_idxs = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_idxs.add(r['problem_idx'])
        logger.info(f"Resuming: {len(done_idxs)} done")

    with open(output_path, "a") as f:
        for pi, problem in enumerate(problems):
            idx = problem['idx']
            if idx in done_idxs:
                continue

            question = problem["problem"]
            ground_truth = problem["answer"]
            convo = [{"role": "user", "content": question}]
            prompt_mi = renderer.build_generation_prompt(convo)

            # Generate one rollout
            sp = types.SamplingParams(
                max_tokens=config.max_tokens,
                stop=renderer.get_stop_sequences(),
                temperature=1.0,
            )
            sample_result = await sampling_client.sample_async(
                prompt=prompt_mi, num_samples=1, sampling_params=sp
            )
            cot_tokens = list(sample_result.sequences[0].tokens)
            if len(cot_tokens) < 20:
                logger.warning(f"[{pi+1}] idx={idx} too short, skip")
                continue

            result = {"problem_idx": idx, "cot_len": len(cot_tokens)}

            # PSC at each fraction
            for frac in [0.10, 0.30, 0.50]:
                prefix_len = max(1, int(frac * len(cot_tokens)))
                prefix = cot_tokens[:prefix_len]

                # PSC
                n_corr, n_total = await run_psc(
                    sampling_client, prompt_mi, prefix, ground_truth,
                    renderer, tokenizer, config,
                    total_cot_len=len(cot_tokens), grader="exact",
                )
                result[f"psc_{frac:.2f}"] = n_corr / n_total if n_total > 0 else 0

                # EFA with each suffix
                for sname, suffix in EFA_SUFFIXES.items():
                    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
                    combined = list(prefix) + list(suffix_tokens)
                    forced_mi = prompt_mi.append(types.EncodedTextChunk(tokens=combined))
                    efa_sp = types.SamplingParams(
                        max_tokens=64, stop=SUFFIX_STOP[sname], temperature=0.0,
                    )
                    try:
                        efa_result = await sampling_client.sample_async(
                            prompt=forced_mi, num_samples=1, sampling_params=efa_sp
                        )
                        efa_text = efa_result.sequences[0].text.strip().rstrip('}.')
                        correct = safe_grade(efa_text, ground_truth, grader="exact")
                        result[f"efa_{sname}_{frac:.2f}"] = {
                            "text": efa_text[:100], "correct": correct,
                        }
                    except Exception as e:
                        result[f"efa_{sname}_{frac:.2f}"] = {"error": str(e), "correct": False}

            f.write(json.dumps(result) + "\n")
            f.flush()

            psc_10 = result.get("psc_0.10", 0)
            efa_orig = result.get("efa_original_0.10", {}).get("correct", False)
            logger.info(f"[{pi+1}/{len(problems)}] idx={idx} psc@10%={psc_10:.0%} efa_orig@10%={efa_orig}")

    # Summary
    results = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"\n{'='*60}")
    print(f"GPQA Suffix Ablation Summary (n={len(results)})")
    print(f"{'='*60}")
    for frac in [0.10, 0.30, 0.50]:
        psc_vals = [r.get(f"psc_{frac:.2f}", 0) for r in results]
        print(f"\nf={frac:.2f}: PSC mean = {sum(psc_vals)/len(psc_vals):.1%}")
        for sname in EFA_SUFFIXES:
            efa_vals = [r.get(f"efa_{sname}_{frac:.2f}", {}).get("correct", False) for r in results]
            print(f"  {sname:12s}: EFA acc = {sum(efa_vals)/len(efa_vals):.1%}")


if __name__ == "__main__":
    asyncio.run(main())
