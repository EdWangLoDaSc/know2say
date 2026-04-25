"""
RAPO Signal Validation Script.

Rigorous validation of reachability probe signals:
1. Correlation: Does Φ(prefix) correlate with actual outcome?
2. Leakage: Does the prefix already contain the answer?
3. Stability: Do different trajectories' prefixes give consistent Φ?
4. Correctness: Is prompt+prefix construction correct?
5. Monotonicity: Does Φ change sensibly along trajectories?
6. Calibration: Is Φ close to the actual success rate in the group?

Usage:
    python -m tinker_cookbook.recipes.math_rl.rapo_validate
"""

import json
import logging
import math
import os
import time
from collections import defaultdict

import chz
import datasets
import numpy as np
import tinker
import tinker.types as types

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    base_url: str | None = None
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    batch_size: int = 8  # small for detailed analysis
    group_size: int = 16
    max_tokens: int = 256
    n_probes: int = 16
    output_dir: str = "/tmp/tinker-examples/rapo_validate"


def main(config: Config):
    import asyncio
    asyncio.run(_async_main(config))


async def _async_main(config: Config):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(config.output_dir, exist_ok=True)

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    dataset = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"]

    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."
    convo_prefix = [
        {"role": "user", "content": "How many r's are in strawberry?" + question_suffix},
        {"role": "assistant", "content": "Let's spell the word out and number all the letters: "
         "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
         "We have r's at positions 3, 8, and 9. \\boxed{3}"},
    ]

    service_client = tinker.ServiceClient(base_url=config.base_url)
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=32
    )
    sampling_client = training_client.save_weights_and_get_sampling_client()
    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )

    batch_rows = train_dataset.select(range(config.batch_size))
    all_validations = []

    # ─── Sample all groups ───
    print("\n" + "=" * 80)
    print("PHASE 1: Sampling trajectories")
    print("=" * 80)

    futures = []
    prompts = []
    for question in batch_rows["question"]:
        convo = [*convo_prefix, {"role": "user", "content": question + question_suffix}]
        model_input = renderer.build_generation_prompt(convo)
        future = sampling_client.sample(
            prompt=model_input,
            num_samples=config.group_size,
            sampling_params=sampling_params,
        )
        futures.append(future)
        prompts.append(model_input)

    for prob_idx, (future, prompt, row) in enumerate(
        zip(futures, prompts, batch_rows)
    ):
        question = row["question"]
        answer = row["answer"]
        gt_answer = extract_gsm8k_final_answer(answer)
        sample_result = future.result()

        print(f"\n{'='*80}")
        print(f"PROBLEM {prob_idx}: {question[:100]}...")
        print(f"Ground truth answer: {gt_answer}")
        print(f"{'='*80}")

        # Collect trajectories
        token_seqs = []
        responses = []
        rewards = []
        for seq in sample_result.sequences:
            token_seqs.append(seq.tokens)
            parsed_msg, _ = renderer.parse_response(seq.tokens)
            content = renderers.get_text_content(parsed_msg)
            responses.append(content)
            try:
                given = extract_boxed(content)
                reward = 1.0 if grade_answer(given, gt_answer) else 0.0
            except ValueError:
                reward = 0.0
            rewards.append(reward)

        mean_reward = sum(rewards) / len(rewards)
        group_type = "all_correct" if all(r > 0 for r in rewards) else \
                     "all_incorrect" if all(r == 0 for r in rewards) else "mixed"

        print(f"\nGroup type: {group_type}, mean_reward: {mean_reward:.2f}")
        print(f"Rewards: {rewards}")

        # ─── CHECK 1: Prefix content inspection ───
        print(f"\n--- CHECK 1: Prefix Content Inspection ---")
        # Look at what the prefix actually contains at different points
        traj0_tokens = token_seqs[0]
        traj0_text = tokenizer.decode(traj0_tokens)
        seq_len = len(traj0_tokens)
        checkpoints = [0.10, 0.25, 0.50, 0.75, 0.90]

        for frac in checkpoints:
            depth = max(1, int(frac * seq_len))
            prefix_text = tokenizer.decode(traj0_tokens[:depth])
            # Check if answer is already in prefix
            has_boxed = "\\boxed" in prefix_text or "boxed{" in prefix_text
            has_answer = gt_answer in prefix_text
            print(f"  {frac:.0%} ({depth}/{seq_len} tokens):")
            print(f"    Contains \\boxed: {has_boxed}")
            print(f"    Contains answer '{gt_answer}': {has_answer}")
            # Show last 80 chars of prefix
            snippet = prefix_text[-80:].replace('\n', '\\n')
            print(f"    Tail: ...{snippet}")

        # ─── CHECK 2: Probe reachability at checkpoints ───
        print(f"\n--- CHECK 2: Reachability Probes (traj 0) ---")
        phi_results = []
        for frac in checkpoints:
            depth = max(1, int(frac * seq_len))
            prefix = traj0_tokens[:depth]

            # Build probe input
            probe_input = prompt.append(types.EncodedTextChunk(tokens=prefix))
            remaining = max(config.max_tokens - len(prefix), 64)
            probe_params = types.SamplingParams(
                max_tokens=remaining,
                stop=renderer.get_stop_sequences(),
                temperature=1.0,
            )

            probe_future = sampling_client.sample(
                prompt=probe_input,
                num_samples=config.n_probes,
                sampling_params=probe_params,
            )
            probe_result = probe_future.result()

            n_success = 0
            probe_answers = []
            for seq in probe_result.sequences:
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = renderers.get_text_content(parsed_msg)
                try:
                    given = extract_boxed(content)
                    correct = grade_answer(given, gt_answer)
                    probe_answers.append(f"{'✓' if correct else '✗'} {given}")
                    if correct:
                        n_success += 1
                except ValueError:
                    probe_answers.append("✗ (no boxed)")

            phi = n_success / config.n_probes
            psi = float(np.log((phi + 0.01) / (1 - phi + 0.01)))

            # Check if prefix already has the answer
            prefix_text = tokenizer.decode(prefix)
            has_boxed = "\\boxed" in prefix_text or "boxed{" in prefix_text

            phi_results.append({
                "frac": frac, "depth": depth, "phi": phi, "psi": psi,
                "has_boxed_in_prefix": has_boxed,
            })

            flag = " ⚠️ ANSWER IN PREFIX" if has_boxed else ""
            print(f"  {frac:.0%} ({depth:3d} tok): Φ={phi:.2f}, Ψ={psi:+.2f}{flag}")
            print(f"    Probe answers: {probe_answers[:8]}")

        # ─── CHECK 3: Stability across trajectories ───
        print(f"\n--- CHECK 3: Cross-Trajectory Stability ---")
        # Probe at 50% for multiple trajectories
        mid_frac = 0.50
        traj_phis = []
        n_trajs_to_check = min(4, len(token_seqs))
        for ti in range(n_trajs_to_check):
            traj_tokens = token_seqs[ti]
            depth = max(1, int(mid_frac * len(traj_tokens)))
            prefix = traj_tokens[:depth]

            probe_input = prompt.append(types.EncodedTextChunk(tokens=prefix))
            remaining = max(config.max_tokens - len(prefix), 64)
            probe_params = types.SamplingParams(
                max_tokens=remaining,
                stop=renderer.get_stop_sequences(),
                temperature=1.0,
            )

            probe_future = sampling_client.sample(
                prompt=probe_input,
                num_samples=config.n_probes,
                sampling_params=probe_params,
            )
            probe_result = probe_future.result()

            n_success = 0
            for seq in probe_result.sequences:
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = renderers.get_text_content(parsed_msg)
                try:
                    given = extract_boxed(content)
                    if grade_answer(given, gt_answer):
                        n_success += 1
                except ValueError:
                    pass

            phi = n_success / config.n_probes
            traj_phis.append(phi)
            outcome = "correct" if rewards[ti] > 0 else "incorrect"
            print(f"  Traj {ti} ({outcome}): Φ={phi:.2f} (prefix {depth}/{len(traj_tokens)} tokens)")

        if len(traj_phis) >= 2:
            phi_std = np.std(traj_phis)
            print(f"  Φ std across trajectories: {phi_std:.3f}")
            if phi_std < 0.15:
                print(f"  → STABLE: probes from different trajectories agree")
            else:
                print(f"  → UNSTABLE: high variance across trajectories")

        # ─── CHECK 4: Φ vs actual outcome correlation ───
        print(f"\n--- CHECK 4: Φ-Outcome Correlation ---")
        # For mixed groups, check if trajectories that succeed have higher Φ at 50%
        if group_type == "mixed":
            correct_phis = [traj_phis[i] for i in range(n_trajs_to_check) if rewards[i] > 0]
            incorrect_phis = [traj_phis[i] for i in range(n_trajs_to_check) if rewards[i] == 0]
            if correct_phis and incorrect_phis:
                print(f"  Correct trajectories   Φ@50%: {np.mean(correct_phis):.2f} (n={len(correct_phis)})")
                print(f"  Incorrect trajectories Φ@50%: {np.mean(incorrect_phis):.2f} (n={len(incorrect_phis)})")
                if np.mean(correct_phis) > np.mean(incorrect_phis):
                    print(f"  → VALID: correct trajectories have higher reachability")
                else:
                    print(f"  → ⚠️ INVALID: incorrect trajectories have higher Φ!")
            else:
                print(f"  (Cannot compare - all checked trajectories have same outcome)")
        else:
            print(f"  (Skipped - group is {group_type})")

        # ─── CHECK 5: Calibration ───
        print(f"\n--- CHECK 5: Calibration (Φ@10% vs group success rate) ---")
        phi_at_10 = phi_results[0]["phi"] if phi_results else None
        if phi_at_10 is not None:
            print(f"  Φ at 10%:        {phi_at_10:.2f}")
            print(f"  Group mean reward: {mean_reward:.2f}")
            print(f"  Difference:        {abs(phi_at_10 - mean_reward):.2f}")
            if abs(phi_at_10 - mean_reward) < 0.25:
                print(f"  → CALIBRATED: Φ@10% ≈ group success rate")
            else:
                print(f"  → MISCALIBRATED: Φ@10% differs from group success rate by >{abs(phi_at_10 - mean_reward):.2f}")

        # ─── CHECK 6: Monotonicity within correct/incorrect trajectories ───
        print(f"\n--- CHECK 6: Φ Trajectory Shape ---")
        phis_along = [r["phi"] for r in phi_results]
        increases = sum(1 for i in range(1, len(phis_along)) if phis_along[i] > phis_along[i-1])
        decreases = sum(1 for i in range(1, len(phis_along)) if phis_along[i] < phis_along[i-1])
        flat = sum(1 for i in range(1, len(phis_along)) if phis_along[i] == phis_along[i-1])
        traj0_correct = rewards[0] > 0
        print(f"  Traj 0 outcome: {'correct' if traj0_correct else 'incorrect'}")
        print(f"  Φ curve: {' → '.join(f'{p:.2f}' for p in phis_along)}")
        print(f"  Trend: {increases} increases, {decreases} decreases, {flat} flat")
        if traj0_correct and increases >= decreases:
            print(f"  → CONSISTENT: correct trajectory has non-decreasing Φ")
        elif not traj0_correct and decreases >= increases:
            print(f"  → CONSISTENT: incorrect trajectory has non-increasing Φ")
        elif traj0_correct and increases < decreases:
            print(f"  → ⚠️ ANOMALY: correct trajectory has decreasing Φ")
        elif not traj0_correct and decreases < increases:
            print(f"  → ⚠️ ANOMALY: incorrect trajectory has increasing Φ")

        # ─── CHECK 7: Answer leakage check ───
        print(f"\n--- CHECK 7: Answer Leakage ---")
        for frac in [0.50, 0.75, 0.90]:
            depth = max(1, int(frac * seq_len))
            prefix_text = tokenizer.decode(traj0_tokens[:depth])
            has_boxed = "\\boxed" in prefix_text or "boxed{" in prefix_text
            if has_boxed:
                # Extract what's in the boxed
                try:
                    boxed_answer = extract_boxed(prefix_text)
                    correct_in_prefix = grade_answer(boxed_answer, gt_answer)
                    print(f"  {frac:.0%}: \\boxed{{{boxed_answer}}} in prefix → "
                          f"{'CORRECT' if correct_in_prefix else 'WRONG'} answer")
                    if correct_in_prefix and phi_results:
                        matching = [r for r in phi_results if r["frac"] == frac]
                        if matching and matching[0]["phi"] > 0.8:
                            print(f"    → ⚠️ HIGH Φ MAY BE DUE TO ANSWER LEAKAGE")
                except ValueError:
                    print(f"  {frac:.0%}: incomplete \\boxed in prefix")

        # Store validation results
        all_validations.append({
            "problem_idx": prob_idx,
            "question": question[:200],
            "gt_answer": gt_answer,
            "group_type": group_type,
            "mean_reward": mean_reward,
            "phi_results": phi_results,
            "cross_traj_phis": traj_phis,
            "cross_traj_std": float(np.std(traj_phis)) if len(traj_phis) >= 2 else None,
        })

    # ─── GLOBAL SUMMARY ───
    print(f"\n{'='*80}")
    print("GLOBAL VALIDATION SUMMARY")
    print(f"{'='*80}")

    # Calibration
    calibration_errors = []
    for v in all_validations:
        if v["phi_results"]:
            phi10 = v["phi_results"][0]["phi"]
            calibration_errors.append(abs(phi10 - v["mean_reward"]))
    if calibration_errors:
        print(f"\nCalibration (|Φ@10% - group_reward|):")
        print(f"  Mean error: {np.mean(calibration_errors):.3f}")
        print(f"  Max error:  {max(calibration_errors):.3f}")

    # Stability
    stabilities = [v["cross_traj_std"] for v in all_validations if v["cross_traj_std"] is not None]
    if stabilities:
        print(f"\nStability (Φ std across trajectories at 50%):")
        print(f"  Mean std: {np.mean(stabilities):.3f}")
        print(f"  Max std:  {max(stabilities):.3f}")

    # Signal strength
    all_delta_psi = []
    for v in all_validations:
        phis = v["phi_results"]
        for i in range(1, len(phis)):
            all_delta_psi.append(phis[i]["psi"] - phis[i-1]["psi"])
    if all_delta_psi:
        print(f"\nΔΨ (step advantage signal):")
        print(f"  Mean:    {np.mean(all_delta_psi):+.3f}")
        print(f"  Std:     {np.std(all_delta_psi):.3f}")
        print(f"  |ΔΨ|>0.1: {sum(1 for d in all_delta_psi if abs(d)>0.1)}/{len(all_delta_psi)} "
              f"({100*sum(1 for d in all_delta_psi if abs(d)>0.1)/len(all_delta_psi):.0f}%)")
        print(f"  |ΔΨ|>0.5: {sum(1 for d in all_delta_psi if abs(d)>0.5)}/{len(all_delta_psi)} "
              f"({100*sum(1 for d in all_delta_psi if abs(d)>0.5)/len(all_delta_psi):.0f}%)")

    # Leakage assessment
    leakage_count = 0
    total_high_phi = 0
    for v in all_validations:
        for r in v["phi_results"]:
            if r["phi"] > 0.8:
                total_high_phi += 1
                if r["has_boxed_in_prefix"]:
                    leakage_count += 1
    if total_high_phi > 0:
        print(f"\nLeakage Assessment:")
        print(f"  High Φ (>0.8) checkpoints: {total_high_phi}")
        print(f"  Of those, answer in prefix: {leakage_count} ({100*leakage_count/total_high_phi:.0f}%)")
        if leakage_count / total_high_phi > 0.5:
            print(f"  → ⚠️ HIGH LEAKAGE: most high-Φ points have answer in prefix")
        else:
            print(f"  → LOW LEAKAGE: high Φ is mostly from reasoning quality, not answer presence")

    print(f"\n{'='*80}")
    print("VERDICT:")
    issues = []
    if calibration_errors and np.mean(calibration_errors) > 0.3:
        issues.append("Poor calibration")
    if stabilities and np.mean(stabilities) > 0.2:
        issues.append("Unstable across trajectories")
    if total_high_phi > 0 and leakage_count / total_high_phi > 0.5:
        issues.append("Answer leakage in prefix")
    if not all_delta_psi or sum(1 for d in all_delta_psi if abs(d) > 0.1) / len(all_delta_psi) < 0.2:
        issues.append("Weak step-level signal")

    if not issues:
        print("  ✓ ALL CHECKS PASSED - Reachability signal is valid")
    else:
        print(f"  ⚠️ ISSUES FOUND: {', '.join(issues)}")
    print(f"{'='*80}\n")

    # Save
    output_path = os.path.join(config.output_dir, "validation_results.json")
    with open(output_path, "w") as f:
        json.dump(all_validations, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
