"""
RAPO (Reachability-Aware Policy Optimization) Analysis Script.

Runs a few batches of GRPO-style sampling, then analyzes:
1. Prefix tree structure (divergence points)
2. Entropy at divergence points
3. All-incorrect group statistics
4. Mock reachability probe analysis (using additional rollouts from same prefix)

Usage:
    python -m tinker_cookbook.recipes.math_rl.rapo_analysis [n_batches=2] [batch_size=32] [group_size=16]
"""

import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import chz
import datasets
import numpy as np
import tinker
import tinker.types as types
from tqdm import tqdm

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Prefix Tree
# ─────────────────────────────────────────────

@dataclass
class PrefixTreeNode:
    """A node in the prefix tree built from a group of trajectories."""
    token: int | None = None  # token at this node (None for root)
    children: dict[int, "PrefixTreeNode"] = field(default_factory=dict)
    trajectory_ids: list[int] = field(default_factory=list)  # which trajectories pass through
    rewards: list[float] = field(default_factory=list)  # rewards of passing trajectories
    depth: int = 0
    logprobs: list[float] = field(default_factory=list)  # logprobs at this node


def build_prefix_tree(
    token_sequences: list[list[int]],
    rewards: list[float],
    logprobs_sequences: list[list[float]],
) -> PrefixTreeNode:
    """Build a prefix tree from a group of token sequences."""
    root = PrefixTreeNode(
        trajectory_ids=list(range(len(token_sequences))),
        rewards=list(rewards),
    )

    for traj_id, (tokens, reward, logprobs) in enumerate(
        zip(token_sequences, rewards, logprobs_sequences)
    ):
        node = root
        for depth, (tok, lp) in enumerate(zip(tokens, logprobs)):
            if tok not in node.children:
                node.children[tok] = PrefixTreeNode(token=tok, depth=depth + 1)
            child = node.children[tok]
            child.trajectory_ids.append(traj_id)
            child.rewards.append(reward)
            child.logprobs.append(lp)
            node = child

    return root


@dataclass
class DivergencePoint:
    """A point where trajectories diverge in the prefix tree."""
    depth: int
    n_branches: int
    n_trajectories: int  # total trajectories at parent
    branch_sizes: list[int]  # size of each branch
    branch_success_rates: list[float]  # success rate per branch
    parent_success_rate: float
    max_success_rate_delta: float  # max |branch_sr - parent_sr|
    entropy_estimate: float  # entropy based on branch distribution


def find_divergence_points(root: PrefixTreeNode) -> list[DivergencePoint]:
    """Find all branching points in the prefix tree."""
    points = []

    def _traverse(node: PrefixTreeNode):
        if len(node.children) > 1:
            n_traj = len(node.trajectory_ids)
            parent_sr = sum(node.rewards) / max(len(node.rewards), 1)
            branch_sizes = []
            branch_srs = []
            for child in node.children.values():
                branch_sizes.append(len(child.trajectory_ids))
                child_sr = sum(child.rewards) / max(len(child.rewards), 1)
                branch_srs.append(child_sr)

            # Entropy of branch distribution
            probs = np.array(branch_sizes, dtype=float) / sum(branch_sizes)
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            max_delta = max(abs(sr - parent_sr) for sr in branch_srs) if branch_srs else 0.0

            points.append(DivergencePoint(
                depth=node.depth,
                n_branches=len(node.children),
                n_trajectories=n_traj,
                branch_sizes=branch_sizes,
                branch_success_rates=branch_srs,
                parent_success_rate=parent_sr,
                max_success_rate_delta=max_delta,
                entropy_estimate=entropy,
            ))

        for child in node.children.values():
            _traverse(child)

    _traverse(root)
    return points


def compute_consensus_depth(
    token_sequences: list[list[int]],
) -> list[int]:
    """For each trajectory, compute how deep it stays on the plurality branch."""
    if not token_sequences:
        return []

    max_len = max(len(s) for s in token_sequences)
    consensus_depths = [0] * len(token_sequences)

    for depth in range(max_len):
        # Count token frequencies at this depth
        token_counts: dict[int, list[int]] = defaultdict(list)
        for traj_id, seq in enumerate(token_sequences):
            if depth < len(seq):
                token_counts[seq[depth]].append(traj_id)

        if not token_counts:
            break

        # Find plurality token
        plurality_token = max(token_counts, key=lambda t: len(token_counts[t]))
        plurality_trajs = set(token_counts[plurality_token])

        for traj_id in range(len(token_sequences)):
            if traj_id in plurality_trajs:
                consensus_depths[traj_id] = depth + 1

    return consensus_depths


# ─────────────────────────────────────────────
# 2. Reachability Estimation (mock via probe rollouts)
# ─────────────────────────────────────────────

def probe_reachability_sync(
    sampling_client: tinker.SamplingClient,
    prompt: types.ModelInput,
    prefix_tokens: list[int],
    answer: str,
    renderer: renderers.Renderer,
    n_probes: int = 8,
    max_tokens: int = 256,
) -> float:
    """Estimate reachability Φ(s) = P(success | prefix) via probe rollouts."""
    # Build prompt + prefix
    probe_input = prompt.append(types.EncodedTextChunk(tokens=prefix_tokens))

    remaining_tokens = max(max_tokens - len(prefix_tokens), 64)
    sampling_params = types.SamplingParams(
        max_tokens=remaining_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=1.0,
    )

    future = sampling_client.sample(
        prompt=probe_input,
        num_samples=n_probes,
        sampling_params=sampling_params,
    )
    result = future.result()

    n_success = 0
    for seq in result.sequences:
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        content = renderers.get_text_content(parsed_msg)
        try:
            given = extract_boxed(content)
            gt = extract_gsm8k_final_answer(answer)
            if grade_answer(given, gt):
                n_success += 1
        except ValueError:
            pass

    return n_success / n_probes


# ─────────────────────────────────────────────
# 3. Analysis Functions
# ─────────────────────────────────────────────

@dataclass
class GroupAnalysis:
    """Analysis results for a single group."""
    problem_idx: int
    group_type: str  # "all_correct", "all_incorrect", "mixed"
    n_trajectories: int
    mean_reward: float
    rewards: list[float]
    n_divergence_points: int
    divergence_depths: list[int]
    max_success_rate_delta: float  # best divergence point's delta
    consensus_depths: list[int]
    mean_consensus_depth: float
    mean_seq_length: float
    # Reachability (filled if probed)
    reachability_at_checkpoints: list[dict] | None = None


def analyze_group(
    problem_idx: int,
    token_sequences: list[list[int]],
    rewards: list[float],
    logprobs_sequences: list[list[float]],
) -> GroupAnalysis:
    """Analyze a single group's prefix tree and divergence structure."""

    # Classify group
    if all(r > 0 for r in rewards):
        group_type = "all_correct"
    elif all(r == 0 for r in rewards):
        group_type = "all_incorrect"
    else:
        group_type = "mixed"

    # Build prefix tree
    tree = build_prefix_tree(token_sequences, rewards, logprobs_sequences)
    div_points = find_divergence_points(tree)

    # Consensus depths
    c_depths = compute_consensus_depth(token_sequences)

    # Best divergence point (highest success rate delta)
    max_delta = max((dp.max_success_rate_delta for dp in div_points), default=0.0)

    return GroupAnalysis(
        problem_idx=problem_idx,
        group_type=group_type,
        n_trajectories=len(token_sequences),
        mean_reward=sum(rewards) / len(rewards),
        rewards=rewards,
        n_divergence_points=len(div_points),
        divergence_depths=[dp.depth for dp in div_points],
        max_success_rate_delta=max_delta,
        consensus_depths=c_depths,
        mean_consensus_depth=sum(c_depths) / len(c_depths) if c_depths else 0,
        mean_seq_length=sum(len(s) for s in token_sequences) / len(token_sequences),
    )


# ─────────────────────────────────────────────
# 4. Main Script
# ─────────────────────────────────────────────

@chz.chz
class AnalysisConfig:
    base_url: str | None = None
    model_name: str = "meta-llama/Llama-3.1-8B"
    batch_size: int = 32
    group_size: int = 16
    max_tokens: int = 256
    n_batches: int = 2
    n_probes: int = 8  # probe rollouts for reachability
    n_probe_checkpoints: int = 4  # how many checkpoints to probe per trajectory
    probe_all_incorrect_only: bool = True  # only probe all-incorrect groups
    output_dir: str = "/tmp/tinker-examples/rapo_analysis"


def main(config: AnalysisConfig):
    import asyncio
    asyncio.run(_async_main(config))


async def _async_main(config: AnalysisConfig):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(config.output_dir, exist_ok=True)

    # Setup
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    dataset = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"]

    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."
    convo_prefix = [
        {"role": "user", "content": "How many r's are in strawberry?" + question_suffix},
        {"role": "assistant", "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}"},
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

    all_analyses: list[dict] = []
    summary_stats = {
        "all_correct": 0, "all_incorrect": 0, "mixed": 0,
        "total_groups": 0,
        "total_div_points": 0,
        "div_points_per_group": [],
        "consensus_depth_ratio_all_incorrect": [],  # consensus_depth / seq_length
        "max_delta_mixed": [],
        "max_delta_all_incorrect": [],
    }

    for batch_idx in range(config.n_batches):
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        logger.info(f"=== Batch {batch_idx} ({batch_end - batch_start} problems) ===")

        # Sample all groups
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

        # Collect and analyze
        for prob_idx, (future, prompt, answer) in enumerate(tqdm(
            zip(futures, prompts, batch_rows["answer"]),
            total=len(futures),
            desc=f"Batch {batch_idx}",
        )):
            sample_result = future.result()

            token_seqs = []
            logprob_seqs = []
            rewards = []
            responses = []

            for seq in sample_result.sequences:
                token_seqs.append(seq.tokens)
                logprob_seqs.append(seq.logprobs)
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = renderers.get_text_content(parsed_msg)
                responses.append(content)
                try:
                    given = extract_boxed(content)
                    gt = extract_gsm8k_final_answer(answer)
                    reward = 1.0 if grade_answer(given, gt) else 0.0
                except ValueError:
                    reward = 0.0
                rewards.append(reward)

            # Analyze group
            analysis = analyze_group(
                problem_idx=batch_start + prob_idx,
                token_sequences=token_seqs,
                rewards=rewards,
                logprobs_sequences=logprob_seqs,
            )

            # Update summary
            summary_stats[analysis.group_type] += 1
            summary_stats["total_groups"] += 1
            summary_stats["total_div_points"] += analysis.n_divergence_points
            summary_stats["div_points_per_group"].append(analysis.n_divergence_points)

            if analysis.group_type == "all_incorrect":
                ratio = analysis.mean_consensus_depth / max(analysis.mean_seq_length, 1)
                summary_stats["consensus_depth_ratio_all_incorrect"].append(ratio)
                summary_stats["max_delta_all_incorrect"].append(analysis.max_success_rate_delta)
            elif analysis.group_type == "mixed":
                summary_stats["max_delta_mixed"].append(analysis.max_success_rate_delta)

            # Probe reachability for all-incorrect groups (or all if configured)
            should_probe = (
                not config.probe_all_incorrect_only or analysis.group_type == "all_incorrect"
            )
            if should_probe and len(token_seqs) > 0:
                # Probe at evenly-spaced positions along the sequence (e.g. 10%, 25%, 50%, 75%)
                seq_len = len(token_seqs[0])
                if seq_len < 8:
                    selected = []
                else:
                    fracs = [0.10, 0.25, 0.50, 0.75]
                    selected = [max(1, int(f * seq_len)) for f in fracs]
                    selected = sorted(set(selected))  # dedup

                # Use first trajectory's prefix for probing
                reachability_results = []
                for ckpt_depth in selected:
                    prefix = token_seqs[0][:ckpt_depth]
                    if len(prefix) == 0:
                        continue
                    try:
                        phi = probe_reachability_sync(
                            sampling_client=sampling_client,
                            prompt=prompt,
                            prefix_tokens=prefix,
                            answer=answer,
                            renderer=renderer,
                            n_probes=config.n_probes,
                            max_tokens=config.max_tokens,
                        )
                        reachability_results.append({
                            "depth": ckpt_depth,
                            "depth_frac": ckpt_depth / len(token_seqs[0]),
                            "phi": phi,
                            "psi": float(np.log((phi + 0.01) / (1 - phi + 0.01))),
                        })
                    except Exception as e:
                        logger.warning(f"Probe failed at depth {ckpt_depth}: {e}")

                analysis.reachability_at_checkpoints = reachability_results

            # Store
            all_analyses.append({
                "problem_idx": analysis.problem_idx,
                "group_type": analysis.group_type,
                "n_trajectories": analysis.n_trajectories,
                "mean_reward": analysis.mean_reward,
                "n_divergence_points": analysis.n_divergence_points,
                "mean_consensus_depth": analysis.mean_consensus_depth,
                "mean_seq_length": analysis.mean_seq_length,
                "consensus_depth_ratio": analysis.mean_consensus_depth / max(analysis.mean_seq_length, 1),
                "max_success_rate_delta": analysis.max_success_rate_delta,
                "reachability": analysis.reachability_at_checkpoints,
                # Sample one response for inspection
                "sample_response": responses[0][:500] if responses else "",
            })

    # ─────────────────────────────────────────────
    # Print Summary
    # ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RAPO ANALYSIS SUMMARY")
    print("=" * 70)

    total = summary_stats["total_groups"]
    print(f"\nGroup Distribution ({total} total):")
    for gt in ["all_correct", "mixed", "all_incorrect"]:
        n = summary_stats[gt]
        print(f"  {gt:20s}: {n:4d} ({100*n/max(total,1):.1f}%)")

    print(f"\nDivergence Points:")
    dps = summary_stats["div_points_per_group"]
    print(f"  Mean per group: {np.mean(dps):.1f}")
    print(f"  Median:         {np.median(dps):.0f}")
    print(f"  Max:            {max(dps) if dps else 0}")

    print(f"\nAll-Incorrect Groups:")
    ratios = summary_stats["consensus_depth_ratio_all_incorrect"]
    if ratios:
        print(f"  Mean consensus_depth / seq_length: {np.mean(ratios):.3f}")
        print(f"  Std:                               {np.std(ratios):.3f}")
        print(f"  → Interpretation: {np.mean(ratios)*100:.1f}% of tokens are on consensus path")
    else:
        print("  No all-incorrect groups found")

    print(f"\nMixed Groups - Success Rate Delta at Divergence:")
    deltas = summary_stats["max_delta_mixed"]
    if deltas:
        print(f"  Mean max delta: {np.mean(deltas):.3f}")
        print(f"  → High delta = divergence points are highly predictive of outcome")

    # Reachability analysis
    probed = [a for a in all_analyses if a["reachability"]]
    if probed:
        print(f"\nReachability Probe Results ({len(probed)} groups probed):")
        all_phis = []
        all_psi_deltas = []
        for a in probed:
            reach = a["reachability"]
            phis = [r["phi"] for r in reach]
            all_phis.extend(phis)
            if len(reach) >= 2:
                for j in range(1, len(reach)):
                    all_psi_deltas.append(reach[j]["psi"] - reach[j-1]["psi"])

            print(f"\n  Problem {a['problem_idx']} ({a['group_type']}):")
            print(f"    Mean reward: {a['mean_reward']:.2f}")
            for r in reach:
                print(f"    Depth {r['depth']:3d} ({r['depth_frac']:.0%}): "
                      f"Φ={r['phi']:.2f}, Ψ={r['psi']:.2f}")

        if all_phis:
            print(f"\n  Overall Φ statistics:")
            print(f"    Mean: {np.mean(all_phis):.3f}")
            print(f"    Std:  {np.std(all_phis):.3f}")
            print(f"    Range: [{min(all_phis):.2f}, {max(all_phis):.2f}]")

        if all_psi_deltas:
            print(f"\n  Ψ increment (step advantage proxy) statistics:")
            print(f"    Mean: {np.mean(all_psi_deltas):.3f}")
            print(f"    Std:  {np.std(all_psi_deltas):.3f}")
            nonzero = [d for d in all_psi_deltas if abs(d) > 0.1]
            print(f"    Non-trivial increments (|ΔΨ|>0.1): {len(nonzero)}/{len(all_psi_deltas)}")
            print(f"    → Shows whether reachability changes across steps (signal for credit assignment)")

    print(f"\n{'='*70}")
    print("KEY TAKEAWAYS FOR RAPO:")
    print(f"  1. All-incorrect groups: {summary_stats['all_incorrect']}/{total} "
          f"({100*summary_stats['all_incorrect']/max(total,1):.0f}%) → GRPO wastes these")
    print(f"  2. Avg divergence points per group: {np.mean(dps):.0f} → enough structure for step-level credit")
    if ratios:
        print(f"  3. Consensus covers {np.mean(ratios)*100:.0f}% of tokens → early reasoning is shared")
    if probed:
        print(f"  4. Reachability varies across steps → step-level credit is informative")
    print(f"{'='*70}\n")

    # Save detailed results
    output_path = os.path.join(config.output_dir, "analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(all_analyses, f, indent=2, default=str)
    print(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
