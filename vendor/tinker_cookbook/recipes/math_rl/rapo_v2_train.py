"""
RAPO v2: Branch-Level Reachability Contrast

Key difference from v1: probe at prefix tree BRANCHING POINTS (not fixed checkpoints).
Advantages telescope correctly and provide signal for all-incorrect groups.

Usage:
    python -m tinker_cookbook.recipes.math_rl.rapo_v2_train dataset_name=math
"""

import logging
import math
import time
from collections import defaultdict
from concurrent.futures import Future

import chz
import datasets
import numpy as np
import tinker
import tinker.types as types
import torch
from tinker.types.tensor_data import TensorData
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import (
    MathEnv, extract_gsm8k_final_answer,
    _get_hendrycks_math_train,
)
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/rapo_v2"
    model_name: str = "meta-llama/Llama-3.1-8B"
    dataset_name: str = "math"
    batch_size: int = 128
    group_size: int = 16
    learning_rate: float = 4e-5
    lora_rank: int = 32
    save_every: int = 20
    max_tokens: int = 512
    ttl_seconds: int | None = 604800

    # RAPO v2
    n_probes: int = 8
    max_branch_probes: int = 3  # max branches to probe per branching point
    max_branching_points: int = 5  # max branching points to probe per group
    beta: float = 0.3  # scale for all-incorrect reachability advantage

    wandb_project: str | None = None
    wandb_name: str | None = None


# ─── Reward functions ───

def get_reward_gsm8k(response: str, answer: str) -> float:
    try:
        return 1.0 if grade_answer(extract_boxed(response), extract_gsm8k_final_answer(answer)) else 0.0
    except ValueError:
        return 0.0

def get_reward_math(response: str, answer: str) -> float:
    try:
        return 1.0 if grade_answer(extract_boxed(response), answer) else 0.0
    except ValueError:
        return 0.0

def phi_to_psi(phi: float, eps: float = 0.01) -> float:
    return float(np.log((phi + eps) / (1 - phi + eps)))


# ─── Prefix Tree ───

class PrefixTreeNode:
    __slots__ = ['children', 'traj_ids', 'depth']
    def __init__(self, depth: int = 0):
        self.children: dict[int, PrefixTreeNode] = {}
        self.traj_ids: list[int] = []
        self.depth = depth


def build_prefix_tree(token_seqs: list[list[int]]) -> PrefixTreeNode:
    root = PrefixTreeNode()
    root.traj_ids = list(range(len(token_seqs)))
    for tid, tokens in enumerate(token_seqs):
        node = root
        for d, tok in enumerate(tokens):
            if tok not in node.children:
                node.children[tok] = PrefixTreeNode(depth=d + 1)
            child = node.children[tok]
            child.traj_ids.append(tid)
            node = child
    return root


def find_branching_points(root: PrefixTreeNode, max_points: int) -> list[PrefixTreeNode]:
    """Find branching points sorted by number of trajectories (most populated first)."""
    points = []
    def _traverse(node: PrefixTreeNode):
        if len(node.children) > 1:
            points.append(node)
        for child in node.children.values():
            _traverse(child)
    _traverse(root)
    # Sort by population (more trajectories = more informative)
    points.sort(key=lambda n: len(n.traj_ids), reverse=True)
    return points[:max_points]


# ─── Branch Probing ───

def submit_branch_probes(
    sampling_client: tinker.SamplingClient,
    prompt: types.ModelInput,
    token_seqs: list[list[int]],
    branching_node: PrefixTreeNode,
    n_probes: int,
    max_branches: int,
    max_tokens: int,
    stop: list | None,
) -> list[tuple[list[int], Future]]:
    """Submit probes for each branch at a branching point. Returns (traj_ids, future) pairs."""
    branches = sorted(branching_node.children.values(), key=lambda c: len(c.traj_ids), reverse=True)
    branches = branches[:max_branches]

    probe_futures: list[tuple[list[int], Future]] = []
    for branch in branches:
        if not branch.traj_ids:
            continue
        # Use first trajectory in this branch as the probe prefix
        rep_traj = branch.traj_ids[0]
        prefix = token_seqs[rep_traj][:branch.depth]
        if len(prefix) < 1:
            continue

        probe_input = prompt.append(types.EncodedTextChunk(tokens=prefix))
        remaining = max(max_tokens - len(prefix), 64)
        params = types.SamplingParams(max_tokens=remaining, stop=stop, temperature=1.0)
        future = sampling_client.sample(prompt=probe_input, num_samples=n_probes, sampling_params=params)
        probe_futures.append((branch.traj_ids, future))

    return probe_futures


def compute_branch_advantages(
    token_seqs: list[list[int]],
    rewards: list[float],
    branching_points: list[PrefixTreeNode],
    branch_probe_results: list[list[tuple[list[int], float]]],
    config: Config,
) -> list[list[float]]:
    """
    Compute per-token advantages using branch-level reachability contrast.
    Ensures telescoping: sum of token advantages = r_i - mean(r).
    """
    G = len(token_seqs)
    mean_reward = sum(rewards) / G
    traj_advantages = [r - mean_reward for r in rewards]
    all_same = all(r == rewards[0] for r in rewards)

    # Initialize per-token advantages to 0
    token_advs: list[list[float]] = [[0.0] * len(seq) for seq in token_seqs]

    # Track cumulative branch advantage per trajectory (for terminal correction)
    cumulative_branch_adv = [0.0] * G

    for bp_node, branch_results in zip(branching_points, branch_probe_results):
        if not branch_results:
            continue

        # Compute weighted mean Ψ across branches
        total_trajs = sum(len(tids) for tids, _ in branch_results)
        if total_trajs == 0:
            continue

        weighted_psi = sum(len(tids) * psi for tids, psi in branch_results) / total_trajs

        # Assign advantage to each trajectory at the branching depth
        for tids, branch_psi in branch_results:
            if all_same:
                adv = config.beta * (branch_psi - weighted_psi)
            else:
                adv = branch_psi - weighted_psi

            for tid in tids:
                depth = bp_node.depth
                if depth < len(token_advs[tid]):
                    token_advs[tid][depth] += adv
                    cumulative_branch_adv[tid] += adv

    # Terminal correction: ensure telescoping
    # sum(A_{i,t}) should equal traj_advantage[i]
    for tid in range(G):
        if len(token_advs[tid]) == 0:
            continue
        residual = traj_advantages[tid] - cumulative_branch_adv[tid]
        # Distribute residual uniformly across all tokens
        n_tokens = len(token_advs[tid])
        per_token_residual = residual / n_tokens
        for t in range(n_tokens):
            token_advs[tid][t] += per_token_residual

    return token_advs


# ─── Main ───

def main(config: Config):
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    question_suffix = MathEnv.question_suffix()
    convo_prefix = MathEnv.standard_fewshot_prefix()

    if config.dataset_name == "gsm8k":
        ds = datasets.load_dataset("openai/gsm8k", "main")
        assert isinstance(ds, datasets.DatasetDict)
        train_dataset = ds["train"]
        q_key, a_key = "question", "answer"
        get_reward = get_reward_gsm8k
        question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."
    elif config.dataset_name == "math":
        train_dataset = _get_hendrycks_math_train().shuffle(seed=0)
        q_key, a_key = "problem", "solution"
        get_reward = get_reward_math
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")

    n_train_batches = len(train_dataset) // config.batch_size
    stop = renderer.get_stop_sequences()

    service_client = tinker.ServiceClient(base_url=config.base_url)
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(resume_info["state_path"])
        start_batch = resume_info["batch"]
    else:
        training_client = service_client.create_lora_training_client(base_model=config.model_name, rank=config.lora_rank)
        start_batch = 0

    sampling_params = types.SamplingParams(max_tokens=config.max_tokens, stop=stop)
    adam_params = types.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    logger.info(f"RAPO v2: training for {n_train_batches} batches on {config.dataset_name}")

    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client, name=f"{batch_idx:06d}",
                log_path=config.log_path, kind="state",
                loop_state={"batch": batch_idx}, ttl_seconds=config.ttl_seconds,
            )

        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        sampling_client = training_client.save_weights_and_get_sampling_client()

        # ─── Phase 1: Sample trajectories ───
        futures: list[Future] = []
        prompts: list[types.ModelInput] = []
        for row in batch_rows:
            convo = [*convo_prefix, {"role": "user", "content": row[q_key] + question_suffix}]
            model_input = renderer.build_generation_prompt(convo)
            futures.append(sampling_client.sample(prompt=model_input, num_samples=config.group_size, sampling_params=sampling_params))
            prompts.append(model_input)

        # Extract answers
        batch_answers = []
        for row in batch_rows:
            raw = row[a_key]
            if config.dataset_name == "math":
                try:
                    batch_answers.append(extract_boxed(raw))
                except ValueError:
                    batch_answers.append("")
            else:
                batch_answers.append(raw)

        # ─── Phase 2: Collect samples, build trees, submit branch probes ───
        groups_info = []
        n_all_incorrect = 0
        n_mixed = 0
        n_all_correct = 0
        n_probed = 0
        rewards_P = []

        for future, prompt, answer in tqdm(
            zip(futures, prompts, batch_answers), total=len(futures),
            desc=f"Batch {batch_idx} (sampling)",
        ):
            result = future.result()
            token_seqs = []
            logprob_seqs = []
            rewards = []

            for seq in result.sequences:
                token_seqs.append(seq.tokens)
                logprob_seqs.append(seq.logprobs)
                msg, _ = renderer.parse_response(seq.tokens)
                content = renderers.get_text_content(msg)
                rewards.append(get_reward(content, answer))

            mean_r = sum(rewards) / len(rewards)
            rewards_P.append(mean_r)
            all_same = all(r == rewards[0] for r in rewards)
            all_incorrect = all(r == 0 for r in rewards)

            if all_incorrect:
                n_all_incorrect += 1
            elif all_same:
                n_all_correct += 1
            else:
                n_mixed += 1

            # Skip all-correct (no signal)
            if all_same and not all_incorrect:
                groups_info.append(None)
                continue

            # Build prefix tree and find branching points
            tree = build_prefix_tree(token_seqs)
            bp_nodes = find_branching_points(tree, config.max_branching_points)

            # Submit branch probes in parallel
            all_branch_futures = []
            for bp_node in bp_nodes:
                bf = submit_branch_probes(
                    sampling_client, prompt, token_seqs, bp_node,
                    config.n_probes, config.max_branch_probes, config.max_tokens, stop,
                )
                all_branch_futures.append(bf)
            n_probed += 1

            groups_info.append({
                "token_seqs": token_seqs,
                "logprob_seqs": logprob_seqs,
                "rewards": rewards,
                "prompt": prompt,
                "answer": answer,
                "bp_nodes": bp_nodes,
                "branch_futures": all_branch_futures,
            })

        # ─── Phase 3: Collect probes, compute advantages, build datums ───
        datums_D: list[types.Datum] = []

        for group in groups_info:
            if group is None:
                continue

            token_seqs = group["token_seqs"]
            logprob_seqs = group["logprob_seqs"]
            rewards = group["rewards"]
            prompt = group["prompt"]
            answer = group["answer"]
            bp_nodes = group["bp_nodes"]
            branch_futures = group["branch_futures"]

            # Collect probe results per branching point
            branch_probe_results = []
            for bf_list in branch_futures:
                bp_results = []
                for tids, future in bf_list:
                    result = future.result()
                    n_success = 0
                    for seq in result.sequences:
                        msg, _ = renderer.parse_response(seq.tokens)
                        content = renderers.get_text_content(msg)
                        try:
                            if grade_answer(extract_boxed(content), answer):
                                n_success += 1
                        except ValueError:
                            pass
                    phi = n_success / max(len(result.sequences), 1)
                    bp_results.append((tids, phi_to_psi(phi)))
                branch_probe_results.append(bp_results)

            # Compute advantages
            step_advs = compute_branch_advantages(
                token_seqs, rewards, bp_nodes, branch_probe_results, config,
            )

            # Build datums
            for tokens, logprobs, tok_advs in zip(token_seqs, logprob_seqs, step_advs):
                if len(tokens) < 2:
                    continue
                ob_len = prompt.length - 1
                model_input = prompt.append(types.EncodedTextChunk(tokens=tokens[:-1]))
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advs = [0.0] * ob_len + tok_advs
                while len(padded_advs) < model_input.length:
                    padded_advs.append(0.0)
                padded_advs = padded_advs[:model_input.length]

                if model_input.length != len(target_tokens) or len(target_tokens) != len(padded_logprobs) or len(padded_logprobs) != len(padded_advs):
                    continue  # skip malformed

                datums_D.append(types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advs)),
                    },
                ))

        if not datums_D:
            logger.warning(f"Batch {batch_idx}: no datums")
            continue

        # ─── Phase 4: Train ───
        fwd_bwd_future = training_client.forward_backward(datums_D, loss_fn="importance_sampling")
        optim_future = training_client.optim_step(adam_params)
        fwd_bwd_future.result()
        optim_result = optim_future.result()
        if optim_result.metrics:
            metrics.update(optim_result.metrics)

        total = n_all_incorrect + n_mixed + n_all_correct
        metrics["time/total"] = time.time() - t_start
        metrics["reward/total"] = sum(rewards_P) / max(len(rewards_P), 1)
        metrics["rapo/n_datums"] = len(datums_D)
        metrics["rapo/n_probed"] = n_probed
        metrics["rapo/n_all_incorrect"] = n_all_incorrect
        metrics["rapo/n_mixed"] = n_mixed
        metrics["rapo/n_all_correct"] = n_all_correct
        metrics["rapo/frac_all_incorrect"] = n_all_incorrect / max(total, 1)
        ml_logger.log_metrics(metrics, step=batch_idx)

    checkpoint_utils.save_checkpoint(
        training_client=training_client, name="final",
        log_path=config.log_path, kind="both",
        loop_state={"batch": n_train_batches}, ttl_seconds=None,
    )
    ml_logger.close()
    logger.info("RAPO v2 training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
