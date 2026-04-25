"""
RAPO: Reachability-Aware Policy Optimization

GRPO + step-level credit assignment via reachability probes.
Based on rl_loop.py pattern.

Key differences from standard GRPO:
1. For mixed groups: α-mixture of trajectory-level + reachability increment advantages
2. For all-incorrect groups: reachability probes provide non-zero gradients
3. Adaptive α per group based on reward variance

Usage:
    python -m tinker_cookbook.recipes.math_rl.rapo_train
    python -m tinker_cookbook.recipes.math_rl.rapo_train alpha_max=0.5 n_probes=8
"""

import logging
import math
import time
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
    _get_hendrycks_math_train, _get_hendrycks_math_test,
)
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/rapo"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    batch_size: int = 128
    group_size: int = 16
    learning_rate: float = 4e-5
    lora_rank: int = 32
    save_every: int = 20
    max_tokens: int = 256
    ttl_seconds: int | None = 604800
    dataset_name: str = "gsm8k"  # "gsm8k" or "math"

    # RAPO-specific
    alpha_max: float = 0.3       # max weight for step-level advantage (reduced from 0.5)
    alpha_warmup_frac: float = 0.3  # fraction of training for α warmup (slower warmup)
    beta: float = 0.3            # scaling for all-incorrect group advantages
    n_probes: int = 8            # probe rollouts per checkpoint
    n_checkpoints: int = 4       # checkpoints per trajectory (evenly spaced)
    probe_all_incorrect_only: bool = False  # if True, only probe all-incorrect groups

    # Logging
    wandb_project: str | None = None
    wandb_name: str | None = None


def get_reward_gsm8k(response: str, answer: str) -> float:
    try:
        given_answer = extract_boxed(response)
        ground_truth = extract_gsm8k_final_answer(answer)
        return 1.0 if grade_answer(given_answer, ground_truth) else 0.0
    except ValueError:
        return 0.0


def get_reward_math(response: str, answer: str) -> float:
    """answer is already extracted boxed answer from MATH dataset."""
    try:
        given_answer = extract_boxed(response)
        return 1.0 if grade_answer(given_answer, answer) else 0.0
    except ValueError:
        return 0.0


def phi_to_psi(phi: float, eps: float = 0.01) -> float:
    """Convert reachability probability to logit potential."""
    return float(np.log((phi + eps) / (1 - phi + eps)))


def submit_probe_futures(
    sampling_client: tinker.SamplingClient,
    prompt: types.ModelInput,
    token_seqs: list[list[int]],
    renderer: renderers.Renderer,
    n_probes: int,
    max_tokens: int,
    checkpoint_fracs: list[float],
) -> list[list[tuple[int, Future]]]:
    """Submit all probe requests in parallel. Returns futures per trajectory."""
    all_futures: list[list[tuple[int, Future]]] = []
    stop = renderer.get_stop_sequences()

    for traj_idx, tokens in enumerate(token_seqs):
        seq_len = len(tokens)
        traj_futures: list[tuple[int, Future]] = []

        if seq_len < 8:
            all_futures.append(traj_futures)
            continue

        # Only probe the first trajectory in the group (share Φ across all)
        if traj_idx > 0:
            all_futures.append([])
            continue

        ckpt_depths = sorted(set(max(1, int(f * seq_len)) for f in checkpoint_fracs))
        for depth in ckpt_depths:
            prefix = tokens[:depth]
            probe_input = prompt.append(types.EncodedTextChunk(tokens=prefix))
            remaining = max(max_tokens - len(prefix), 64)
            probe_params = types.SamplingParams(
                max_tokens=remaining, stop=stop, temperature=1.0,
            )
            future = sampling_client.sample(
                prompt=probe_input, num_samples=n_probes, sampling_params=probe_params,
            )
            traj_futures.append((depth, future))

        all_futures.append(traj_futures)

    return all_futures


def collect_phis(
    probe_futures: list[tuple[int, Future]],
    answer: str,
    renderer: renderers.Renderer,
    n_probes: int,
) -> list[tuple[int, float]]:
    """Collect probe results and compute Φ at each checkpoint."""
    results = []
    for depth, future in probe_futures:
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
        results.append((depth, n_success / n_probes))
    return results


def compute_rapo_advantages(
    token_seqs: list[list[int]],
    rewards: list[float],
    phis_with_depths: list[tuple[int, float]],
    config: Config,
    alpha: float,
) -> list[list[float]]:
    """
    Compute RAPO step-level advantages for a group.
    Uses pre-computed Φ values (from probe futures).

    Returns: list of per-token advantages for each trajectory.
    """
    G = len(token_seqs)
    mean_reward = sum(rewards) / G

    all_same = all(r == rewards[0] for r in rewards)
    traj_advantages = [r - mean_reward for r in rewards]

    # If no probe data, fallback to GRPO
    if not phis_with_depths:
        return [[traj_adv] * len(tokens) for traj_adv, tokens in zip(traj_advantages, token_seqs)]

    # Shared Φ curve (probed from trajectory 0)
    ckpt_depths = [d for d, _ in phis_with_depths]
    phis = [p for _, p in phis_with_depths]
    psis = [phi_to_psi(p) for p in phis]

    all_advantages: list[list[float]] = []

    for traj_idx in range(G):
        tokens = token_seqs[traj_idx]
        seq_len = len(tokens)
        traj_adv = traj_advantages[traj_idx]

        if seq_len < 8:
            all_advantages.append([traj_adv] * seq_len)
            continue

        # Build segment boundaries for this trajectory
        # Remap checkpoint depths proportionally if seq lengths differ
        ref_len = len(token_seqs[0])
        if ref_len > 0 and seq_len != ref_len:
            mapped_depths = [max(1, int(d * seq_len / ref_len)) for d in ckpt_depths]
        else:
            mapped_depths = list(ckpt_depths)
        mapped_depths = [min(d, seq_len - 1) for d in mapped_depths]

        # ΔΨ between consecutive checkpoints only (no terminal reward in ΔΨ)
        segment_psis = psis
        segment_boundaries = mapped_depths + [seq_len]

        delta_psis: list[float] = []
        if len(segment_psis) >= 2:
            for k in range(len(segment_psis) - 1):
                delta_psis.append(segment_psis[k + 1] - segment_psis[k])
        delta_psis.append(0.0)  # last segment

        # Prepend segment before first checkpoint
        first_ckpt = segment_boundaries[0] if segment_boundaries else seq_len
        if first_ckpt > 0:
            segment_boundaries = [0] + segment_boundaries
            delta_psis = [0.0] + delta_psis

        # Z-score normalize ΔΨ
        dpsi_nonzero = [d for d in delta_psis if d != 0.0]
        if dpsi_nonzero:
            dpsi_std = max(float(np.std(dpsi_nonzero)), 0.1)
            dpsi_mean = float(np.mean(dpsi_nonzero))
            norm_delta_psis = [(d - dpsi_mean) / dpsi_std if d != 0.0 else 0.0
                               for d in delta_psis]
        else:
            norm_delta_psis = delta_psis

        # Broadcast to tokens
        token_advantages = [0.0] * seq_len
        seg_idx = 0
        for t in range(seq_len):
            while seg_idx < len(segment_boundaries) - 2 and t >= segment_boundaries[seg_idx + 1]:
                seg_idx += 1
            si = min(seg_idx, len(norm_delta_psis) - 1)
            reach_adv = norm_delta_psis[si]

            if all_same:
                token_advantages[t] = config.beta * reach_adv
            else:
                token_advantages[t] = (1 - alpha) * traj_adv + alpha * reach_adv

        all_advantages.append(token_advantages)

    return all_advantages


def main(config: Config):
    # Setup logging
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

    question_suffix = MathEnv.question_suffix()  # " Write your answer in \boxed{} format."
    convo_prefix = MathEnv.standard_fewshot_prefix()

    if config.dataset_name == "gsm8k":
        ds = datasets.load_dataset("openai/gsm8k", "main")
        assert isinstance(ds, datasets.DatasetDict)
        train_dataset = ds["train"]
        question_key = "question"
        answer_key = "answer"
        get_reward = get_reward_gsm8k
        question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."
    elif config.dataset_name == "math":
        train_dataset = _get_hendrycks_math_train().shuffle(seed=0)
        question_key = "problem"
        answer_key = "solution"  # will extract boxed from solution
        get_reward = get_reward_math
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")

    n_train_batches = len(train_dataset) // config.batch_size

    service_client = tinker.ServiceClient(base_url=config.base_url)
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(f"Training for {n_train_batches} batches (RAPO)")

    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()

        # α schedule: linear warmup
        warmup_steps = max(1, int(config.alpha_warmup_frac * n_train_batches))
        alpha = config.alpha_max * min(1.0, batch_idx / warmup_steps)

        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
            "rapo/alpha": alpha,
        }

        # Checkpoint
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
                ttl_seconds=config.ttl_seconds,
            )

        # Get batch
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        sampling_client = training_client.save_weights_and_get_sampling_client()

        # ─── Phase 1: Sample trajectories ───
        futures: list[Future[types.SampleResponse]] = []
        prompts: list[types.ModelInput] = []
        for row in batch_rows:
            question = row[question_key]
            convo = [*convo_prefix, {"role": "user", "content": question + question_suffix}]
            model_input = renderer.build_generation_prompt(convo)
            future = sampling_client.sample(
                prompt=model_input,
                num_samples=config.group_size,
                sampling_params=sampling_params,
            )
            futures.append(future)
            prompts.append(model_input)

        # ─── Phase 2: Collect samples, submit probes in parallel ───
        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        n_all_incorrect = 0
        n_mixed = 0
        n_all_correct = 0
        n_probed = 0

        checkpoint_fracs = [0.10, 0.25, 0.50, 0.75]
        if config.n_checkpoints != 4:
            checkpoint_fracs = [(i + 1) / (config.n_checkpoints + 1) for i in range(config.n_checkpoints)]

        # First pass: collect all samples and submit all probes
        groups_data = []
        # Extract answers for this batch
        batch_answers = []
        for row in batch_rows:
            raw_answer = row[answer_key]
            if config.dataset_name == "math":
                try:
                    batch_answers.append(extract_boxed(raw_answer))
                except ValueError:
                    batch_answers.append("")
            else:
                batch_answers.append(raw_answer)

        for future, prompt, answer in tqdm(
            zip(futures, prompts, batch_answers),
            total=len(futures),
            desc=f"Batch {batch_idx} (sampling)",
        ):
            sample_result = future.result()
            token_seqs: list[list[int]] = []
            logprob_seqs: list[list[float]] = []
            rewards: list[float] = []

            for seq in sample_result.sequences:
                token_seqs.append(seq.tokens)
                logprob_seqs.append(seq.logprobs)
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = renderers.get_text_content(parsed_msg)
                rewards.append(get_reward(content, answer))

            mean_reward = sum(rewards) / len(rewards)
            rewards_P.append(mean_reward)

            all_same = all(r == rewards[0] for r in rewards)
            all_incorrect = all(r == 0 for r in rewards)
            if all_incorrect:
                n_all_incorrect += 1
            elif all_same:
                n_all_correct += 1
            else:
                n_mixed += 1

            # Decide whether to probe and submit futures
            should_probe = (not all_same) or all_incorrect
            if all_same and not all_incorrect:
                groups_data.append(None)  # all-correct, skip
                continue

            probe_futures_list: list[list[tuple[int, Future]]] = []
            if should_probe:
                n_probed += 1
                probe_futures_list = submit_probe_futures(
                    sampling_client, prompt, token_seqs, renderer,
                    config.n_probes, config.max_tokens, checkpoint_fracs,
                )

            groups_data.append({
                "token_seqs": token_seqs,
                "logprob_seqs": logprob_seqs,
                "rewards": rewards,
                "prompt": prompt,
                "answer": answer,
                "all_same": all_same,
                "all_incorrect": all_incorrect,
                "probe_futures": probe_futures_list,
            })

        # Second pass: collect probe results and compute advantages
        for group in groups_data:
            if group is None:
                continue

            token_seqs = group["token_seqs"]
            logprob_seqs = group["logprob_seqs"]
            rewards = group["rewards"]
            prompt = group["prompt"]
            answer = group["answer"]
            probe_futures_list = group["probe_futures"]

            # Collect Φ from probes (traj 0 has the futures)
            if probe_futures_list and probe_futures_list[0]:
                phis_with_depths = collect_phis(
                    probe_futures_list[0], answer, renderer, config.n_probes,
                )
            else:
                phis_with_depths = []

            step_advantages = compute_rapo_advantages(
                token_seqs, rewards, phis_with_depths, config, alpha,
            )

            # Check signal for all-incorrect groups
            if group["all_same"]:
                has_signal = any(any(abs(a) > 1e-6 for a in adv) for adv in step_advantages)
                if not has_signal:
                    continue

            # Build datums with per-token advantages
            for traj_idx, (tokens, logprobs, tok_advantages) in enumerate(
                zip(token_seqs, logprob_seqs, step_advantages)
            ):
                if len(tokens) < 2:
                    continue  # Skip empty/single-token outputs
                ob_len = prompt.length - 1
                model_input = prompt.append(types.EncodedTextChunk(tokens=tokens[:-1]))
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + logprobs

                # Per-token advantages: pad observation tokens with 0
                padded_advantages = [0.0] * ob_len + tok_advantages
                # Ensure lengths match (tok_advantages may be len(tokens), need len(tokens) for model_input)
                while len(padded_advantages) < model_input.length:
                    padded_advantages.append(0.0)
                padded_advantages = padded_advantages[:model_input.length]

                assert (
                    model_input.length == len(target_tokens)
                    == len(padded_logprobs) == len(padded_advantages)
                ), (
                    f"Length mismatch: model_input={model_input.length}, "
                    f"targets={len(target_tokens)}, logprobs={len(padded_logprobs)}, "
                    f"advantages={len(padded_advantages)}"
                )

                datum = types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                )
                datums_D.append(datum)

        # ─── Phase 3: Training step ───
        if not datums_D:
            logger.warning(f"Batch {batch_idx}: no datums to train on")
            continue

        fwd_bwd_future = training_client.forward_backward(datums_D, loss_fn="importance_sampling")
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_step_future.result()

        if optim_result.metrics:
            metrics.update(optim_result.metrics)

        # Log metrics
        total_groups = n_all_incorrect + n_mixed + n_all_correct
        metrics["time/total"] = time.time() - t_start
        metrics["reward/total"] = sum(rewards_P) / len(rewards_P) if rewards_P else 0
        metrics["rapo/n_datums"] = len(datums_D)
        metrics["rapo/n_probed"] = n_probed
        metrics["rapo/n_all_incorrect"] = n_all_incorrect
        metrics["rapo/n_mixed"] = n_mixed
        metrics["rapo/n_all_correct"] = n_all_correct
        metrics["rapo/frac_all_incorrect"] = n_all_incorrect / max(total_groups, 1)

        ml_logger.log_metrics(metrics, step=batch_idx)

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
        ttl_seconds=None,
    )
    ml_logger.close()
    logger.info("RAPO training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
