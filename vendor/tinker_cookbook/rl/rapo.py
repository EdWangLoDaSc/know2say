"""
RAPO: Reachability-Aware Policy Optimization

Step-level credit assignment via reachability probes.
Integrates with the standard train.py pipeline by replacing compute_advantages().
"""

import logging
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Sequence

import chz
import numpy as np
import tinker
import tinker.types as types
import torch

from tinker_cookbook.rl.data_processing import compute_advantages as grpo_compute_advantages
from tinker_cookbook.rl.types import TrajectoryGroup

logger = logging.getLogger(__name__)


@chz.chz
class RAPOConfig:
    """Configuration for RAPO step-level advantage computation."""
    alpha_max: float = 0.5          # max weight for step-level advantage
    alpha_warmup_frac: float = 0.1  # fraction of training for α warmup
    beta: float = 0.3               # scaling for all-incorrect group advantages
    n_probes: int = 8               # probe rollouts per checkpoint
    checkpoint_fracs: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    eps: float = 0.01               # epsilon for logit transform


def phi_to_psi(phi: float, eps: float = 0.01) -> float:
    """Convert reachability probability to logit potential."""
    return float(np.log((phi + eps) / (1 - phi + eps)))


@dataclass
class ProbeRequest:
    """A pending probe request with its future."""
    depth: int
    future: Future


def submit_group_probes(
    sampling_client: tinker.SamplingClient,
    traj_group: TrajectoryGroup,
    rapo_cfg: RAPOConfig,
    max_tokens: int,
    stop: list[str] | list[int] | None = None,
) -> list[ProbeRequest]:
    """
    Submit probe rollouts for a group. Uses entropy-guided placement:
    probes at the most uncertain token positions. Returns futures (non-blocking).
    """
    if not traj_group.trajectories_G:
        return []

    # Use first trajectory for probing
    traj = traj_group.trajectories_G[0]
    if not traj.transitions:
        return []

    # Get the full token sequence and logprobs from transitions
    all_tokens: list[int] = []
    all_logprobs: list[float] = []
    for transition in traj.transitions:
        all_tokens.extend(transition.ac.tokens)
        all_logprobs.extend(transition.ac.logprobs)

    seq_len = len(all_tokens)
    if seq_len < 8:
        return []

    # Build prompt ModelInput from first observation
    prompt = traj.transitions[0].ob

    # Entropy-guided checkpoint selection:
    # Use negative logprob as uncertainty proxy (higher = more uncertain)
    # Select top-K most uncertain positions, but ensure minimum spacing
    n_checkpoints = len(rapo_cfg.checkpoint_fracs)
    min_spacing = max(seq_len // (n_checkpoints * 3), 4)  # avoid clustering

    # Score each position by uncertainty (negative logprob)
    uncertainty = [-lp for lp in all_logprobs]

    # Greedy selection with minimum spacing
    selected: list[int] = []
    scored = sorted(range(len(uncertainty)), key=lambda i: uncertainty[i], reverse=True)
    for pos in scored:
        if len(selected) >= n_checkpoints:
            break
        # Skip positions too close to already selected ones or at edges
        if pos < 4 or pos > seq_len - 4:
            continue
        if any(abs(pos - s) < min_spacing for s in selected):
            continue
        selected.append(pos)

    # Fallback: if not enough positions found, use evenly-spaced
    if len(selected) < n_checkpoints:
        selected = sorted(set(
            max(1, int(f * seq_len)) for f in rapo_cfg.checkpoint_fracs
        ))

    selected.sort()

    # Submit probes
    probe_requests: list[ProbeRequest] = []
    for depth in selected:
        prefix = all_tokens[:depth]
        probe_input = prompt.append(types.EncodedTextChunk(tokens=prefix))
        remaining = max(max_tokens - len(prefix), 64)
        probe_params = types.SamplingParams(
            max_tokens=remaining,
            stop=stop,
            temperature=1.0,
        )
        future = sampling_client.sample(
            prompt=probe_input,
            num_samples=rapo_cfg.n_probes,
            sampling_params=probe_params,
        )
        probe_requests.append(ProbeRequest(depth=depth, future=future))

    return probe_requests


def collect_probe_phis(
    probe_requests: list[ProbeRequest],
    grade_fn,  # Callable[[str], bool] - grades a response string
    parse_fn,  # Callable[[list[int]], str] - parses tokens to response string
) -> list[tuple[int, float]]:
    """Collect probe results and compute Φ at each checkpoint."""
    results: list[tuple[int, float]] = []
    for req in probe_requests:
        result = req.future.result()
        n_success = 0
        for seq in result.sequences:
            response_str = parse_fn(seq.tokens)
            if grade_fn(response_str):
                n_success += 1
        phi = n_success / max(len(result.sequences), 1)
        results.append((req.depth, phi))
    return results


def compute_rapo_advantages(
    trajectory_groups_P: list[TrajectoryGroup],
    probe_results_P: list[list[tuple[int, float]] | None],
    rapo_cfg: RAPOConfig,
    alpha: float,
) -> list[torch.Tensor | list[list[float]]]:
    """
    Compute RAPO advantages for all groups.

    For groups with probe results: returns per-step advantages (list of list of float).
    For groups without: returns standard GRPO scalar advantages (torch.Tensor).

    The return type is a union to maintain backward compatibility:
    - torch.Tensor of shape (G,) for standard GRPO groups
    - list[list[float]] for RAPO groups (per-trajectory, per-token)
    """
    # First get standard GRPO advantages
    grpo_advantages_P = grpo_compute_advantages(trajectory_groups_P)

    rapo_advantages_P: list[torch.Tensor | list[list[float]]] = []

    for group_idx, (traj_group, grpo_adv, probes) in enumerate(
        zip(trajectory_groups_P, grpo_advantages_P, probe_results_P)
    ):
        rewards = traj_group.get_total_rewards()
        mean_reward = sum(rewards) / len(rewards)
        all_same = all(r == rewards[0] for r in rewards)

        # No probe data or probing failed: use standard GRPO
        if probes is None or len(probes) == 0:
            rapo_advantages_P.append(grpo_adv)
            continue

        # Extract Φ curve
        ckpt_depths = [d for d, _ in probes]
        phis = [p for _, p in probes]
        psis = [phi_to_psi(p, rapo_cfg.eps) for p in phis]

        # Per-trajectory step-level advantages
        group_step_advantages: list[list[float]] = []

        for traj_idx, traj in enumerate(traj_group.trajectories_G):
            traj_adv = float(grpo_adv[traj_idx])

            # Get token count for this trajectory
            seq_len = sum(len(t.ac.tokens) for t in traj.transitions)
            if seq_len < 2:
                group_step_advantages.append([traj_adv])
                continue

            # Map checkpoint depths proportionally to this trajectory's length
            ref_len = sum(len(t.ac.tokens) for t in traj_group.trajectories_G[0].transitions)
            if ref_len > 0 and seq_len != ref_len:
                mapped_depths = [max(1, min(int(d * seq_len / ref_len), seq_len - 1)) for d in ckpt_depths]
            else:
                mapped_depths = [min(d, seq_len - 1) for d in ckpt_depths]

            # Build segment boundaries and ΔΨ
            # Only use intermediate Φ checkpoints for ΔΨ (not terminal reward).
            # Terminal reward is already captured by traj_adv (GRPO component).
            # Including it in ΔΨ causes the last segment to have extreme values
            # that dominate the advantage.
            segment_psis = psis  # just the probed checkpoints
            segment_boundaries = mapped_depths + [seq_len]

            # ΔΨ between consecutive checkpoints
            delta_psis: list[float] = []
            if len(segment_psis) >= 2:
                for k in range(len(segment_psis) - 1):
                    delta_psis.append(segment_psis[k + 1] - segment_psis[k])
            # Last segment gets 0 (no ΔΨ info beyond last checkpoint)
            delta_psis.append(0.0)

            # Prepend segment for tokens before first checkpoint
            first_ckpt = segment_boundaries[0] if segment_boundaries else seq_len
            if first_ckpt > 0:
                segment_boundaries = [0] + segment_boundaries
                delta_psis = [0.0] + delta_psis  # no info before first checkpoint

            # Normalize ΔΨ: z-score within the group's ΔΨ values
            dpsi_arr = [d for d in delta_psis if d != 0.0]
            if dpsi_arr:
                dpsi_std = max(float(np.std(dpsi_arr)), 0.1)
                dpsi_mean = float(np.mean(dpsi_arr))
                normalized_delta_psis = [(d - dpsi_mean) / dpsi_std if d != 0.0 else 0.0
                                         for d in delta_psis]
            else:
                normalized_delta_psis = delta_psis

            # Broadcast segment advantages to tokens
            token_advantages: list[float] = []
            seg_idx = 0
            token_pos = 0

            for transition in traj.transitions:
                n_tokens = len(transition.ac.tokens)
                for _ in range(n_tokens):
                    # Find which segment this token belongs to
                    while seg_idx < len(segment_boundaries) - 2 and token_pos >= segment_boundaries[seg_idx + 1]:
                        seg_idx += 1
                    seg_idx_clamped = min(seg_idx, len(normalized_delta_psis) - 1)
                    reach_adv = normalized_delta_psis[seg_idx_clamped]

                    if all_same:
                        # All-incorrect: only reachability signal, scaled by β
                        token_advantages.append(rapo_cfg.beta * reach_adv)
                    else:
                        # Mixed: α-blend of trajectory + step-level
                        token_advantages.append((1 - alpha) * traj_adv + alpha * reach_adv)

                    token_pos += 1

            group_step_advantages.append(token_advantages)

        rapo_advantages_P.append(group_step_advantages)

    return rapo_advantages_P


def get_alpha(batch_idx: int, total_batches: int, rapo_cfg: RAPOConfig) -> float:
    """Compute α for current training step."""
    warmup_steps = max(1, int(rapo_cfg.alpha_warmup_frac * total_batches))
    return rapo_cfg.alpha_max * min(1.0, batch_idx / warmup_steps)
