"""Sequence-level clipped on-policy objective for unordered interest sets."""

from __future__ import annotations

import torch


def sequence_log_ratio(
    current_token_logps: torch.Tensor, old_token_logps: torch.Tensor
) -> torch.Tensor:
    """Length-normalized log likelihood ratio in Eq. 6."""

    if current_token_logps.shape != old_token_logps.shape:
        raise ValueError("current and old token log-probabilities must align")
    if current_token_logps.numel() == 0:
        raise ValueError("a rollout must contain at least one response token")
    return (current_token_logps - old_token_logps).mean()


def clipped_sequence_opd_loss(
    current_token_logps: torch.Tensor,
    old_token_logps: torch.Tensor,
    advantage: torch.Tensor | float,
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Eq. 4--6 for a single student-generated rollout."""

    log_ratio = sequence_log_ratio(current_token_logps, old_token_logps).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    adv = torch.as_tensor(advantage, dtype=ratio.dtype, device=ratio.device)
    loss = -torch.minimum(ratio * adv, clipped * adv)
    return loss, {
        "sequence_log_ratio": log_ratio.detach(),
        "sequence_ratio": ratio.detach(),
        "clipped_ratio": clipped.detach(),
    }


def sampled_reverse_kl(
    current_token_logps: torch.Tensor, reference_token_logps: torch.Tensor
) -> torch.Tensor:
    """Non-negative sampled KL estimator used as the frozen-SFT anchor."""

    if current_token_logps.shape != reference_token_logps.shape:
        raise ValueError("policy and reference token log-probabilities must align")
    reverse_log_ratio = (reference_token_logps - current_token_logps).clamp(-20.0, 20.0)
    return (reverse_log_ratio.exp() - reverse_log_ratio - 1.0).mean()

