"""Unlearning loss functions: Gradient Ascent, Gradient Difference, NPO.

All operate on token-level causal-LM outputs. Two utility-constrained methods
(GradDiff, NPO) add a retain term — these are where quantization-recovery is dramatic,
because the constraint keeps the unlearned weights close to the original (small
perturbation that int4 rounding erases). GA has no retain term and serves as a control.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def token_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean next-token CE over non-masked (answer) tokens. Standard LM loss."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log p(token) over answer tokens, per example -> shape [batch]."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    logp = F.log_softmax(shift_logits, dim=-1)
    mask = shift_labels != IGNORE_INDEX
    gather_labels = shift_labels.clone()
    gather_labels[~mask] = 0
    tok_logp = logp.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)
    tok_logp = tok_logp * mask
    return tok_logp.sum(dim=-1)


def ga_loss(forget_logits, forget_labels):
    """Gradient ascent: minimize negative forget CE == push forget loss up."""
    return -token_cross_entropy(forget_logits, forget_labels)


def grad_diff_loss(forget_logits, forget_labels, retain_logits, retain_labels, retain_weight=1.0):
    """GradDiff: ascend on forget, descend (normal SFT) on retain."""
    return -token_cross_entropy(forget_logits, forget_labels) + retain_weight * token_cross_entropy(
        retain_logits, retain_labels
    )


def npo_loss(
    forget_logits,
    forget_labels,
    ref_forget_logprob,
    retain_logits=None,
    retain_labels=None,
    beta=0.1,
    retain_weight=1.0,
):
    """Negative Preference Optimization (Zhang et al. 2024).

    L_NPO = (2/beta) * mean( -log sigmoid( -beta * (logp_theta - logp_ref) ) )

    Treats the forget answer as a *negative* preference relative to the frozen reference
    policy, which gives a bounded, more stable signal than raw gradient ascent. We add an
    SFT retain term to preserve utility (the constraint that makes recovery possible).
    """
    cur_logprob = sequence_logprob(forget_logits, forget_labels)          # [batch]
    log_ratio = cur_logprob - ref_forget_logprob                           # [batch]
    loss = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio).mean()
    if retain_logits is not None and retain_weight > 0:
        loss = loss + retain_weight * token_cross_entropy(retain_logits, retain_labels)
    return loss


def needs_reference(method_type: str) -> bool:
    """NPO needs a frozen copy of the pre-unlearning model for the log-ratio."""
    return method_type == "npo"
