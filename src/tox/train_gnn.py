"""B3/B4 - GNN training. This portion: the masked multi-task loss.

Training loop, early stopping, checkpoints and C3 prediction export are added in later commits.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_with_logits(logits: torch.Tensor, y: torch.Tensor,
                           pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Multi-task binary cross-entropy that ignores missing labels.

    logits: [B, T] raw scores.  y: [B, T] with 1.0 / 0.0 / NaN (NaN = not measured).
    pos_weight: optional [T] weight on the positive term of each task (as in
    `F.binary_cross_entropy_with_logits`), computed from TRAINING labels only.

    Returns the mean loss over labelled entries only, so a batch with many missing labels is
    not diluted. Missing entries contribute exactly 0 to the loss and 0 to the gradient. A batch
    with no labels at all returns 0 (finite, with zero gradients) instead of NaN.

    Why not just call BCE on y: NaN targets would make the loss NaN. We replace NaN by a dummy 0
    *before* the BCE (so no NaN is ever computed) and then multiply by the mask, which zeroes
    both the value and the gradient of those entries.
    """
    if logits.shape != y.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and y {tuple(y.shape)} must match")
    mask = ~torch.isnan(y)
    y_safe = torch.where(mask, y, torch.zeros_like(y))
    per_entry = F.binary_cross_entropy_with_logits(
        logits, y_safe, pos_weight=pos_weight, reduction="none")
    per_entry = per_entry * mask                      # missing -> 0 value, 0 gradient
    n = mask.sum().clamp(min=1)                       # avoid 0/0 when nothing is labelled
    return per_entry.sum() / n
