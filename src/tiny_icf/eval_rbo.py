"""Rank-Biased Overlap (RBO) evaluation metric.

RBO emphasizes top-ranked results, addressing Spearman's limitation of
masking poor performance on top items.
"""

import torch
import numpy as np
from typing import Optional


def rbo(
    list1: list | np.ndarray | torch.Tensor,
    list2: list | np.ndarray | torch.Tensor,
    p: float = 0.9,
) -> float:
    """
    Compute Rank-Biased Overlap (RBO) between two ranked lists.

    RBO is a position-biased metric that emphasizes top-ranked items,
    addressing Spearman's limitation of treating all positions equally.

    Args:
        list1: First ranked list (indices or values)
        list2: Second ranked list (indices or values)
        p: Persistence parameter (0 < p < 1), higher = more weight on top

    Returns:
        RBO score in [0, 1], higher is better

    Reference:
        Webber et al. "A Similarity Measure for Indefinite Rankings"
    """
    if len(list1) == 0 or len(list2) == 0:
        return 0.0

    # Convert to lists of indices if needed
    if isinstance(list1, torch.Tensor):
        list1 = list1.cpu().numpy()
    if isinstance(list2, torch.Tensor):
        list2 = list2.cpu().numpy()

    list1 = list(list1)
    list2 = list(list2)

    # Compute overlap at each depth
    rbo_sum = 0.0
    depth = min(len(list1), len(list2))

    for d in range(1, depth + 1):
        # Items at depth d
        items1 = set(list1[:d])
        items2 = set(list2[:d])

        # Overlap at depth d
        overlap = len(items1 & items2)
        union = len(items1 | items2)

        if union > 0:
            agreement = overlap / union
        else:
            agreement = 0.0

        # Weight by p^(d-1)
        weight = (1 - p) * (p ** (d - 1))
        rbo_sum += weight * agreement

    return rbo_sum


def compute_rbo_from_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    p: float = 0.9,
    top_k: Optional[int] = None,
) -> float:
    """
    Compute RBO between predicted and target rankings.

    Args:
        predictions: [N] predicted ICF scores
        targets: [N] ground truth ICF scores
        p: RBO persistence parameter (default: 0.9)
        top_k: If provided, only evaluate top K items

    Returns:
        RBO score
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        pred_np = predictions.cpu().numpy()
    else:
        pred_np = np.array(predictions)

    if isinstance(targets, torch.Tensor):
        target_np = targets.cpu().numpy()
    else:
        target_np = np.array(targets)

    # Get indices sorted by score (ascending: common first)
    pred_ranks = np.argsort(pred_np)
    target_ranks = np.argsort(target_np)

    # If top_k specified, only use top K
    if top_k is not None:
        pred_ranks = pred_ranks[:top_k]
        target_ranks = target_ranks[:top_k]

    # Compute RBO
    return rbo(pred_ranks, target_ranks, p=p)


def compute_rbo_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    top_k_values: list[int] = [10, 50, 100],
) -> dict[str, float]:
    """
    Compute RBO metrics at multiple top-K values.

    Args:
        predictions: [N] predicted ICF scores
        targets: [N] ground truth ICF scores
        top_k_values: List of K values to evaluate

    Returns:
        Dictionary of RBO scores at different K values
    """
    results = {}

    for k in top_k_values:
        if k <= len(predictions):
            rbo_score = compute_rbo_from_predictions(predictions, targets, top_k=k)
            results[f"rbo_top_{k}"] = rbo_score

    # Full RBO (all items)
    results["rbo_full"] = compute_rbo_from_predictions(predictions, targets)

    return results
