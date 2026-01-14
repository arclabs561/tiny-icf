"""Differentiable sorting-based ranking losses.

Uses diffsort or fast-soft-sort to directly optimize ranking quality.
This allows gradients to flow through the sorting operation, enabling
direct optimization of Spearman correlation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _try_import_diffsort():
    """Try to import diffsort, return None if not available."""
    try:
        from diffsort import DiffSortNet
        return DiffSortNet
    except ImportError:
        return None


def _try_import_fast_soft_sort():
    """Try to import fast-soft-sort, return None if not available."""
    try:
        from fast_soft_sort.pytorch_ops import soft_rank, soft_sort
        return soft_rank, soft_sort
    except ImportError:
        return None, None


def spearman_loss_diffsort(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    steepness: float = 5.0,
) -> torch.Tensor:
    """
    Spearman correlation loss using differentiable sorting (diffsort).
    
    Uses DiffSortNet to get differentiable ranks, then computes correlation.
    
    Args:
        predictions: [Batch] or [Batch, 1] model predictions
        targets: [Batch] or [Batch, 1] ground truth ICF scores
        steepness: Steepness parameter for diffsort (higher = sharper sorting)
    
    Returns:
        Loss value (1 - Spearman correlation, so lower is better)
    """
    DiffSortNet = _try_import_diffsort()
    if DiffSortNet is None:
        raise ImportError("diffsort not installed. Install with: pip install diffsort")
    
    # Flatten to [Batch]
    if predictions.dim() > 1:
        predictions = predictions.squeeze(1)
    if targets.dim() > 1:
        targets = targets.squeeze(1)
    
    batch_size = len(predictions)
    if batch_size < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    # For small batches, use a simpler approach
    # For larger batches, use full diffsort
    if batch_size <= 16:
        # Use bitonic network (works for powers of 2, or pad)
        # Pad to next power of 2
        next_power = 2 ** (batch_size - 1).bit_length()
        if next_power < 2:
            next_power = 2
        
        # Pad predictions and targets
        if batch_size < next_power:
            padding = next_power - batch_size
            pred_padded = torch.cat([predictions, predictions[-1].repeat(padding)])
            target_padded = torch.cat([targets, targets[-1].repeat(padding)])
        else:
            pred_padded = predictions
            target_padded = targets
        
        # Reshape for DiffSortNet: [1, Batch]
        pred_reshaped = pred_padded.unsqueeze(0)  # [1, NextPower]
        target_reshaped = target_padded.unsqueeze(0)  # [1, NextPower]
        
        # Create sorters
        pred_sorter = DiffSortNet('bitonic', next_power, steepness=steepness)
        target_sorter = DiffSortNet('bitonic', next_power, steepness=steepness)
        
        # Get sorted values and permutation matrices
        pred_sorted, pred_P = pred_sorter(pred_reshaped)
        target_sorted, target_P = target_sorter(target_reshaped)
        
        # Extract ranks from permutation matrices (only for original batch)
        # Rank[i] = position in sorted order = sum of P[:, i] * [0, 1, 2, ...]
        ranks = torch.arange(next_power, dtype=torch.float32, device=predictions.device)
        pred_ranks_full = torch.sum(pred_P[0] * ranks.unsqueeze(0), dim=1)
        target_ranks_full = torch.sum(target_P[0] * ranks.unsqueeze(0), dim=1)
        
        # Use only original batch size
        pred_ranks = pred_ranks_full[:batch_size]
        target_ranks = target_ranks_full[:batch_size]
    else:
        # For larger batches, use odd-even network or process in chunks
        # For now, fall back to simpler ranking approach
        # TODO: Implement chunked processing for large batches
        pred_ranks = torch.argsort(torch.argsort(predictions, descending=False), descending=False).float()
        target_ranks = torch.argsort(torch.argsort(targets, descending=False), descending=False).float()
    
    # Compute Spearman correlation (Pearson on ranks)
    pred_ranks_centered = pred_ranks - pred_ranks.mean()
    target_ranks_centered = target_ranks - target_ranks.mean()
    
    numerator = (pred_ranks_centered * target_ranks_centered).sum()
    pred_std = torch.sqrt((pred_ranks_centered ** 2).sum() + 1e-8)
    target_std = torch.sqrt((target_ranks_centered ** 2).sum() + 1e-8)
    
    spearman = numerator / (pred_std * target_std + 1e-8)
    
    # Loss = 1 - Spearman (so lower is better, Spearman higher is better)
    loss = 1.0 - spearman
    
    return loss


def spearman_loss_fast_soft_sort(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    regularization_strength: float = 1.0,
) -> torch.Tensor:
    """
    Spearman correlation loss using fast-soft-sort.
    
    Uses soft_rank to get differentiable ranks, then computes correlation.
    
    Args:
        predictions: [Batch] or [Batch, 1] model predictions
        targets: [Batch] or [Batch, 1] ground truth ICF scores
        regularization_strength: Regularization strength (higher = sharper ranking)
    
    Returns:
        Loss value (1 - Spearman correlation)
    """
    soft_rank, _ = _try_import_fast_soft_sort()
    if soft_rank is None:
        raise ImportError("fast-soft-sort not installed. Install from: https://github.com/google-research/fast-soft-sort")
    
    # Flatten to [Batch]
    if predictions.dim() > 1:
        predictions = predictions.squeeze(1)
    if targets.dim() > 1:
        targets = targets.squeeze(1)
    
    batch_size = len(predictions)
    if batch_size < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    # Reshape for soft_rank: [1, Batch]
    pred_reshaped = predictions.unsqueeze(0)  # [1, Batch]
    target_reshaped = targets.unsqueeze(0)  # [1, Batch]
    
    # Get soft ranks
    pred_ranks = soft_rank(pred_reshaped, regularization_strength=regularization_strength)[0]  # [Batch]
    target_ranks = soft_rank(target_reshaped, regularization_strength=regularization_strength)[0]  # [Batch]
    
    # Compute Spearman correlation (Pearson on ranks)
    pred_ranks_centered = pred_ranks - pred_ranks.mean()
    target_ranks_centered = target_ranks - target_ranks.mean()
    
    numerator = (pred_ranks_centered * target_ranks_centered).sum()
    pred_std = torch.sqrt((pred_ranks_centered ** 2).sum() + 1e-8)
    target_std = torch.sqrt((target_ranks_centered ** 2).sum() + 1e-8)
    
    spearman = numerator / (pred_std * target_std + 1e-8)
    
    # Loss = 1 - Spearman
    loss = 1.0 - spearman
    
    return loss


class DifferentiableSortingLoss(nn.Module):
    """
    Loss function using differentiable sorting to directly optimize Spearman correlation.
    
    Supports both diffsort and fast-soft-sort backends.
    """
    
    def __init__(
        self,
        method: str = "fast_soft_sort",
        regularization_strength: float = 1.0,
        steepness: float = 5.0,
        huber_delta: float = 0.1,
        huber_weight: float = 0.5,
    ):
        """
        Args:
            method: "diffsort" or "fast_soft_sort"
            regularization_strength: For fast-soft-sort (higher = sharper)
            steepness: For diffsort (higher = sharper)
            huber_delta: Delta for Huber loss component
            huber_weight: Weight for Huber loss (rest is Spearman loss)
        """
        super().__init__()
        self.method = method
        self.regularization_strength = regularization_strength
        self.steepness = steepness
        self.huber_delta = huber_delta
        self.huber_weight = huber_weight
        
        # Check availability
        if method == "diffsort":
            if _try_import_diffsort() is None:
                raise ImportError("diffsort not installed. Install with: pip install diffsort")
        elif method == "fast_soft_sort":
            if _try_import_fast_soft_sort()[0] is None:
                raise ImportError("fast-soft-sort not installed. See: https://github.com/google-research/fast-soft-sort")
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: [Batch, 1] or [Batch] model predictions
            targets: [Batch, 1] or [Batch] ground truth ICF scores
        
        Returns:
            Scalar loss value
        """
        # Ensure same shape
        if predictions.dim() > 1:
            predictions_flat = predictions.squeeze(1)
        else:
            predictions_flat = predictions
        
        if targets.dim() > 1:
            targets_flat = targets.squeeze(1)
        else:
            targets_flat = targets
        
        # Huber loss for absolute accuracy
        huber = F.smooth_l1_loss(
            predictions_flat, targets_flat,
            reduction='mean', beta=self.huber_delta
        )
        
        # Spearman loss using differentiable sorting
        if self.method == "diffsort":
            spearman_loss = spearman_loss_diffsort(
                predictions, targets, steepness=self.steepness
            )
        else:  # fast_soft_sort
            spearman_loss = spearman_loss_fast_soft_sort(
                predictions, targets,
                regularization_strength=self.regularization_strength
            )
        
        # Combine
        total_loss = self.huber_weight * huber + (1.0 - self.huber_weight) * spearman_loss
        
        return total_loss


def check_differentiable_sorting_available() -> dict:
    """Check which differentiable sorting libraries are available."""
    diffsort_available = _try_import_diffsort() is not None
    fast_soft_sort_available = _try_import_fast_soft_sort()[0] is not None
    
    return {
        "diffsort": diffsort_available,
        "fast_soft_sort": fast_soft_sort_available,
    }

