"""Loss functions for training: Huber + Ranking loss + NeuralNDCG + Spearman.

This module consolidates all loss functions:
- Basic losses: Huber, Ranking
- Research losses: NeuralNDCG (from loss_research.py)
- Listwise losses: LambdaRank, ApproxNDCG (from loss_listwise.py)
- Spearman loss: Differentiable Spearman correlation (from loss_spearman.py)
- Multi-objective losses: (from loss_multi.py)

Now supports rank-relax integration for optimized Spearman loss.
"""

from typing import Optional, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import rank-relax for optimized Spearman loss
# Use ONLY public API - no local hacks or examples directory
try:
    import rank_relax
    
    # Check if spearman_loss_pytorch is in public API
    if hasattr(rank_relax, 'spearman_loss_pytorch'):
        # Use public PyTorch tensor function if available
        rank_relax_spearman_loss = rank_relax.spearman_loss_pytorch
        HAS_RANK_RELAX = True
    elif hasattr(rank_relax, 'spearman_loss'):
        # Fallback: Use public spearman_loss (expects lists, we'll wrap it)
        # We'll create a wrapper that converts tensors to lists
        def _spearman_loss_wrapper(predictions, targets, regularization_strength):
            """Wrapper to convert tensors to lists for public API."""
            import torch
            if isinstance(predictions, torch.Tensor):
                pred_list = predictions.detach().cpu().tolist()
            else:
                pred_list = predictions
            if isinstance(targets, torch.Tensor):
                target_list = targets.detach().cpu().tolist()
            else:
                target_list = targets
            
            loss_val = rank_relax.spearman_loss(pred_list, target_list, regularization_strength)
            
            # Convert back to tensor if input was tensor
            if isinstance(predictions, torch.Tensor):
                return torch.tensor(loss_val, device=predictions.device, dtype=predictions.dtype, requires_grad=predictions.requires_grad)
            return loss_val
        
        rank_relax_spearman_loss = _spearman_loss_wrapper
        HAS_RANK_RELAX = True
    else:
        HAS_RANK_RELAX = False
        rank_relax_spearman_loss = None
except ImportError:
    HAS_RANK_RELAX = False
    rank_relax_spearman_loss = None


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    """
    Huber loss (smooth L1).
    
    Acts like MSE for small errors, MAE for large errors.
    Prevents rare word outliers from exploding gradients.
    """
    error = pred - target
    is_small = torch.abs(error) < delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * torch.abs(error) - 0.5 * delta**2
    return torch.where(is_small, squared_loss, linear_loss).mean()


def ranking_loss(
    pred1: torch.Tensor, pred2: torch.Tensor, margin: float = 0.1, 
    target_diff: Optional[torch.Tensor] = None, smooth: bool = True, temperature: float = 10.0
) -> torch.Tensor:
    """
    Pairwise ranking loss with smooth rewards.
    
    Enforces that if word1 is more common than word2,
    then pred1 < pred2 (lower ICF = more common).
    
    Args:
        pred1: Predictions for more common words
        pred2: Predictions for less common words
        margin: Minimum margin between predictions
        target_diff: Optional [N_pairs] actual ICF differences (for weighted loss)
        smooth: If True, use smooth sigmoid instead of hard ReLU
        temperature: Temperature for smooth sigmoid (higher = sharper)
    """
    # We want pred1 < pred2 (common < rare in ICF space)
    diff = pred2 - pred1  # Should be positive if correctly ranked
    
    if smooth:
        # Smooth sigmoid-based loss: loss = sigmoid((margin - diff) * temperature)
        # This provides smooth gradients even when diff is close to margin
        # Higher temperature = sharper transition, lower = smoother
        violation = margin - diff  # Positive when violated
        loss = torch.sigmoid(violation * temperature)
    else:
        # Original hard ReLU loss
        loss = F.relu(margin - diff)
    
    # Weight by target difference if provided (larger differences = more important)
    if target_diff is not None:
        # Validate that target_diff matches the number of pairs
        if len(target_diff) != len(loss):
            # Mismatch: use uniform weighting as fallback
            loss = loss.mean()
        else:
            # Normalize target_diff to [0, 1] range for weighting
            # Use softmax-like weighting: exp(alpha * diff) / sum(exp(alpha * diff))
            # This emphasizes pairs with larger ICF differences
            weights = torch.softmax(target_diff * 5.0, dim=0)  # Scale factor controls emphasis
            weights_sum = weights.sum()
            if weights_sum > 1e-8:  # Avoid division by zero
                loss = (loss * weights).sum() / weights_sum  # Weighted average
            else:
                loss = loss.mean()  # Fallback to uniform if weights sum to zero
    else:
        loss = loss.mean()
    
    return loss


def neural_ndcg_loss_simple(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: Optional[int] = None,
) -> torch.Tensor:
    """
    Simplified NeuralNDCG loss for integration.
    
    Based on research showing NeuralNDCG outperforms other ranking losses.
    This is a simplified version that works well in practice.
    
    Args:
        predictions: [Batch] predicted scores
        targets: [Batch] ICF scores (higher = rarer)
        k: Top-k cutoff (None = full batch)
    
    Returns:
        Scalar loss (1 - NDCG approximation)
    """
    batch_size = predictions.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    if k is None:
        k = batch_size
    
    # For ICF: lower ICF = more common = higher relevance
    # So we invert: relevance = 1 - ICF
    relevance = 1.0 - targets
    
    # Sort predictions descending (rare first)
    pred_sorted, pred_indices = torch.sort(predictions, descending=True)
    
    # Get top-k relevance scores (sorted by predictions)
    top_k_indices = pred_indices[:k]
    top_k_relevance = relevance[top_k_indices]
    
    # Compute DCG
    positions = torch.arange(1, len(top_k_relevance) + 1, dtype=torch.float32, device=predictions.device)
    discounts = torch.log2(positions + 1)
    dcg = (top_k_relevance / discounts).sum()
    
    # Ideal DCG (perfect ranking)
    ideal_relevance, _ = torch.sort(relevance, descending=True)
    ideal_dcg = (ideal_relevance[:k] / discounts[:len(ideal_relevance[:k])]).sum()
    
    # NDCG = DCG / IDCG
    if ideal_dcg > 1e-8:
        ndcg = dcg / ideal_dcg
        loss = 1.0 - ndcg  # Minimize (1 - NDCG)
    else:
        loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    return loss


# ============================================================================
# Listwise Ranking Losses (consolidated from loss_listwise.py)
# ============================================================================

def lambdarank_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    LambdaRank loss: gradient-based ranking loss.
    
    Based on: "Learning to Rank with Nonsmooth Cost Functions" (Burges et al., 2006)
    
    Args:
        predictions: [Batch] predicted scores
        targets: [Batch] relevance scores (higher = more relevant)
        sigma: Smoothing parameter
    
    Returns:
        Scalar loss value
    """
    batch_size = predictions.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    # Flatten
    if predictions.dim() > 1:
        predictions = predictions.squeeze(1)
    if targets.dim() > 1:
        targets = targets.squeeze(1)
    
    # For ICF: lower ICF = more common = higher relevance
    # So we invert: relevance = 1 - ICF
    relevance = 1.0 - targets
    
    # Compute pairwise differences
    pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # [Batch, Batch]
    rel_diff = relevance.unsqueeze(1) - relevance.unsqueeze(0)  # [Batch, Batch]
    
    # LambdaRank: lambda_ij = |delta_NDCG| * sigmoid(-sigma * (pred_i - pred_j))
    # where delta_NDCG is the change in NDCG if i and j are swapped
    
    # Simplified: use relevance difference as proxy for delta_NDCG
    lambda_ij = torch.abs(rel_diff) * torch.sigmoid(-sigma * pred_diff)
    
    # Loss = -sum(lambda_ij * pred_diff)
    loss = -torch.sum(lambda_ij * pred_diff)
    
    return loss / (batch_size * (batch_size - 1))


def approx_ndcg_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Approximate NDCG loss: differentiable approximation of NDCG.
    
    Uses softmax to approximate ranking, making it differentiable.
    More stable than LambdaRank for small batches.
    
    Args:
        predictions: [Batch] or [Batch, 1] model predictions
        targets: [Batch] or [Batch, 1] ground truth ICF scores
        temperature: Temperature for softmax ranking (default: 1.0)
    
    Returns:
        Scalar loss value (1 - NDCG approximation)
    """
    # Flatten to [Batch]
    if predictions.dim() > 1:
        predictions = predictions.squeeze(1)
    if targets.dim() > 1:
        targets = targets.squeeze(1)
    
    batch_size = len(predictions)
    if batch_size < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    # For ICF: rank by descending (rare first) for NDCG
    # Higher target = rarer = should be ranked higher
    
    # Compute gains (higher target = higher gain)
    gains = torch.pow(2.0, targets) - 1.0
    
    # Position discounts (rank 1 gets discount[0], rank 2 gets discount[1], etc.)
    ranks = torch.arange(1, batch_size + 1, dtype=torch.float32, device=predictions.device)
    discounts = 1.0 / torch.log2(ranks + 1.0)
    
    # Soft ranking: use softmax to get importance weights
    # Higher predictions get higher weights, approximating ranking
    soft_weights = F.softmax(predictions / temperature, dim=0)
    
    # Sort by predictions to get ranking order
    pred_sorted, pred_indices = torch.sort(predictions, descending=True)
    
    # Approximate DCG: weight gains by softmax probabilities and position discounts
    sorted_gains = gains[pred_indices]
    sorted_weights = soft_weights[pred_indices]
    approx_dcg = torch.sum(sorted_gains * sorted_weights * discounts[:len(sorted_gains)])
    
    # Ideal DCG (sorted by targets descending)
    ideal_indices = torch.argsort(targets, descending=True)
    ideal_dcg = torch.sum(gains[ideal_indices] * discounts)
    
    # Approximate NDCG
    if ideal_dcg > 0:
        approx_ndcg = approx_dcg / ideal_dcg
        loss = 1.0 - approx_ndcg  # Loss = 1 - NDCG
    else:
        loss = torch.tensor(1.0, device=predictions.device, requires_grad=True)
    
    return loss


# ============================================================================
# Spearman Correlation Loss (consolidated from loss_spearman.py)
# ============================================================================

def spearman_loss_simple(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    regularization_strength: float = 0.1,
) -> torch.Tensor:
    """
    Simple differentiable Spearman correlation loss using soft ranking.
    
    Uses argsort-based soft ranking approximation for differentiability.
    This is a lightweight alternative to fast-soft-sort that works well in practice.
    
    Args:
        predictions: [Batch] or [Batch, 1] model predictions
        targets: [Batch] or [Batch, 1] ground truth ICF scores
        regularization_strength: Temperature for soft ranking (higher = sharper)
    
    Returns:
        Loss value (1 - Spearman correlation, so lower is better)
    """
    # Flatten to [Batch]
    if predictions.dim() > 1:
        predictions = predictions.squeeze(1)
    if targets.dim() > 1:
        targets = targets.squeeze(1)
    
    batch_size = len(predictions)
    if batch_size < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    # Soft ranking: use temperature-scaled softmax to approximate ranking
    # Higher temperature = sharper ranking, lower = smoother (more differentiable)
    pred_sorted, pred_indices = torch.sort(predictions, descending=False)
    target_sorted, target_indices = torch.sort(targets, descending=False)
    
    # Create soft ranks using temperature-scaled distances
    # For each element, compute its "soft rank" based on how many elements it's greater than
    pred_ranks = torch.zeros_like(predictions)
    target_ranks = torch.zeros_like(targets)
    
    for i in range(batch_size):
        # Soft rank = sum of sigmoid((pred[i] - pred[j]) * temperature) for all j
        # This approximates how many elements pred[i] is greater than
        pred_diff = predictions[i] - predictions
        pred_ranks[i] = torch.sigmoid(pred_diff * regularization_strength).sum()
        
        target_diff = targets[i] - targets
        target_ranks[i] = torch.sigmoid(target_diff * regularization_strength).sum()
    
    # Normalize ranks to [0, 1] range
    pred_ranks = pred_ranks / (batch_size - 1) if batch_size > 1 else pred_ranks
    target_ranks = target_ranks / (batch_size - 1) if batch_size > 1 else target_ranks
    
    # Compute Spearman correlation (Pearson correlation of ranks)
    pred_ranks_centered = pred_ranks - pred_ranks.mean()
    target_ranks_centered = target_ranks - target_ranks.mean()
    
    numerator = (pred_ranks_centered * target_ranks_centered).sum()
    pred_std = torch.sqrt((pred_ranks_centered ** 2).sum() + 1e-8)
    target_std = torch.sqrt((target_ranks_centered ** 2).sum() + 1e-8)
    
    spearman = numerator / (pred_std * target_std + 1e-8)
    
    # Loss = 1 - Spearman (so lower is better, Spearman higher is better)
    loss = 1.0 - spearman
    
    return loss


def spearman_loss_vectorized(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    regularization_strength: float = 0.1,
) -> torch.Tensor:
    """
    Vectorized differentiable Spearman correlation loss.
    
    More efficient than simple version, uses broadcasting.
    
    Args:
        predictions: [Batch] or [Batch, 1] model predictions
        targets: [Batch] or [Batch, 1] ground truth ICF scores
        regularization_strength: Temperature for soft ranking
    
    Returns:
        Loss value (1 - Spearman correlation)
    """
    # Flatten to [Batch]
    if predictions.dim() > 1:
        predictions = predictions.squeeze(1)
    if targets.dim() > 1:
        targets = targets.squeeze(1)
    
    batch_size = len(predictions)
    if batch_size < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    # Vectorized soft ranking: [Batch, Batch] pairwise differences
    pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # [Batch, Batch]
    target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # [Batch, Batch]
    
    # Soft ranks: for each element, count how many others it's greater than
    pred_ranks = torch.sigmoid(pred_diff * regularization_strength).sum(dim=1)  # [Batch]
    target_ranks = torch.sigmoid(target_diff * regularization_strength).sum(dim=1)  # [Batch]
    
    # Normalize to [0, 1] range
    pred_ranks = pred_ranks / (batch_size - 1) if batch_size > 1 else pred_ranks
    target_ranks = target_ranks / (batch_size - 1) if batch_size > 1 else target_ranks
    
    # Compute Spearman correlation
    pred_ranks_centered = pred_ranks - pred_ranks.mean()
    target_ranks_centered = target_ranks - target_ranks.mean()
    
    numerator = (pred_ranks_centered * target_ranks_centered).sum()
    pred_std = torch.sqrt((pred_ranks_centered ** 2).sum() + 1e-8)
    target_std = torch.sqrt((target_ranks_centered ** 2).sum() + 1e-8)
    
    spearman = numerator / (pred_std * target_std + 1e-8)
    
    # Loss = 1 - Spearman
    loss = 1.0 - spearman
    
    return loss


def _try_import_torchsort():
    """Try to import torchsort (PyTorch implementation of fast-soft-sort)."""
    try:
        import torchsort
        return torchsort
    except ImportError:
        return None


def _try_import_diffsort():
    """Try to import diffsort."""
    try:
        from diffsort import DiffSortNet
        return DiffSortNet
    except ImportError:
        return None


def spearman_loss_torchsort(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    regularization_strength: float = 1.0,
) -> torch.Tensor:
    """
    Spearman loss using torchsort (fast-soft-sort).
    
    O(n log n) complexity, compiled kernels, best performance.
    
    Args:
        predictions: [Batch] or [Batch, 1] model predictions
        targets: [Batch] or [Batch, 1] ground truth ICF scores
        regularization_strength: Temperature for soft ranking (higher = sharper)
    
    Returns:
        Loss value (1 - Spearman correlation)
    """
    torchsort = _try_import_torchsort()
    if torchsort is None:
        raise ImportError("torchsort not installed. Install with: pip install torchsort")
    
    # Flatten to [Batch]
    if predictions.dim() > 1:
        predictions = predictions.squeeze(1)
    if targets.dim() > 1:
        targets = targets.squeeze(1)
    
    batch_size = len(predictions)
    if batch_size < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    # Compute soft ranks using torchsort (O(n log n))
    pred_ranks = torchsort.soft_rank(predictions.unsqueeze(0), regularization_strength=regularization_strength)
    target_ranks = torchsort.soft_rank(targets.unsqueeze(0), regularization_strength=regularization_strength)
    
    # Remove batch dimension
    pred_ranks = pred_ranks.squeeze(0)
    target_ranks = target_ranks.squeeze(0)
    
    # Compute Spearman correlation (Pearson correlation of ranks)
    pred_ranks_centered = pred_ranks - pred_ranks.mean()
    target_ranks_centered = target_ranks - target_ranks.mean()
    
    numerator = (pred_ranks_centered * target_ranks_centered).sum()
    pred_std = torch.sqrt((pred_ranks_centered ** 2).sum() + 1e-8)
    target_std = torch.sqrt((target_ranks_centered ** 2).sum() + 1e-8)
    
    spearman = numerator / (pred_std * target_std + 1e-8)
    
    # Loss = 1 - Spearman
    loss = 1.0 - spearman
    
    return loss


def spearman_loss_diffsort(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    steepness: float = 5.0,
) -> torch.Tensor:
    """
    Spearman loss using diffsort (differentiable sorting networks).
    
    O(n²(log n)²) complexity, more structured but slower than torchsort.
    
    Args:
        predictions: [Batch] or [Batch, 1] model predictions
        targets: [Batch] or [Batch, 1] ground truth ICF scores
        steepness: Steepness parameter (higher = sharper sorting)
    
    Returns:
        Loss value (1 - Spearman correlation)
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
    
    # Pad to next power of 2 for bitonic sort
    next_power = 2 ** (batch_size - 1).bit_length()
    if next_power > batch_size:
        # Pad with minimum values
        pad_size = next_power - batch_size
        min_pred = predictions.min() - 1.0
        min_target = targets.min() - 1.0
        pred_padded = torch.cat([predictions, torch.full((pad_size,), min_pred, device=predictions.device)])
        target_padded = torch.cat([targets, torch.full((pad_size,), min_target, device=targets.device)])
    else:
        pred_padded = predictions
        target_padded = targets
    
    # Create sorters
    pred_sorter = DiffSortNet('bitonic', next_power, steepness=steepness)
    target_sorter = DiffSortNet('bitonic', next_power, steepness=steepness)
    
    # Get sorted values and permutation matrices
    pred_sorted, pred_perm = pred_sorter(pred_padded.unsqueeze(0))
    target_sorted, target_perm = target_sorter(target_padded.unsqueeze(0))
    
    # Extract ranks from permutation matrices (sum of columns gives rank)
    pred_ranks = pred_perm.squeeze(0).sum(dim=0)[:batch_size] + 1.0
    target_ranks = target_perm.squeeze(0).sum(dim=0)[:batch_size] + 1.0
    
    # Compute Spearman correlation
    pred_ranks_centered = pred_ranks - pred_ranks.mean()
    target_ranks_centered = target_ranks - target_ranks.mean()
    
    numerator = (pred_ranks_centered * target_ranks_centered).sum()
    pred_std = torch.sqrt((pred_ranks_centered ** 2).sum() + 1e-8)
    target_std = torch.sqrt((target_ranks_centered ** 2).sum() + 1e-8)
    
    spearman = numerator / (pred_std * target_std + 1e-8)
    
    # Loss = 1 - Spearman
    loss = 1.0 - spearman
    
    return loss


class SpearmanLoss(nn.Module):
    """
    Differentiable Spearman correlation loss module.
    
    Directly optimizes Spearman correlation, which is our primary metric.
    
    Supports multiple backends:
    - 'rank-relax' (preferred): Optimized Rust implementation with analytical gradients
    - 'torchsort': O(n log n), compiled kernels, best performance
    - 'diffsort': O(n²(log n)²), more structured but slower
    - 'built-in': O(n²), simple sigmoid-based, no dependencies (fallback)
    """
    
    def __init__(
        self,
        regularization_strength: float = 0.1,
        use_vectorized: bool = True,
        backend: Literal['auto', 'rank-relax', 'torchsort', 'diffsort', 'built-in'] = 'auto',
        steepness: float = 5.0,  # For diffsort
    ):
        super().__init__()
        self.regularization_strength = regularization_strength
        self.use_vectorized = use_vectorized
        self.steepness = steepness
        
        # Determine backend
        if backend == 'auto':
            # Try torchsort first (best), then diffsort, then built-in
            if _try_import_torchsort() is not None:
                self.backend = 'torchsort'
            elif _try_import_diffsort() is not None:
                self.backend = 'diffsort'
            else:
                self.backend = 'built-in'
        else:
            self.backend = backend
        
        # Validate backend availability
        if self.backend == 'torchsort' and _try_import_torchsort() is None:
            raise ImportError("torchsort not installed. Install with: pip install torchsort")
        if self.backend == 'diffsort' and _try_import_diffsort() is None:
            raise ImportError("diffsort not installed. Install with: pip install diffsort")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Spearman correlation loss.
        
        Args:
            predictions: [Batch] or [Batch, 1] model predictions
            targets: [Batch] or [Batch, 1] ground truth ICF scores
        
        Returns:
            Loss value (1 - Spearman correlation)
        """
        if self.backend == 'rank-relax':
            # Use optimized rank-relax implementation with analytical gradients
            return rank_relax_spearman_loss(
                predictions, targets, self.regularization_strength
            )
        elif self.backend == 'torchsort':
            return spearman_loss_torchsort(
                predictions, targets, self.regularization_strength
            )
        elif self.backend == 'diffsort':
            return spearman_loss_diffsort(
                predictions, targets, self.steepness
            )
        else:  # built-in
            if self.use_vectorized:
                return spearman_loss_vectorized(
                    predictions, targets, self.regularization_strength
                )
            else:
                return spearman_loss_simple(
                    predictions, targets, self.regularization_strength
                )
    
    def get_backend_info(self) -> dict:
        """Get information about available backends."""
        return {
            'current': self.backend,
            'rank_relax_available': HAS_RANK_RELAX,
            'torchsort_available': _try_import_torchsort() is not None,
            'diffsort_available': _try_import_diffsort() is not None,
            'built-in_available': True,
        }


class CombinedLoss(nn.Module):
    """
    Combined Huber + Ranking loss + optional NeuralNDCG + optional Listwise Ranking + optional Spearman.
    
    Enhanced with loss component tracking and diagnostics.
    """
    
    def __init__(
        self,
        huber_delta: float = 0.1,
        rank_margin: float = 0.1,
        rank_weight: float = 0.5,  # Reduced from 2.0 to balance with regression
        use_neural_ndcg: bool = False,
        neural_ndcg_weight: float = 0.5,
        use_listwise_ranking: bool = False,
        listwise_method: str = "lambdarank",
        listwise_weight: float = 1.0,
        listwise_sigma: float = 1.0,
        listwise_temperature: float = 1.0,
        use_spearman: bool = False,
        spearman_weight: float = 1.0,
        spearman_reg_strength: float = 0.1,
        track_components: bool = True,
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.rank_margin = rank_margin  # Increased from 0.05 to 0.1 for better separation
        self.rank_weight = rank_weight  # Increased from 1.0 to 2.0 for stronger ranking signal
        self.use_neural_ndcg = use_neural_ndcg
        self.neural_ndcg_weight = neural_ndcg_weight
        self.use_listwise_ranking = use_listwise_ranking
        self.listwise_method = listwise_method
        self.listwise_weight = listwise_weight
        self.listwise_sigma = listwise_sigma
        self.listwise_temperature = listwise_temperature
        self.use_spearman = use_spearman
        self.spearman_weight = spearman_weight
        self.spearman_reg_strength = spearman_reg_strength
        self.track_components = track_components
        
        # Initialize Spearman loss if needed (now in same module)
        if use_spearman:
            try:
                # Use 'auto' backend: rank-relax (best) -> torchsort -> diffsort -> built-in (fallback)
                self.spearman_loss = SpearmanLoss(
                    regularization_strength=spearman_reg_strength,
                    use_vectorized=True,
                    backend='auto',  # Automatically selects best available backend
                )
                # Log which backend was selected
                backend_info = self.spearman_loss.get_backend_info()
                if backend_info['current'] == 'rank-relax':
                    print(f"✅ Spearman loss using rank-relax (optimized Rust, analytical gradients)")
                elif backend_info['current'] == 'torchsort':
                    print(f"✅ Spearman loss using torchsort (O(n log n), best performance)")
                elif backend_info['current'] == 'diffsort':
                    print(f"✅ Spearman loss using diffsort (O(n²(log n)²))")
                else:
                    print(f"✅ Spearman loss using built-in (O(n²), no dependencies)")
            except Exception as e:
                print(f"Warning: Spearman loss initialization failed ({e}), disabling")
                self.use_spearman = False
        
        # Track loss components for monitoring
        if track_components:
            self.register_buffer('huber_history', torch.zeros(100))
            self.register_buffer('ranking_history', torch.zeros(100))
            self.register_buffer('neural_ndcg_history', torch.zeros(100))
            self.register_buffer('listwise_history', torch.zeros(100))
            self.history_idx = 0
    
    def huber_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss component."""
        return huber_loss(predictions, targets, delta=self.huber_delta)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pairs: Optional[torch.Tensor] = None,
        pair_target_diffs: Optional[torch.Tensor] = None,
        smooth_ranking: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            predictions: [Batch, 1] model predictions
            targets: [Batch, 1] ground truth ICF scores
            pairs: Optional [N_pairs, 2] indices of word pairs for ranking loss
            pair_target_diffs: Optional [N_pairs] actual ICF differences for weighted loss
            smooth_ranking: If True, use smooth sigmoid-based ranking loss
        
        Returns:
            Total loss tensor
        """
        # Ensure 1D for NeuralNDCG and Listwise losses
        pred_1d = predictions.squeeze() if predictions.dim() > 1 else predictions
        target_1d = targets.squeeze() if targets.dim() > 1 else targets
        
        # Huber loss on individual predictions
        huber = huber_loss(predictions, targets, delta=self.huber_delta)
        
        total_loss = huber
        
        # Track components if enabled
        if self.track_components:
            idx = self.history_idx % 100
            self.huber_history[idx] = huber.detach()
        
        # NeuralNDCG loss (if enabled)
        if self.use_neural_ndcg:
            ndcg_loss = neural_ndcg_loss_simple(pred_1d, target_1d)
            total_loss = total_loss + self.neural_ndcg_weight * ndcg_loss
            if self.track_components:
                self.neural_ndcg_history[idx] = ndcg_loss.detach()
        
        # Listwise Ranking loss (if enabled)
        if self.use_listwise_ranking:
            if self.listwise_method == "lambdarank":
                listwise_loss = lambdarank_loss(pred_1d, target_1d, sigma=self.listwise_sigma)
            elif self.listwise_method == "approx_ndcg":
                listwise_loss = approx_ndcg_loss(pred_1d, target_1d, temperature=self.listwise_temperature)
            else:
                raise ValueError(f"Unknown listwise method: {self.listwise_method}")
            total_loss = total_loss + self.listwise_weight * listwise_loss
            if self.track_components:
                self.listwise_history[idx] = listwise_loss.detach()
        
        # Ranking loss on pairs (if provided)
        if pairs is not None and len(pairs) > 0:
            idx1, idx2 = pairs[:, 0], pairs[:, 1]
            pred1 = predictions[idx1]
            pred2 = predictions[idx2]
            rank = ranking_loss(
                pred1, pred2, 
                margin=self.rank_margin,
                target_diff=pair_target_diffs,
                smooth=smooth_ranking,
            )
            total_loss = total_loss + self.rank_weight * rank
            if self.track_components:
                self.ranking_history[idx] = rank.detach()
        
        # Direct Spearman correlation loss (if enabled)
        if self.use_spearman:
            spearman_loss_val = self.spearman_loss(pred_1d, target_1d)
            total_loss = total_loss + self.spearman_weight * spearman_loss_val
            if self.track_components:
                # Store in neural_ndcg_history slot if not used, or create new buffer
                if not self.use_neural_ndcg:
                    self.neural_ndcg_history[idx] = spearman_loss_val.detach()
        
        if self.track_components:
            self.history_idx += 1
        
        return total_loss
    
    def get_component_stats(self) -> Dict[str, float]:
        """
        Get statistics about loss components.
        
        Returns:
            Dictionary with mean, std, and ratios for each component
        """
        if not self.track_components:
            return {}
        
        stats = {}
        
        # Huber stats
        valid_huber = self.huber_history[self.huber_history > 0]
        if len(valid_huber) > 0:
            stats['huber_mean'] = valid_huber.mean().item()
            if len(valid_huber) > 1:
                stats['huber_std'] = valid_huber.std().item()
            else:
                stats['huber_std'] = 0.0
        
        # Ranking stats
        valid_ranking = self.ranking_history[self.ranking_history > 0]
        if len(valid_ranking) > 0:
            stats['ranking_mean'] = valid_ranking.mean().item()
            if len(valid_ranking) > 1:
                stats['ranking_std'] = valid_ranking.std().item()
            else:
                stats['ranking_std'] = 0.0
        
        # NeuralNDCG stats
        if self.use_neural_ndcg:
            valid_ndcg = self.neural_ndcg_history[self.neural_ndcg_history > 0]
            if len(valid_ndcg) > 0:
                stats['neural_ndcg_mean'] = valid_ndcg.mean().item()
                if len(valid_ndcg) > 1:
                    stats['neural_ndcg_std'] = valid_ndcg.std().item()
                else:
                    stats['neural_ndcg_std'] = 0.0
        
        # Listwise stats
        if self.use_listwise_ranking:
            valid_listwise = self.listwise_history[self.listwise_history > 0]
            if len(valid_listwise) > 0:
                stats['listwise_mean'] = valid_listwise.mean().item()
                if len(valid_listwise) > 1:
                    stats['listwise_std'] = valid_listwise.std().item()
                else:
                    stats['listwise_std'] = 0.0
        
        # Spearman stats (stored in neural_ndcg_history if neural_ndcg not used)
        if self.use_spearman and not self.use_neural_ndcg:
            valid_spearman = self.neural_ndcg_history[self.neural_ndcg_history > 0]
            if len(valid_spearman) > 0:
                stats['spearman_mean'] = valid_spearman.mean().item()
                if len(valid_spearman) > 1:
                    stats['spearman_std'] = valid_spearman.std().item()
                else:
                    stats['spearman_std'] = 0.0
        
        # Compute ratios (approximate)
        total_mean = sum([
            stats.get('huber_mean', 0),
            stats.get('ranking_mean', 0) * self.rank_weight,
            stats.get('neural_ndcg_mean', 0) * (self.neural_ndcg_weight if self.use_neural_ndcg else 0),
            stats.get('listwise_mean', 0) * (self.listwise_weight if self.use_listwise_ranking else 0),
            stats.get('spearman_mean', 0) * (self.spearman_weight if self.use_spearman else 0),
        ])
        
        if total_mean > 0:
            stats['huber_ratio'] = stats.get('huber_mean', 0) / total_mean
            stats['ranking_ratio'] = (stats.get('ranking_mean', 0) * self.rank_weight) / total_mean
            if self.use_neural_ndcg:
                stats['neural_ndcg_ratio'] = (stats.get('neural_ndcg_mean', 0) * self.neural_ndcg_weight) / total_mean
            if self.use_spearman:
                spearman_mean = stats.get('spearman_mean', 0)
                if spearman_mean > 0:
                    stats['spearman_ratio'] = (spearman_mean * self.spearman_weight) / total_mean
            if self.use_listwise_ranking:
                stats['listwise_ratio'] = (stats.get('listwise_mean', 0) * self.listwise_weight) / total_mean
        
        return stats

