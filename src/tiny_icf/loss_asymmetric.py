"""Asymmetric loss functions for ICF prediction.

Addresses the insight that some errors are much worse than others:
- Polar opposites (0.0 → 1.0) should be penalized MUCH more
- Large errors should be penalized exponentially more
- Error direction matters (common→rare worse than rare→common)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def asymmetric_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 0.1,
    asymmetry_factor: float = 2.0,
) -> torch.Tensor:
    """
    Asymmetric Huber loss that penalizes:
    - Large errors more heavily (exponential)
    - Common→rare direction more than rare→common
    
    Args:
        predictions: [batch, 1] or [batch] model predictions
        targets: [batch, 1] or [batch] ground truth ICF scores
        delta: Huber loss delta parameter
        asymmetry_factor: Multiplier for common→rare errors
    
    Returns:
        Asymmetric Huber loss value
    """
    # Ensure same shape
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    if targets.dim() > 1:
        targets = targets.squeeze()
    
    error = predictions - targets
    
    # Base Huber loss (symmetric)
    huber_base = F.smooth_l1_loss(predictions, targets, beta=delta)
    
    # Asymmetric penalty: common→rare is worse than rare→common
    # error > 0: predicted more rare (common word predicted as rare) → BAD
    # error < 0: predicted more common (rare word predicted as common) → less bad
    asymmetric_penalty = torch.where(
        error > 0,
        asymmetry_factor * F.relu(error),  # Common→rare: higher penalty
        F.relu(-error)  # Rare→common: lower penalty
    )
    
    return huber_base + asymmetric_penalty.mean()


def magnitude_weighted_ranking_loss(
    pred1: torch.Tensor,
    pred2: torch.Tensor,
    target1: torch.Tensor,
    target2: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Ranking loss weighted by ICF difference magnitude.
    
    Large ICF differences (common vs rare) are more important to get right.
    
    Args:
        pred1, pred2: Model predictions for word pair
        target1, target2: Ground truth ICF scores
        margin: Minimum margin between predictions
    
    Returns:
        Magnitude-weighted ranking loss
    """
    target_diff = target1 - target2
    pred_diff = pred1 - pred2
    
    # Weight by target difference: larger differences = more important
    # At least 1.0, scales with difference magnitude
    weight = 1.0 + torch.abs(target_diff)
    
    # Standard margin loss
    violation = F.relu(margin - pred_diff * torch.sign(target_diff))
    
    return (weight * violation).mean()


def focal_ranking_loss(
    pred1: torch.Tensor,
    pred2: torch.Tensor,
    target1: torch.Tensor,
    target2: torch.Tensor,
    margin: float = 0.1,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal ranking loss: focus on hard examples (large errors).
    
    Similar to focal loss for classification, but for ranking.
    Hard examples (large errors) get exponentially more weight.
    
    Args:
        pred1, pred2: Model predictions for word pair
        target1, target2: Ground truth ICF scores
        margin: Minimum margin between predictions
        gamma: Focal loss exponent (higher = more focus on hard examples)
    
    Returns:
        Focal ranking loss
    """
    target_diff = target1 - target2
    pred_diff = pred1 - pred2
    
    # Standard margin loss
    base_loss = F.relu(margin - pred_diff * torch.sign(target_diff))
    
    # Focal weighting: large errors get exponentially more weight
    error_magnitude = torch.abs(pred_diff - target_diff)
    focal_weight = (1.0 + error_magnitude) ** gamma
    
    return (focal_weight * base_loss).mean()


def direction_aware_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    common_penalty: float = 2.0,
    rare_penalty: float = 1.0,
    delta: float = 0.1,
) -> torch.Tensor:
    """
    Direction-aware loss with different penalties for different error directions.
    
    - Predicting common word as rare: HIGH penalty (common_penalty)
    - Predicting rare word as common: LOW penalty (rare_penalty)
    
    Args:
        predictions: [batch, 1] or [batch] model predictions
        targets: [batch, 1] or [batch] ground truth ICF scores
        common_penalty: Penalty multiplier for common→rare errors
        rare_penalty: Penalty multiplier for rare→common errors
        delta: Huber loss delta parameter
    
    Returns:
        Direction-aware loss value
    """
    # Ensure same shape
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    if targets.dim() > 1:
        targets = targets.squeeze()
    
    error = predictions - targets
    
    # Base Huber loss
    base_loss = F.smooth_l1_loss(predictions, targets, beta=delta)
    
    # Direction-aware penalty
    penalty = torch.where(
        error > 0,  # Predicted more rare (common → rare)
        common_penalty,  # Higher penalty
        rare_penalty  # Lower penalty
    )
    
    return (penalty * base_loss).mean()


class AsymmetricICFLoss(nn.Module):
    """
    Combined asymmetric loss that addresses multiple perspectives:
    1. Penalizes large errors exponentially
    2. Penalizes common→rare more than rare→common
    3. Weights ranking by ICF difference magnitude
    4. Uses focal weighting for hard examples
    
    Research-aligned: Incorporates findings on:
    - Focal loss for hard example mining (arXiv 2017)
    - Asymmetric penalties for error direction
    - Magnitude-weighted ranking
    
    This loss function recognizes that:
    - Polar opposites (0.0 → 1.0) are MUCH worse than slight errors
    - Large errors should be penalized exponentially more
    - Error direction matters (common→rare worse than rare→common)
    """
    
    def __init__(
        self,
        huber_delta: float = 0.1,
        asymmetry_factor: float = 2.0,  # Common→rare penalty multiplier
        focal_gamma: float = 2.0,  # Focal loss exponent (research: 2.0 is effective)
        magnitude_weight: bool = True,  # Weight ranking by ICF difference
        rank_margin: float = 0.1,
        rank_weight: float = 0.5,
        use_focal: bool = True,  # Use focal weighting for ranking
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.asymmetry_factor = asymmetry_factor
        self.focal_gamma = focal_gamma
        self.magnitude_weight = magnitude_weight
        self.rank_margin = rank_margin
        self.rank_weight = rank_weight
        self.use_focal = use_focal
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pairs: Optional[torch.Tensor] = None,
        pair_target_diffs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute asymmetric ICF loss.
        
        Args:
            predictions: [batch, 1] or [batch] model predictions
            targets: [batch, 1] or [batch] ground truth ICF scores
            pairs: Optional [n_pairs, 2] indices for pairwise ranking
            pair_target_diffs: Optional [n_pairs] actual ICF differences
        
        Returns:
            (total_loss, component_losses)
        """
        # Ensure 1D
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        # 1. Asymmetric Huber loss
        error = predictions - targets
        huber_base = F.smooth_l1_loss(predictions, targets, beta=self.huber_delta)
        
        # Asymmetric penalty
        asymmetric_penalty = torch.where(
            error > 0,  # Common → rare
            self.asymmetry_factor * F.relu(error),
            F.relu(-error)  # Rare → common (less penalty)
        )
        huber_loss = huber_base + asymmetric_penalty.mean()
        
        # 2. Ranking loss (if pairs provided)
        rank_loss = torch.tensor(0.0, device=predictions.device)
        if pairs is not None and len(pairs) > 0:
            idx1, idx2 = pairs[:, 0], pairs[:, 1]
            pred1 = predictions[idx1]
            pred2 = predictions[idx2]
            target1 = targets[idx1]
            target2 = targets[idx2]
            
            target_diff = target1 - target2
            pred_diff = pred1 - pred2
            
            # Base margin loss
            base_rank_loss = F.relu(self.rank_margin - pred_diff * torch.sign(target_diff))
            
            # Magnitude weighting
            if self.magnitude_weight:
                weight = 1.0 + torch.abs(target_diff)  # Larger differences = more important
                base_rank_loss = weight * base_rank_loss
            
            # Focal weighting for hard examples
            if self.use_focal:
                error_magnitude = torch.abs(pred_diff - target_diff)
                focal_weight = (1.0 + error_magnitude) ** self.focal_gamma
                base_rank_loss = focal_weight * base_rank_loss
            
            rank_loss = base_rank_loss.mean()
        
        total_loss = huber_loss + self.rank_weight * rank_loss
        
        components = {
            'huber': huber_loss,
            'asymmetric_penalty': asymmetric_penalty.mean(),
            'rank': rank_loss,
            'total': total_loss,
        }
        
        return total_loss, components

