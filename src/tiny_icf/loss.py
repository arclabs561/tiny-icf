"""Loss functions for training: Huber + Ranking loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    pred1: torch.Tensor, pred2: torch.Tensor, margin: float = 0.1
) -> torch.Tensor:
    """
    Pairwise ranking loss.
    
    Enforces that if word1 is more common than word2,
    then pred1 < pred2 (lower ICF = more common).
    
    Args:
        pred1: Predictions for more common words
        pred2: Predictions for less common words
        margin: Minimum margin between predictions
    """
    # We want pred1 < pred2 (common < rare in ICF space)
    # Loss = max(0, margin - (pred2 - pred1))
    diff = pred2 - pred1
    loss = F.relu(margin - diff)
    return loss.mean()


class CombinedLoss(nn.Module):
    """Combined Huber + Ranking loss."""
    
    def __init__(self, huber_delta: float = 0.1, rank_margin: float = 0.05, rank_weight: float = 1.0):
        super().__init__()
        self.huber_delta = huber_delta
        self.rank_margin = rank_margin
        self.rank_weight = rank_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pairs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            predictions: [Batch, 1] model predictions
            targets: [Batch, 1] ground truth ICF scores
            pairs: Optional [N_pairs, 2] indices of word pairs for ranking loss
        """
        # Huber loss on individual predictions
        huber = huber_loss(predictions, targets, delta=self.huber_delta)
        
        # Ranking loss on pairs (if provided)
        if pairs is not None and len(pairs) > 0:
            idx1, idx2 = pairs[:, 0], pairs[:, 1]
            pred1 = predictions[idx1]
            pred2 = predictions[idx2]
            rank = ranking_loss(pred1, pred2, margin=self.rank_margin)
            return huber + self.rank_weight * rank
        
        return huber

