"""Listwise ranking losses for word frequency prediction.

Based on research: pairwise ranking losses may not be effective for Spearman correlation.
Listwise losses (LambdaRank, ApproxNDCG) directly optimize ranking quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def lambdarank_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    LambdaRank loss: listwise ranking loss that directly optimizes NDCG.
    
    Simplified implementation: computes pairwise lambdas based on NDCG change.
    For efficiency, we compute lambdas for all pairs and aggregate.
    
    Args:
        predictions: [Batch] or [Batch, 1] model predictions
        targets: [Batch] or [Batch, 1] ground truth ICF scores
        sigma: Smoothing parameter for sigmoid (default: 1.0)
    
    Returns:
        Scalar loss value
    """
    # Flatten to [Batch]
    if predictions.dim() > 1:
        predictions = predictions.squeeze(1)
    if targets.dim() > 1:
        targets = targets.squeeze(1)
    
    batch_size = len(predictions)
    if batch_size < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    # For ICF: lower = more common, higher = rarer
    # We want to rank by ascending ICF (common first) for NDCG
    # But NDCG typically ranks by relevance (high first)
    # So we invert: rank by descending ICF (rare first) = rank by descending targets
    
    # Compute gains (higher target = rarer = higher gain)
    gains = torch.pow(2.0, targets) - 1.0
    
    # Simplified LambdaRank: compute pairwise lambdas
    # For each pair (i, j), if target[i] > target[j] (i is rarer):
    #   - We want pred[i] > pred[j]
    #   - Lambda[i] += |delta_NDCG| * sigmoid(-sigma * (pred[i] - pred[j]))
    #   - Lambda[j] -= |delta_NDCG| * sigmoid(-sigma * (pred[i] - pred[j]))
    
    lambdas = torch.zeros(batch_size, device=predictions.device, requires_grad=False)
    
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if targets[i] > targets[j]:  # i is rarer, should have higher pred
                # Delta NDCG approximation: difference in gains
                delta_ndcg = torch.abs(gains[i] - gains[j])
                
                # Lambda: gradient signal
                # If pred[i] < pred[j] (wrong), sigmoid gives high value
                lambda_ij = delta_ndcg * F.sigmoid(-sigma * (predictions[i] - predictions[j]))
                lambdas[i] += lambda_ij
                lambdas[j] -= lambda_ij
            elif targets[j] > targets[i]:  # j is rarer
                delta_ndcg = torch.abs(gains[j] - gains[i])
                lambda_ij = delta_ndcg * F.sigmoid(-sigma * (predictions[j] - predictions[i]))
                lambdas[j] += lambda_ij
                lambdas[i] -= lambda_ij
    
    # Loss: negative sum of lambdas * predictions (maximize NDCG = minimize negative)
    loss = -torch.sum(lambdas * predictions)
    
    return loss


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
    
    # Soft ranking: probability of each item being at each rank
    # Use predictions directly (higher pred = should be ranked higher)
    # Softmax over predictions gives probability distribution over ranks
    pred_sorted, pred_indices = torch.sort(predictions, descending=True)
    
    # Approximate DCG: for each item, compute expected DCG contribution
    # Item i contributes gain[i] * expected_discount[i]
    # Expected discount = sum over ranks: P(rank) * discount[rank]
    # Simplified: use softmax to get rank probabilities
    soft_rank_probs = F.softmax(predictions / temperature, dim=0)
    
    # Approximate DCG: each item's gain weighted by its expected discount
    # Expected discount for item at position k = discount[k] * soft_rank_prob[k]
    approx_dcg = torch.sum(gains[pred_indices] * discounts * soft_rank_probs[pred_indices])
    
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


class ListwiseRankingLoss(nn.Module):
    """
    Listwise ranking loss combining LambdaRank and ApproxNDCG.
    
    Based on research: listwise losses directly optimize ranking metrics
    like NDCG, providing better signal than pairwise ranking for Spearman correlation.
    """
    
    def __init__(
        self,
        method: str = "lambdarank",
        sigma: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Args:
            method: "lambdarank" or "approx_ndcg"
            sigma: Smoothing parameter for LambdaRank
            temperature: Temperature for ApproxNDCG
        """
        super().__init__()
        self.method = method
        self.sigma = sigma
        self.temperature = temperature
    
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
        if self.method == "lambdarank":
            return lambdarank_loss(predictions, targets, sigma=self.sigma)
        elif self.method == "approx_ndcg":
            return approx_ndcg_loss(predictions, targets, temperature=self.temperature)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class CombinedListwiseLoss(nn.Module):
    """
    Combined Huber + Listwise ranking loss.
    
    Huber loss for absolute accuracy, listwise loss for ranking quality.
    """
    
    def __init__(
        self,
        huber_delta: float = 0.1,
        listwise_weight: float = 1.0,
        listwise_method: str = "lambdarank",
        listwise_sigma: float = 1.0,
        listwise_temperature: float = 1.0,
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.listwise_weight = listwise_weight
        self.listwise = ListwiseRankingLoss(
            method=listwise_method,
            sigma=listwise_sigma,
            temperature=listwise_temperature,
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: [Batch, 1] model predictions
            targets: [Batch, 1] ground truth ICF scores
        
        Returns:
            Scalar loss value
        """
        # Ensure same shape for Huber loss
        if predictions.dim() > 1:
            predictions_flat = predictions.squeeze(1)
        else:
            predictions_flat = predictions
        
        if targets.dim() > 1:
            targets_flat = targets.squeeze(1)
        else:
            targets_flat = targets
        
        # Huber loss for absolute accuracy
        huber = F.smooth_l1_loss(predictions_flat, targets_flat, reduction='mean', beta=self.huber_delta)
        
        # Listwise ranking loss for ranking quality
        listwise = self.listwise(predictions, targets)
        
        # Combine
        total_loss = huber + self.listwise_weight * listwise
        
        return total_loss

