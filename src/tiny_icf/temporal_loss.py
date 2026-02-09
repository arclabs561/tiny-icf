"""Temporal loss functions for multi-objective optimization with historical data."""

import torch
import torch.nn as nn
from typing import Dict, Optional, List


def temporal_icf_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    temporal_targets: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Loss that encourages predictions to match historical ICF patterns.

    Args:
        predictions: Model predictions [Batch, 1]
        targets: Current ICF targets [Batch, 1]
        temporal_targets: Dict mapping decade -> ICF targets [Batch, 1]
        alpha: Weight for temporal consistency

    Returns:
        Combined loss
    """
    # Base loss (Huber or MSE)
    base_loss = nn.functional.mse_loss(predictions, targets)

    if temporal_targets is None or len(temporal_targets) == 0:
        return base_loss

    # Temporal consistency: predictions should be consistent with historical trends
    temporal_loss = 0.0
    for decade, hist_targets in temporal_targets.items():
        # Encourage smooth transitions across decades
        temporal_loss += nn.functional.mse_loss(predictions, hist_targets)

    temporal_loss = temporal_loss / len(temporal_targets)

    return base_loss + alpha * temporal_loss


def multi_decade_icf_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Multi-objective loss for predicting ICF across multiple decades.

    Args:
        predictions: Dict mapping decade -> predictions [Batch, 1]
        targets: Dict mapping decade -> targets [Batch, 1]
        weights: Optional weights for each decade

    Returns:
        Weighted sum of losses across decades
    """
    if weights is None:
        # Equal weights
        weights = {decade: 1.0 / len(predictions) for decade in predictions.keys()}

    total_loss = 0.0
    for decade in predictions.keys():
        if decade not in targets:
            continue

        loss = nn.functional.mse_loss(predictions[decade], targets[decade])
        total_loss += weights.get(decade, 0.0) * loss

    return total_loss


class AlignedMultiObjectiveLoss(nn.Module):
    """
    Aligned Multi-Objective Optimization loss.

    Based on the AMOO framework where multiple objectives share a common solution.
    Uses adaptive weighting to exploit alignment.
    """

    def __init__(
        self,
        objectives: List[str],
        initial_weights: Optional[Dict[str, float]] = None,
        adaptive: bool = True,
        curvature_weight: float = 0.1,
    ):
        """
        Args:
            objectives: List of objective names (e.g., ['icf', 'temporal', 'language'])
            initial_weights: Initial weights for each objective
            adaptive: Whether to use adaptive weighting (CAMOO-like)
            curvature_weight: Weight for curvature-based adaptation
        """
        super().__init__()
        self.objectives = objectives
        self.adaptive = adaptive
        self.curvature_weight = curvature_weight

        if initial_weights is None:
            # Equal weights
            self.register_buffer("weights", torch.ones(len(objectives)) / len(objectives))
        else:
            weight_vec = torch.tensor(
                [initial_weights.get(obj, 1.0 / len(objectives)) for obj in objectives]
            )
            self.register_buffer("weights", weight_vec / weight_vec.sum())

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        gradients: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute weighted multi-objective loss.

        Args:
            losses: Dict mapping objective name -> loss tensor
            gradients: Optional dict of gradients for adaptive weighting

        Returns:
            Weighted sum of losses
        """
        # Collect loss values (ensure they're on the same device)
        device = self.weights.device
        loss_values = []
        for obj in self.objectives:
            loss_val = losses.get(obj, torch.tensor(0.0, device=device))
            if not isinstance(loss_val, torch.Tensor):
                loss_val = torch.tensor(float(loss_val), device=device)
            loss_values.append(loss_val)

        loss_values = torch.stack(loss_values)

        if self.adaptive and gradients is not None and len(gradients) > 0:
            # Adaptive weighting based on gradient alignment
            # Similar to CAMOO but simpler for our use case
            try:
                grad_vecs = []
                for obj in self.objectives:
                    grad = gradients.get(obj)
                    if grad is None:
                        # Use zero gradient if not available
                        grad = torch.zeros(1, device=device)
                    elif not isinstance(grad, torch.Tensor):
                        grad = torch.tensor([float(grad)], device=device)
                    grad_vecs.append(grad.flatten())

                if len(grad_vecs) > 0 and len(grad_vecs[0]) > 0:
                    # Pad to same length
                    max_len = max(len(g) for g in grad_vecs)
                    grad_vecs_padded = [
                        torch.cat([g, torch.zeros(max_len - len(g), device=device)])
                        for g in grad_vecs
                    ]
                    grad_vecs = torch.stack(grad_vecs_padded)

                    # Compute gradient alignment
                    grad_alignment = torch.cosine_similarity(
                        grad_vecs.unsqueeze(0), grad_vecs.unsqueeze(1), dim=2
                    )

                    # Increase weight for objectives with better alignment
                    alignment_scores = grad_alignment.mean(dim=1)
                    adaptive_weights = self.weights * (
                        1.0 + self.curvature_weight * alignment_scores
                    )
                    adaptive_weights = adaptive_weights / adaptive_weights.sum()

                    return (adaptive_weights * loss_values).sum()
            except Exception:
                # Fall back to fixed weights if adaptive fails
                pass

        # Fixed weights
        return (self.weights * loss_values).sum()

    def update_weights(self, new_weights: Dict[str, float]):
        """Update objective weights manually."""
        weight_vec = torch.tensor(
            [new_weights.get(obj, self.weights[i].item()) for i, obj in enumerate(self.objectives)]
        )
        self.weights.data = weight_vec / weight_vec.sum()


def compute_temporal_consistency_loss(
    predictions: torch.Tensor,
    historical_predictions: Dict[str, torch.Tensor],
    smoothness_weight: float = 0.1,
) -> torch.Tensor:
    """
    Encourage temporal smoothness in predictions across decades.

    Args:
        predictions: Current predictions [Batch, 1]
        historical_predictions: Dict mapping decade -> predictions [Batch, 1]
        smoothness_weight: Weight for smoothness penalty

    Returns:
        Smoothness loss
    """
    if len(historical_predictions) == 0:
        return torch.tensor(0.0, device=predictions.device)

    # Sort decades
    decades = sorted(historical_predictions.keys())

    # Compute differences between adjacent decades
    smoothness_loss = 0.0
    for i in range(len(decades) - 1):
        diff = historical_predictions[decades[i + 1]] - historical_predictions[decades[i]]
        smoothness_loss += torch.mean(diff**2)

    # Also encourage current predictions to be consistent with recent history
    if len(decades) > 0:
        recent = historical_predictions[decades[-1]]
        consistency_loss = torch.mean((predictions - recent) ** 2)
        smoothness_loss += consistency_loss

    return smoothness_weight * smoothness_loss
