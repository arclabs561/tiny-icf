# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
# ]
# ///
"""
Adaptive loss weighting strategies for multi-objective optimization.

Implements:
- Real-time loss normalization
- Gradient-based balancing (GradNorm-inspired)
- Uncertainty weighting
- Loss component monitoring
"""

from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn

from tiny_icf.loss import (
    huber_loss,
    ranking_loss,
    neural_ndcg_loss_simple,
    lambdarank_loss,
    approx_ndcg_loss,
)


class RealTimeNormalizedLoss(nn.Module):
    """
    Real-time loss normalization: weights losses by inverse of current magnitude.

    Simple but effective baseline: w_i = 1 / L_i
    Ensures all losses contribute equally regardless of scale.
    """

    def __init__(
        self,
        huber_delta: float = 0.1,
        rank_margin: float = 0.1,
        rank_weight: float = 2.0,
        use_neural_ndcg: bool = False,
        neural_ndcg_weight: float = 0.5,
        use_listwise_ranking: bool = False,
        listwise_method: str = "lambdarank",
        listwise_weight: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.rank_margin = rank_margin
        self.rank_weight = rank_weight
        self.use_neural_ndcg = use_neural_ndcg
        self.neural_ndcg_weight = neural_ndcg_weight
        self.use_listwise_ranking = use_listwise_ranking
        self.listwise_method = listwise_method
        self.listwise_weight = listwise_weight
        self.eps = eps

        # Track loss components for normalization
        self.register_buffer("huber_ema", torch.tensor(1.0))
        self.register_buffer("ranking_ema", torch.tensor(1.0))
        self.register_buffer("neural_ndcg_ema", torch.tensor(1.0))
        self.register_buffer("listwise_ema", torch.tensor(1.0))
        self.ema_decay = 0.99

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pairs: Optional[torch.Tensor] = None,
        pair_target_diffs: Optional[torch.Tensor] = None,
        smooth_ranking: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute normalized loss with real-time weighting.

        Returns:
            (total_loss, diagnostics) where diagnostics contains component info
        """
        pred_1d = predictions.squeeze() if predictions.dim() > 1 else predictions
        target_1d = targets.squeeze() if targets.dim() > 1 else targets

        diagnostics = {}

        # Compute individual losses
        huber = huber_loss(predictions, targets, delta=self.huber_delta)

        # Normalize by current magnitude (detached to prevent gradient flow)
        huber_normalized = huber / (huber.detach() + self.eps)

        # Update EMA
        self.huber_ema = self.ema_decay * self.huber_ema + (1 - self.ema_decay) * huber.detach()

        total_loss = huber_normalized
        diagnostics["huber"] = huber.item()
        diagnostics["huber_normalized"] = huber_normalized.item()

        # NeuralNDCG
        if self.use_neural_ndcg:
            ndcg_loss = neural_ndcg_loss_simple(pred_1d, target_1d)
            ndcg_normalized = ndcg_loss / (ndcg_loss.detach() + self.eps)
            self.neural_ndcg_ema = (
                self.ema_decay * self.neural_ndcg_ema + (1 - self.ema_decay) * ndcg_loss.detach()
            )
            total_loss = total_loss + self.neural_ndcg_weight * ndcg_normalized
            diagnostics["neural_ndcg"] = ndcg_loss.item()
            diagnostics["neural_ndcg_normalized"] = ndcg_normalized.item()

        # Listwise
        if self.use_listwise_ranking:
            if self.listwise_method == "lambdarank":
                listwise_loss = lambdarank_loss(pred_1d, target_1d, sigma=1.0)
            elif self.listwise_method == "approx_ndcg":
                listwise_loss = approx_ndcg_loss(pred_1d, target_1d, temperature=1.0)
            else:
                raise ValueError(f"Unknown listwise method: {self.listwise_method}")

            listwise_normalized = listwise_loss / (listwise_loss.detach() + self.eps)
            self.listwise_ema = (
                self.ema_decay * self.listwise_ema + (1 - self.ema_decay) * listwise_loss.detach()
            )
            total_loss = total_loss + self.listwise_weight * listwise_normalized
            diagnostics["listwise"] = listwise_loss.item()
            diagnostics["listwise_normalized"] = listwise_normalized.item()

        # Ranking loss on pairs
        if pairs is not None and len(pairs) > 0:
            idx1, idx2 = pairs[:, 0], pairs[:, 1]
            pred1 = predictions[idx1]
            pred2 = predictions[idx2]
            rank = ranking_loss(
                pred1,
                pred2,
                margin=self.rank_margin,
                target_diff=pair_target_diffs,
                smooth=smooth_ranking,
            )
            rank_normalized = rank / (rank.detach() + self.eps)
            self.ranking_ema = (
                self.ema_decay * self.ranking_ema + (1 - self.ema_decay) * rank.detach()
            )
            total_loss = total_loss + self.rank_weight * rank_normalized
            diagnostics["ranking"] = rank.item()
            diagnostics["ranking_normalized"] = rank_normalized.item()

        diagnostics["total"] = total_loss.item()
        diagnostics["huber_ratio"] = (huber.detach() / (self.huber_ema + self.eps)).item()

        return total_loss, diagnostics


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty weighting: learns task uncertainty as parameters.

    Formula: L = Σ(1/(2σ²) * L_i + log(σ))
    Higher uncertainty = higher weight (counterintuitive but correct)
    """

    def __init__(
        self,
        huber_delta: float = 0.1,
        rank_margin: float = 0.1,
        rank_weight: float = 2.0,
        use_neural_ndcg: bool = False,
        neural_ndcg_weight: float = 0.5,
        use_listwise_ranking: bool = False,
        listwise_method: str = "lambdarank",
        listwise_weight: float = 1.0,
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.rank_margin = rank_margin
        self.rank_weight = rank_weight
        self.use_neural_ndcg = use_neural_ndcg
        self.neural_ndcg_weight = neural_ndcg_weight
        self.use_listwise_ranking = use_listwise_ranking
        self.listwise_method = listwise_method
        self.listwise_weight = listwise_weight

        # Learnable uncertainty parameters (initialized to 1.0)
        # Using log(1 + σ²) to ensure positivity
        self.log_sigma_huber = nn.Parameter(torch.tensor(0.0))  # log(1 + 1²) ≈ 0.69
        self.log_sigma_ranking = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_neural_ndcg = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_listwise = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pairs: Optional[torch.Tensor] = None,
        pair_target_diffs: Optional[torch.Tensor] = None,
        smooth_ranking: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute uncertainty-weighted loss.

        Returns:
            (total_loss, diagnostics) where diagnostics contains component info
        """
        pred_1d = predictions.squeeze() if predictions.dim() > 1 else predictions
        target_1d = targets.squeeze() if targets.dim() > 1 else targets

        diagnostics = {}

        # Compute losses
        huber = huber_loss(predictions, targets, delta=self.huber_delta)
        sigma_huber_sq = 1.0 + torch.exp(2 * self.log_sigma_huber)  # 1 + σ²
        huber_weighted = (1.0 / (2.0 * sigma_huber_sq)) * huber + self.log_sigma_huber

        total_loss = huber_weighted
        diagnostics["huber"] = huber.item()
        diagnostics["sigma_huber"] = torch.sqrt(sigma_huber_sq - 1.0).item()

        # NeuralNDCG
        if self.use_neural_ndcg:
            ndcg_loss = neural_ndcg_loss_simple(pred_1d, target_1d)
            sigma_ndcg_sq = 1.0 + torch.exp(2 * self.log_sigma_neural_ndcg)
            ndcg_weighted = (1.0 / (2.0 * sigma_ndcg_sq)) * ndcg_loss + self.log_sigma_neural_ndcg
            total_loss = total_loss + self.neural_ndcg_weight * ndcg_weighted
            diagnostics["neural_ndcg"] = ndcg_loss.item()
            diagnostics["sigma_neural_ndcg"] = torch.sqrt(sigma_ndcg_sq - 1.0).item()

        # Listwise
        if self.use_listwise_ranking:
            if self.listwise_method == "lambdarank":
                listwise_loss = lambdarank_loss(pred_1d, target_1d, sigma=1.0)
            elif self.listwise_method == "approx_ndcg":
                listwise_loss = approx_ndcg_loss(pred_1d, target_1d, temperature=1.0)
            else:
                raise ValueError(f"Unknown listwise method: {self.listwise_method}")

            sigma_listwise_sq = 1.0 + torch.exp(2 * self.log_sigma_listwise)
            listwise_weighted = (
                1.0 / (2.0 * sigma_listwise_sq)
            ) * listwise_loss + self.log_sigma_listwise
            total_loss = total_loss + self.listwise_weight * listwise_weighted
            diagnostics["listwise"] = listwise_loss.item()
            diagnostics["sigma_listwise"] = torch.sqrt(sigma_listwise_sq - 1.0).item()

        # Ranking loss on pairs
        if pairs is not None and len(pairs) > 0:
            idx1, idx2 = pairs[:, 0], pairs[:, 1]
            pred1 = predictions[idx1]
            pred2 = predictions[idx2]
            rank = ranking_loss(
                pred1,
                pred2,
                margin=self.rank_margin,
                target_diff=pair_target_diffs,
                smooth=smooth_ranking,
            )
            sigma_rank_sq = 1.0 + torch.exp(2 * self.log_sigma_ranking)
            rank_weighted = (1.0 / (2.0 * sigma_rank_sq)) * rank + self.log_sigma_ranking
            total_loss = total_loss + self.rank_weight * rank_weighted
            diagnostics["ranking"] = rank.item()
            diagnostics["sigma_ranking"] = torch.sqrt(sigma_rank_sq - 1.0).item()

        diagnostics["total"] = total_loss.item()

        return total_loss, diagnostics


def compute_gradient_norms(
    model: nn.Module,
    losses: Dict[str, torch.Tensor],
    shared_params: Optional[List[nn.Parameter]] = None,
) -> Dict[str, float]:
    """
    Compute gradient norms for each loss component.

    Useful for monitoring gradient balance and detecting dominance.

    Args:
        model: The model
        losses: Dictionary of loss components
        shared_params: Optional list of shared parameters to compute gradients on

    Returns:
        Dictionary of gradient norms per loss component
    """
    if shared_params is None:
        # Use all parameters by default
        shared_params = list(model.parameters())

    grad_norms = {}

    for name, loss in losses.items():
        # Compute gradients for this loss
        grads = torch.autograd.grad(
            loss,
            shared_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        # Compute norm (filter out None gradients)
        valid_grads = [g for g in grads if g is not None]
        if valid_grads:
            grad_norm = torch.norm(torch.cat([g.flatten() for g in valid_grads]))
            grad_norms[name] = grad_norm.item()
        else:
            grad_norms[name] = 0.0

    return grad_norms


def monitor_loss_components(
    losses: Dict[str, float],
    grad_norms: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute monitoring metrics for loss components.

    Returns:
        Dictionary with:
        - loss_ratios: Percentage contribution of each loss
        - grad_ratios: Percentage contribution of each gradient
        - dominance_warnings: Flags for loss dominance (>70%)
    """
    total_loss = sum(losses.values())
    metrics = {}

    # Loss ratios
    loss_ratios = {f"{k}_ratio": v / (total_loss + 1e-8) for k, v in losses.items()}
    metrics.update(loss_ratios)

    # Gradient ratios (if provided)
    if grad_norms:
        total_grad = sum(grad_norms.values())
        grad_ratios = {f"{k}_grad_ratio": v / (total_grad + 1e-8) for k, v in grad_norms.items()}
        metrics.update(grad_ratios)

    # Dominance warnings
    for k, ratio in loss_ratios.items():
        if ratio > 0.7:
            metrics[f"{k}_dominant"] = 1.0
        else:
            metrics[f"{k}_dominant"] = 0.0

    return metrics
