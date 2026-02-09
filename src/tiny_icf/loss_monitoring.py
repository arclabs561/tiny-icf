# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
# ]
# ///
"""
Loss monitoring utilities for multi-objective optimization.

Provides functions to monitor loss components, detect imbalances,
and compute diagnostic metrics.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


def compute_loss_component_metrics(
    loss_components: Dict[str, float],
    grad_norms: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for loss components.

    Args:
        loss_components: Dictionary of loss component values
        grad_norms: Optional dictionary of gradient norms per component

    Returns:
        Dictionary with:
        - Component values
        - Component ratios (percentage contribution)
        - Dominance warnings (>70% contribution)
        - Gradient ratios (if provided)
        - Balance score (lower = more balanced)
    """
    metrics = {}

    # Component values
    metrics.update({f"{k}_value": v for k, v in loss_components.items()})

    # Total loss
    total_loss = sum(loss_components.values())
    metrics["total_loss"] = total_loss

    # Component ratios
    for k, v in loss_components.items():
        ratio = v / (total_loss + 1e-8)
        metrics[f"{k}_ratio"] = ratio

        # Dominance warning (>70%)
        if ratio > 0.7:
            metrics[f"{k}_dominant"] = 1.0
        else:
            metrics[f"{k}_dominant"] = 0.0

    # Gradient ratios (if provided)
    if grad_norms:
        total_grad = sum(grad_norms.values())
        for k, v in grad_norms.items():
            grad_ratio = v / (total_grad + 1e-8)
            metrics[f"{k}_grad_norm"] = v
            metrics[f"{k}_grad_ratio"] = grad_ratio

            # Gradient dominance warning
            if grad_ratio > 0.7:
                metrics[f"{k}_grad_dominant"] = 1.0
            else:
                metrics[f"{k}_grad_dominant"] = 0.0

    # Balance score: coefficient of variation of ratios
    # Lower = more balanced, higher = less balanced
    ratios = [metrics[f"{k}_ratio"] for k in loss_components.keys()]
    if len(ratios) > 1 and np.std(ratios) > 0:
        balance_score = np.std(ratios) / (np.mean(ratios) + 1e-8)
        metrics["balance_score"] = balance_score
    else:
        metrics["balance_score"] = 0.0

    return metrics


def detect_loss_imbalance(
    loss_components: Dict[str, float],
    threshold: float = 0.7,
) -> Tuple[bool, List[str]]:
    """
    Detect if any loss component is dominating.

    Args:
        loss_components: Dictionary of loss component values
        threshold: Ratio threshold for dominance (default: 0.7 = 70%)

    Returns:
        (is_imbalanced, dominant_components)
    """
    total_loss = sum(loss_components.values())
    if total_loss == 0:
        return False, []

    dominant = []
    for k, v in loss_components.items():
        ratio = v / total_loss
        if ratio > threshold:
            dominant.append(k)

    return len(dominant) > 0, dominant


def compute_gradient_balance(
    model: nn.Module,
    loss_dict: Dict[str, torch.Tensor],
    shared_params: Optional[List[nn.Parameter]] = None,
) -> Dict[str, float]:
    """
    Compute gradient balance metrics.

    Args:
        model: The model
        loss_dict: Dictionary of loss components (as tensors)
        shared_params: Optional list of shared parameters

    Returns:
        Dictionary with gradient norms and balance metrics
    """
    if shared_params is None:
        shared_params = list(model.parameters())

    grad_norms = {}

    # Compute gradients for each loss
    for name, loss in loss_dict.items():
        grads = torch.autograd.grad(
            loss,
            shared_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        # Compute norm
        valid_grads = [g for g in grads if g is not None]
        if valid_grads:
            grad_norm = torch.norm(torch.cat([g.flatten() for g in valid_grads]))
            grad_norms[name] = grad_norm.item()
        else:
            grad_norms[name] = 0.0

    # Compute balance metrics
    total_grad = sum(grad_norms.values())
    metrics = {}

    for name, norm in grad_norms.items():
        metrics[f"{name}_grad_norm"] = norm
        if total_grad > 0:
            metrics[f"{name}_grad_ratio"] = norm / total_grad
        else:
            metrics[f"{name}_grad_ratio"] = 0.0

    # Gradient balance score
    if len(grad_norms) > 1:
        ratios = [v / (total_grad + 1e-8) for v in grad_norms.values()]
        if np.std(ratios) > 0:
            balance_score = np.std(ratios) / (np.mean(ratios) + 1e-8)
            metrics["grad_balance_score"] = balance_score
        else:
            metrics["grad_balance_score"] = 0.0

    return metrics


def log_loss_components(
    epoch: int,
    batch_idx: int,
    loss_components: Dict[str, float],
    logger=None,
    prefix: str = "loss",
) -> None:
    """
    Log loss components in a structured format.

    Args:
        epoch: Current epoch
        batch_idx: Current batch index
        loss_components: Dictionary of loss component values
        logger: Optional logger (e.g., wandb, tensorboard)
        prefix: Prefix for log keys
    """
    if logger is None:
        # Simple print logging
        components_str = ", ".join([f"{k}={v:.4f}" for k, v in loss_components.items()])
        print(f"Epoch {epoch}, Batch {batch_idx}: {components_str}")
        return

    # Structured logging (e.g., wandb, tensorboard)
    log_dict = {f"{prefix}/{k}": v for k, v in loss_components.items()}

    if hasattr(logger, "log"):
        logger.log(log_dict)
    elif hasattr(logger, "add_scalars"):
        logger.add_scalars(prefix, log_dict, epoch * 1000 + batch_idx)
