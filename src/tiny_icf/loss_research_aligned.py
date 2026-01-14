"""Research-aligned loss functions incorporating findings from comprehensive review.

Key improvements:
1. Adaptive regularization strength (matches data scale)
2. Multiple ranking methods from rank-relax (neural_sort, probabilistic, smooth_i)
3. Focal loss for hard examples
4. Monotonicity constraints
5. Quantile regression for uncertainty
6. Temperature scaling for calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math

# Try to import rank-relax
try:
    import rank_relax
    HAS_RANK_RELAX = True
except ImportError:
    HAS_RANK_RELAX = False
    print("Warning: rank-relax not available, falling back to built-in implementations")


def adaptive_regularization_strength(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    min_reg: float = 0.1,
    max_reg: float = 100.0,
) -> torch.Tensor:
    """
    Adaptively set regularization strength based on data scale.
    
    Research finding: Match regularization_strength to typical difference between values.
    Rule of thumb: reg_strength ≈ 1.0 / typical_difference
    
    Args:
        predictions: Model predictions [batch] or [batch, 1]
        targets: Ground truth [batch] or [batch, 1]
        min_reg: Minimum regularization strength
        max_reg: Maximum regularization strength
    
    Returns:
        Adaptive regularization strength (scalar tensor)
    """
    # Ensure 1D
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    if targets.dim() > 1:
        targets = targets.squeeze()
    
    # Compute typical difference between values
    # Research: Use median absolute deviation (MAD) for robustness to outliers
    # Alternative: Use IQR (interquartile range) for better robustness
    pred_std = torch.std(predictions)
    target_std = torch.std(targets)
    
    # Use median absolute deviation as more robust measure
    pred_median = torch.median(predictions)
    target_median = torch.median(targets)
    pred_mad = torch.median(torch.abs(predictions - pred_median))
    target_mad = torch.median(torch.abs(targets - target_median))
    
    # Combine std and MAD for balanced approach
    typical_diff = (pred_std + target_std + pred_mad + target_mad) / 4.0
    
    # Set regularization strength: 1.0 / typical_difference
    # Research: Adaptive reg should match data scale, but be robust to outliers
    reg_strength = 1.0 / (typical_diff + 1e-6)
    
    # Clamp to reasonable range
    reg_strength = torch.clamp(reg_strength, min_reg, max_reg)
    
    return reg_strength


def soft_rank_with_method_adaptive(
    values: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    method: str = "sigmoid",
    base_reg_strength: float = 1.0,
    adaptive: bool = True,
) -> Tuple[torch.Tensor, float]:
    """
    Soft ranking with adaptive regularization and multiple methods.
    
    Research finding: Different methods (neural_sort, probabilistic) have different
    gradient profiles and may be better for different tasks.
    
    Args:
        values: Tensor to rank [n] or [batch, n]
        targets: Optional targets for adaptive regularization [n] or [batch, n]
        method: Ranking method ("sigmoid", "neural_sort", "probabilistic", "smooth_i")
        base_reg_strength: Base regularization strength (used if adaptive=False)
        adaptive: If True, adapt regularization based on data scale
    
    Returns:
        (ranks, regularization_strength_used)
    """
    if not HAS_RANK_RELAX:
        # Fallback: use built-in sigmoid method
        from tiny_icf.loss_unified import soft_rank_tensor
        reg_strength = base_reg_strength if not adaptive else 1.0
        if adaptive and targets is not None:
            reg_strength = adaptive_regularization_strength(values, targets).item()
        ranks = soft_rank_tensor(values, reg_strength, method="sigmoid")
        return ranks, reg_strength
    
    # Determine regularization strength
    if adaptive and targets is not None:
        reg_strength = adaptive_regularization_strength(values, targets).item()
    else:
        reg_strength = base_reg_strength
    
    # Convert to list for rank-relax
    device = values.device
    dtype = values.dtype
    
    if values.dim() == 1:
        values_list = values.detach().cpu().tolist()
        ranks_list = rank_relax.soft_rank_with_method(
            values_list, reg_strength, method=method
        )
        ranks = torch.tensor(ranks_list, device=device, dtype=dtype, requires_grad=True)
    else:
        # Batch: process each item
        batch_size = values.shape[0]
        ranks_list = []
        for i in range(batch_size):
            values_i = values[i].detach().cpu().tolist()
            ranks_i = rank_relax.soft_rank_with_method(
                values_i, reg_strength, method=method
            )
            ranks_list.append(ranks_i)
        
        ranks = torch.stack([
            torch.tensor(r, device=device, dtype=dtype, requires_grad=True)
            for r in ranks_list
        ])
    
    return ranks, reg_strength


def focal_icf_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    base_loss_fn,  # type: ignore
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss for ICF: focus on hard examples (large errors).
    
    Research finding: Focal loss downweights easy examples, focusing on hard cases.
    Particularly effective for class imbalance and hard example mining.
    
    Args:
        predictions: [batch, 1] or [batch] model predictions
        targets: [batch, 1] or [batch] ground truth
        base_loss_fn: Base loss function (e.g., F.smooth_l1_loss)
        gamma: Focusing parameter (higher = more focus on hard examples)
    
    Returns:
        Focal loss value
    """
    # Ensure 1D
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    if targets.dim() > 1:
        targets = targets.squeeze()
    
    # Compute base loss
    base_loss = base_loss_fn(predictions, targets)
    
    # Compute error magnitude
    error = torch.abs(predictions - targets)
    
    # Focal weighting: large errors get exponentially more weight
    focal_weight = (1.0 + error) ** gamma
    
    return (focal_weight * base_loss).mean()


def monotonicity_loss(
    predictions: torch.Tensor,
    features: Dict[str, torch.Tensor],
    constraints: Dict[str, str],
) -> torch.Tensor:
    """
    Enforce monotonicity constraints.
    
    Research finding: Monotonicity constraints improve generalization and interpretability.
    
    Args:
        predictions: [batch, 1] or [batch] ICF predictions
        features: Dict mapping feature names to feature tensors [batch]
        constraints: Dict mapping feature names to direction ("increasing" or "decreasing")
    
    Returns:
        Loss penalizing monotonicity violations
    """
    # Ensure 1D
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    
    loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    for feature_name, direction in constraints.items():
        if feature_name not in features:
            continue
        
        feature = features[feature_name]
        if feature.dim() > 1:
            feature = feature.squeeze()
        
        # Compute correlation between feature and predictions
        # Use cosine similarity as a proxy for correlation
        feature_centered = feature - feature.mean()
        pred_centered = predictions - predictions.mean()
        
        # Cosine similarity (normalized correlation)
        numerator = (feature_centered * pred_centered).sum()
        denominator = (
            torch.sqrt((feature_centered ** 2).sum()) *
            torch.sqrt((pred_centered ** 2).sum()) + 1e-8
        )
        correlation = numerator / denominator
        
        # Penalize if correlation has wrong sign
        if direction == 'increasing' and correlation < 0:
            # Should be positive correlation (longer words → higher ICF)
            loss = loss + F.relu(-correlation)
        elif direction == 'decreasing' and correlation > 0:
            # Should be negative correlation (rare chars → higher ICF)
            loss = loss + F.relu(correlation)
    
    return loss


def quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile: float = 0.5,
) -> torch.Tensor:
    """
    Quantile regression loss.
    
    Research finding: Quantile regression provides principled uncertainty intervals.
    Calibration-guided quantile regression improves both sharpness and calibration.
    
    Args:
        predictions: [batch, 1] or [batch] model predictions
        targets: [batch, 1] or [batch] ground truth
        quantile: Desired quantile (0.5 = median, 0.9 = 90th percentile)
    
    Returns:
        Quantile loss value
    """
    # Ensure 1D
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    if targets.dim() > 1:
        targets = targets.squeeze()
    
    error = predictions - targets
    
    # Asymmetric weighting: quantile loss
    loss = torch.max(
        quantile * error,
        (quantile - 1.0) * error
    )
    
    return loss.mean()


class ResearchAlignedICFLoss(nn.Module):
    """
    Research-aligned ICF loss incorporating multiple research findings:
    
    1. Adaptive regularization strength (matches data scale)
    2. Multiple ranking methods (neural_sort, probabilistic, smooth_i)
    3. Focal loss for hard examples
    4. Monotonicity constraints
    5. Quantile regression for uncertainty
    6. Asymmetric penalties (common→rare worse than rare→common)
    
    This loss function aligns with research findings on:
    - Calibration-guided quantile regression
    - Focal loss for hard example mining
    - Monotonicity constraints for generalization
    - Adaptive regularization for ranking operations
    """
    
    def __init__(
        self,
        # Base loss parameters
        huber_delta: float = 0.1,
        asymmetry_factor: float = 2.0,
        # Ranking parameters
        rank_margin: float = 0.1,
        rank_weight: float = 0.5,
        ranking_method: str = "sigmoid",  # "sigmoid", "neural_sort", "probabilistic", "smooth_i"
        adaptive_reg: bool = True,  # Use adaptive regularization strength
        base_reg_strength: float = 1.0,  # Base if not adaptive
        # Focal loss parameters
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        # Monotonicity parameters
        use_monotonicity: bool = False,
        monotonicity_weight: float = 0.1,
        monotonicity_constraints: Optional[Dict[str, str]] = None,
        # Quantile regression parameters
        use_quantile: bool = False,
        quantile: float = 0.5,
        quantile_weight: float = 0.3,
        # Spearman parameters
        use_spearman: bool = True,
        spearman_weight: float = 10.0,
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.asymmetry_factor = asymmetry_factor
        self.rank_margin = rank_margin
        self.rank_weight = rank_weight
        self.ranking_method = ranking_method
        self.adaptive_reg = adaptive_reg
        self.base_reg_strength = base_reg_strength
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.use_monotonicity = use_monotonicity
        self.monotonicity_weight = monotonicity_weight
        self.monotonicity_constraints = monotonicity_constraints or {}
        self.use_quantile = use_quantile
        self.quantile = quantile
        self.quantile_weight = quantile_weight
        self.use_spearman = use_spearman
        self.spearman_weight = spearman_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pairs: Optional[torch.Tensor] = None,
        features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute research-aligned ICF loss.
        
        Args:
            predictions: [batch, 1] or [batch] model predictions
            targets: [batch, 1] or [batch] ground truth ICF scores
            pairs: Optional [n_pairs, 2] indices for pairwise ranking
            features: Optional dict of feature tensors for monotonicity constraints
        
        Returns:
            (total_loss, component_losses)
        """
        # Ensure 1D
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        components = {}
        
        # 1. Asymmetric Huber loss (with optional focal weighting)
        error = predictions - targets
        huber_base = F.smooth_l1_loss(predictions, targets, beta=self.huber_delta)
        
        # Asymmetric penalty
        asymmetric_penalty = torch.where(
            error > 0,  # Common → rare
            self.asymmetry_factor * F.relu(error),
            F.relu(-error)  # Rare → common (less penalty)
        )
        huber_loss = huber_base + asymmetric_penalty.mean()
        
        # Apply focal weighting if enabled
        # Refined: Apply focal to the base loss before asymmetric penalty for better hard example focus
        if self.use_focal:
            # Compute focal weight based on error magnitude
            abs_error = torch.abs(error)
            focal_weight = (abs_error / (abs_error.mean() + 1e-8)) ** self.focal_gamma
            huber_loss = (focal_weight * huber_base).mean() + asymmetric_penalty.mean()
        
        components['huber'] = huber_loss
        components['asymmetric_penalty'] = asymmetric_penalty.mean()
        
        total_loss = huber_loss
        
        # 2. Spearman loss (with adaptive regularization and multiple methods)
        if self.use_spearman:
            # Use adaptive regularization if enabled
            if self.adaptive_reg:
                reg_strength = adaptive_regularization_strength(predictions, targets).item()
            else:
                reg_strength = self.base_reg_strength
            
            # Use multiple ranking methods
            pred_ranks, _ = soft_rank_with_method_adaptive(
                predictions, targets,
                method=self.ranking_method,
                base_reg_strength=reg_strength,
                adaptive=False,  # Already computed above
            )
            target_ranks, _ = soft_rank_with_method_adaptive(
                targets, None,
                method=self.ranking_method,
                base_reg_strength=reg_strength,
                adaptive=False,
            )
            
            # Compute Spearman correlation
            pred_centered = pred_ranks - pred_ranks.mean()
            target_centered = target_ranks - target_ranks.mean()
            numerator = (pred_centered * target_centered).sum()
            denominator = torch.sqrt(
                (pred_centered ** 2).sum() * (target_centered ** 2).sum()
            ) + 1e-8
            spearman_corr = numerator / denominator
            spearman_loss = 1.0 - spearman_corr
            
            components['spearman'] = spearman_loss
            components['reg_strength'] = torch.tensor(reg_strength, device=predictions.device)
            total_loss = total_loss + self.spearman_weight * spearman_loss
        
        # 3. Ranking loss (if pairs provided)
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
            
            # Magnitude weighting (larger differences = more important)
            # Refined: Use softmax to emphasize pairs with larger ICF differences
            target_diff_abs = torch.abs(target_diff)
            magnitude_weight = torch.softmax(target_diff_abs / (target_diff_abs.mean() + 1e-8), dim=0)
            base_rank_loss = magnitude_weight * base_rank_loss
            
            # Focal weighting for hard examples (ranking errors)
            if self.use_focal:
                # Hard examples: pairs where prediction order is wrong
                pred_order_correct = (pred_diff * torch.sign(target_diff)) > 0
                error_magnitude = torch.abs(pred_diff - target_diff)
                # Higher weight for incorrect order or large errors
                focal_weight = (1.0 + error_magnitude * (1.0 - pred_order_correct.float())) ** self.focal_gamma
                base_rank_loss = focal_weight * base_rank_loss
            
            rank_loss = base_rank_loss.mean()
            components['rank'] = rank_loss
            total_loss = total_loss + self.rank_weight * rank_loss
        
        # 4. Monotonicity constraints (if enabled and features provided)
        if self.use_monotonicity and features is not None:
            mono_loss = monotonicity_loss(
                predictions, features, self.monotonicity_constraints
            )
            components['monotonicity'] = mono_loss
            total_loss = total_loss + self.monotonicity_weight * mono_loss
        
        # 5. Quantile regression loss (if enabled)
        if self.use_quantile:
            quantile_loss_val = quantile_loss(predictions, targets, self.quantile)
            components['quantile'] = quantile_loss_val
            total_loss = total_loss + self.quantile_weight * quantile_loss_val
        
        components['total'] = total_loss
        
        return total_loss, components


class TemperatureScaledModel(nn.Module):
    """
    Wraps model with learnable temperature parameter for calibration.
    
    Research finding: Temperature scaling is simple and effective for post-hoc calibration.
    Single parameter recalibrates entire model's output distribution.
    
    Args:
        base_model: Base model to wrap
        init_temperature: Initial temperature value (typically 1.0)
    """
    
    def __init__(self, base_model: nn.Module, init_temperature: float = 1.0):
        super().__init__()
        self.base_model = base_model
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temperature scaling.
        
        Args:
            x: Input tensor
        
        Returns:
            Scaled predictions: logits / temperature
        """
        logits = self.base_model(x)
        return logits / self.temperature
    
    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()

