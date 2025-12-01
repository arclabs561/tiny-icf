"""Enhanced multi-loss functions for ICF training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_icf.loss import huber_loss, ranking_loss


def contrastive_loss(
    pred_common: torch.Tensor,
    pred_rare: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Contrastive loss: Push common and rare word predictions apart.
    
    Ensures common words (low ICF) and rare words (high ICF) are well-separated.
    This helps the model learn clear distinctions between frequency classes.
    
    Args:
        pred_common: Predictions for common words [N_common, 1]
        pred_rare: Predictions for rare words [N_rare, 1]
        margin: Minimum separation between common and rare predictions
    
    Returns:
        Scalar loss value
    """
    # We want: pred_common < pred_rare with margin
    # Loss = mean(max(0, margin - (pred_rare - pred_common)))
    # For each pair of common/rare words
    if len(pred_common) == 0 or len(pred_rare) == 0:
        return torch.tensor(0.0, device=pred_common.device)
    
    # Create pairs: each common word with each rare word
    # Use broadcasting: [N_common, 1] - [1, N_rare] = [N_common, N_rare]
    diff = pred_rare.T - pred_common  # [N_common, N_rare]
    loss = F.relu(margin - diff).mean()
    return loss


def consistency_loss(
    predictions: torch.Tensor,
    word_similarity: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Consistency loss: Similar words should have similar ICF predictions.
    
    Uses word similarity matrix to encourage consistent predictions.
    Helps model generalize to similar words.
    
    Args:
        predictions: Model predictions [Batch, 1]
        word_similarity: Similarity matrix [Batch, Batch] (0-1)
        temperature: Temperature for similarity weighting
    
    Returns:
        Scalar loss value
    """
    # L2 distance between predictions, weighted by similarity
    pred_diff = predictions - predictions.T  # [Batch, Batch]
    pred_dist = pred_diff.pow(2)  # [Batch, Batch]
    
    # Weight by similarity: high similarity → low distance
    weighted_dist = word_similarity * pred_dist
    loss = weighted_dist.mean()
    return loss


def calibration_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    bins: int = 10,
) -> torch.Tensor:
    """
    Calibration loss: Ensure predicted ICF distribution matches actual.
    
    Checks that predicted ICF values are well-calibrated to actual frequencies.
    Helps model learn correct frequency scale.
    
    Args:
        predictions: Model predictions [Batch, 1]
        targets: Ground truth ICF [Batch, 1]
        bins: Number of bins for calibration check
    
    Returns:
        Scalar loss value
    """
    # Bin predictions and targets
    pred_bins = torch.linspace(0, 1, bins + 1, device=predictions.device)
    target_bins = torch.linspace(0, 1, bins + 1, device=targets.device)
    
    # Compute bin counts
    pred_counts = torch.histc(predictions.squeeze(), bins=bins, min=0, max=1)
    target_counts = torch.histc(targets.squeeze(), bins=bins, min=0, max=1)
    
    # Normalize to probabilities
    pred_probs = pred_counts / (pred_counts.sum() + 1e-8)
    target_probs = target_counts / (target_counts.sum() + 1e-8)
    
    # KL divergence
    kl = F.kl_div(
        pred_probs.log().unsqueeze(0),
        target_probs.unsqueeze(0),
        reduction='batchmean',
    )
    return kl


class EnhancedMultiLoss(nn.Module):
    """
    Enhanced multi-loss combining multiple objectives.
    
    Components:
    1. Huber loss: Absolute ICF accuracy
    2. Ranking loss: Relative ordering (common < rare)
    3. Contrastive loss: Push common/rare apart
    4. Consistency loss: Similar words → similar ICF
    5. Calibration loss: Match frequency distribution
    """
    
    def __init__(
        self,
        huber_delta: float = 0.1,
        rank_margin: float = 0.05,
        contrastive_margin: float = 0.2,
        huber_weight: float = 1.0,
        rank_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        consistency_weight: float = 0.3,
        calibration_weight: float = 0.2,
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.rank_margin = rank_margin
        self.contrastive_margin = contrastive_margin
        self.huber_weight = huber_weight
        self.rank_weight = rank_weight
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight
        self.calibration_weight = calibration_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pairs: torch.Tensor | None = None,
        common_indices: torch.Tensor | None = None,
        rare_indices: torch.Tensor | None = None,
        word_similarity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            predictions: [Batch, 1] model predictions
            targets: [Batch, 1] ground truth ICF
            pairs: [N_pairs, 2] indices for ranking loss
            common_indices: [N_common] indices of common words (ICF < 0.3)
            rare_indices: [N_rare] indices of rare words (ICF > 0.7)
            word_similarity: [Batch, Batch] similarity matrix (optional)
        """
        total_loss = 0.0
        
        # 1. Huber loss (always)
        huber = huber_loss(predictions, targets, delta=self.huber_delta)
        total_loss += self.huber_weight * huber
        
        # 2. Ranking loss (if pairs provided)
        if pairs is not None and len(pairs) > 0:
            idx1, idx2 = pairs[:, 0], pairs[:, 1]
            rank = ranking_loss(
                predictions[idx1],
                predictions[idx2],
                margin=self.rank_margin,
            )
            total_loss += self.rank_weight * rank
        
        # 3. Contrastive loss (if common/rare indices provided)
        if common_indices is not None and rare_indices is not None:
            if len(common_indices) > 0 and len(rare_indices) > 0:
                pred_common = predictions[common_indices]
                pred_rare = predictions[rare_indices]
                contrastive = contrastive_loss(
                    pred_common,
                    pred_rare,
                    margin=self.contrastive_margin,
                )
                total_loss += self.contrastive_weight * contrastive
        
        # 4. Consistency loss (if similarity matrix provided)
        if word_similarity is not None:
            consistency = consistency_loss(predictions, word_similarity)
            total_loss += self.consistency_weight * consistency
        
        # 5. Calibration loss (always, but lightweight)
        if len(predictions) > 10:  # Need enough samples
            calibration = calibration_loss(predictions, targets)
            total_loss += self.calibration_weight * calibration
        
        return total_loss


class CurriculumMultiLoss(nn.Module):
    """
    Curriculum multi-loss: Progressive loss addition during training.
    
    Stage 1 (epochs 0-33%): Huber + Ranking
    Stage 2 (epochs 33-66%): + Contrastive
    Stage 3 (epochs 66-100%): + Consistency + Calibration
    """
    
    def __init__(
        self,
        total_epochs: int,
        current_epoch: int,
        base_loss: EnhancedMultiLoss,
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.current_epoch = current_epoch
        self.base_loss = base_loss
        
        # Determine stage
        progress = current_epoch / total_epochs
        if progress < 0.33:
            self.stage = 1
        elif progress < 0.66:
            self.stage = 2
        else:
            self.stage = 3
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pairs: torch.Tensor | None = None,
        common_indices: torch.Tensor | None = None,
        rare_indices: torch.Tensor | None = None,
        word_similarity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with curriculum weighting."""
        # Stage 1: Only Huber + Ranking
        if self.stage == 1:
            self.base_loss.contrastive_weight = 0.0
            self.base_loss.consistency_weight = 0.0
            self.base_loss.calibration_weight = 0.0
        
        # Stage 2: Add Contrastive
        elif self.stage == 2:
            self.base_loss.contrastive_weight = 0.5
            self.base_loss.consistency_weight = 0.0
            self.base_loss.calibration_weight = 0.0
        
        # Stage 3: All losses
        else:
            self.base_loss.contrastive_weight = 0.5
            self.base_loss.consistency_weight = 0.3
            self.base_loss.calibration_weight = 0.2
        
        return self.base_loss(
            predictions,
            targets,
            pairs,
            common_indices,
            rare_indices,
            word_similarity,
        )

