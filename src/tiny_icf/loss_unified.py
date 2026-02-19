"""Unified loss framework using rank-relax for all ranking operations.

This module rethinks all losses around rank-relax's differentiable ranking
capabilities, structuring losses by task (ICF prediction, text reduction,
temporal prediction, language detection, era classification) and using
soft ranking for all ranking-related operations.

Key principles:
1. Use rank-relax for ALL ranking operations (not just Spearman)
2. Structure losses by task, not by loss type
3. Use soft ranking for text reduction (rank words by ICF)
4. Use soft ranking for temporal consistency (rank predictions across decades)
5. Use soft ranking for multi-class classification (rank confidence scores)
6. Integrate AMOO for multi-objective optimization
7. Make everything differentiable and principled
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any

# Try to import rank-relax
try:
    import rank_relax

    HAS_RANK_RELAX = True
except ImportError:
    HAS_RANK_RELAX = False
    print("Warning: rank-relax not available, falling back to built-in implementations")


def _to_list(tensor: torch.Tensor) -> List[float]:
    """Convert tensor to Python list for rank-relax."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().tolist()
    return tensor


def _from_list(values: List[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert Python list to tensor."""
    return torch.tensor(values, device=device, dtype=dtype, requires_grad=True)


# ============================================================================
# Core Ranking Operations (using rank-relax)
# ============================================================================


def soft_rank_tensor(
    values: torch.Tensor,
    regularization_strength: float = 1.0,
    method: str = "sigmoid",
    adaptive: bool = False,
    targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute soft ranks for a tensor using rank-relax.

    Research-aligned: Supports adaptive regularization and multiple methods.

    Args:
        values: Tensor of values to rank [n] or [batch, n]
        regularization_strength: Temperature parameter (higher = sharper)
        method: Ranking method ("sigmoid", "neural_sort", "probabilistic", "smooth_i")
        adaptive: If True, adapt regularization based on data scale (requires targets)
        targets: Optional targets for adaptive regularization

    Returns:
        Soft ranks tensor with same shape as values
    """
    # Adaptive regularization (research finding: match to data scale)
    if adaptive and targets is not None:
        # Compute typical difference
        if values.dim() > 1:
            values_1d = values.flatten()
            targets_1d = targets.flatten() if targets.dim() > 1 else targets
        else:
            values_1d = values
            targets_1d = targets

        pred_std = torch.std(values_1d)
        target_std = torch.std(targets_1d)
        typical_diff = (pred_std + target_std) / 2.0
        regularization_strength = (1.0 / (typical_diff + 1e-6)).item()
        regularization_strength = max(0.1, min(100.0, regularization_strength))

    if not HAS_RANK_RELAX:
        # Fallback: simple sigmoid-based ranking
        if values.dim() == 1:
            values_2d = values.unsqueeze(0)
        else:
            values_2d = values

        # Compute pairwise comparisons
        values_i = values_2d.unsqueeze(-1)  # [batch, n, 1]
        values_j = values_2d.unsqueeze(-2)  # [batch, 1, n]

        # Sigmoid-based soft ranking
        alpha = regularization_strength
        comparisons = torch.sigmoid(alpha * (values_i - values_j))  # [batch, n, n]

        # Sum over j != i
        ranks = comparisons.sum(dim=-1) - 1.0  # Subtract 1 for self-comparison

        if values.dim() == 1:
            return ranks.squeeze(0)
        return ranks

    # Use rank-relax with multiple methods (research finding: different methods have different gradient profiles)
    device = values.device
    dtype = values.dtype

    if values.dim() == 1:
        # Single vector
        values_list = _to_list(values)
        if method != "sigmoid":
            ranks_list = rank_relax.soft_rank_with_method(
                values_list, regularization_strength, method=method
            )
        else:
            ranks_list = rank_relax.soft_rank(values_list, regularization_strength)
        return _from_list(ranks_list, device, dtype)
    else:
        # Batch: process each item
        batch_size = values.shape[0]
        ranks_list = []
        for i in range(batch_size):
            values_i = _to_list(values[i])
            if method != "sigmoid":
                ranks_i = rank_relax.soft_rank_with_method(
                    values_i, regularization_strength, method=method
                )
            else:
                ranks_i = rank_relax.soft_rank(values_i, regularization_strength)
            ranks_list.append(ranks_i)

        ranks_tensor = torch.stack([_from_list(r, device, dtype) for r in ranks_list])
        return ranks_tensor


def spearman_loss_tensor(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    regularization_strength: float = 1.0,
    method: str = "sigmoid",
    adaptive: bool = False,
) -> torch.Tensor:
    """
    Compute Spearman correlation loss using rank-relax.

    Research-aligned: Supports adaptive regularization and multiple methods.

    Loss = 1 - Spearman correlation (so lower is better).

    Args:
        predictions: Model predictions [n] or [batch, n]
        targets: Ground truth values [n] or [batch, n]
        regularization_strength: Temperature for soft ranking
        method: Ranking method ("sigmoid", "neural_sort", "probabilistic", "smooth_i")
        adaptive: If True, adapt regularization based on data scale

    Returns:
        Loss tensor (scalar or [batch])
    """
    if not HAS_RANK_RELAX:
        # Fallback: use soft ranking and compute correlation manually
        pred_ranks = soft_rank_tensor(
            predictions, regularization_strength, method=method, adaptive=adaptive, targets=targets
        )
        target_ranks = soft_rank_tensor(
            targets, regularization_strength, method=method, adaptive=False
        )

        # Compute Pearson correlation of ranks
        if pred_ranks.dim() == 1:
            pred_centered = pred_ranks - pred_ranks.mean()
            target_centered = target_ranks - target_ranks.mean()
            numerator = (pred_centered * target_centered).sum()
            denominator = torch.sqrt((pred_centered**2).sum() * (target_centered**2).sum()) + 1e-8
            correlation = numerator / denominator
            return 1.0 - correlation
        else:
            # Batch case
            pred_centered = pred_ranks - pred_ranks.mean(dim=-1, keepdim=True)
            target_centered = target_ranks - target_ranks.mean(dim=-1, keepdim=True)
            numerator = (pred_centered * target_centered).sum(dim=-1)
            denominator = (
                torch.sqrt((pred_centered**2).sum(dim=-1) * (target_centered**2).sum(dim=-1)) + 1e-8
            )
            correlation = numerator / denominator
            return 1.0 - correlation

    # Use rank-relax with adaptive regularization and multiple methods
    # Research finding: match regularization to data scale
    if adaptive:
        if predictions.dim() > 1:
            pred_1d = predictions.flatten()
            target_1d = targets.flatten() if targets.dim() > 1 else targets
        else:
            pred_1d = predictions
            target_1d = targets

        pred_std = torch.std(pred_1d)
        target_std = torch.std(target_1d)
        typical_diff = (pred_std + target_std) / 2.0
        regularization_strength = (1.0 / (typical_diff + 1e-6)).item()
        regularization_strength = max(0.1, min(100.0, regularization_strength))

    device = predictions.device
    dtype = predictions.dtype

    if predictions.dim() == 1:
        # Single vector
        pred_list = _to_list(predictions)
        target_list = _to_list(targets)
        loss_val = rank_relax.spearman_loss(pred_list, target_list, regularization_strength)
        return torch.tensor(loss_val, device=device, dtype=dtype, requires_grad=True)
    else:
        # Batch: process each item
        batch_size = predictions.shape[0]
        losses = []
        for i in range(batch_size):
            pred_list = _to_list(predictions[i])
            target_list = _to_list(targets[i])
            loss_val = rank_relax.spearman_loss(pred_list, target_list, regularization_strength)
            losses.append(loss_val)

        return torch.tensor(losses, device=device, dtype=dtype, requires_grad=True)


# ============================================================================
# Task-Specific Losses
# ============================================================================


class ICFPredictionLoss(nn.Module):
    """
    Loss for ICF prediction task.

    Combines:
    - Huber loss for absolute values
    - Spearman loss for ranking (using rank-relax)
    - Optional pairwise ranking loss
    """

    def __init__(
        self,
        huber_delta: float = 0.1,
        spearman_weight: float = 10.0,
        spearman_reg_strength: float = 1.0,
        spearman_method: str = "sigmoid",  # "sigmoid", "neural_sort", "probabilistic", "smooth_i"
        spearman_adaptive: bool = False,  # Use adaptive regularization
        rank_weight: float = 0.5,
        rank_margin: float = 0.1,
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.spearman_weight = spearman_weight
        self.spearman_reg_strength = spearman_reg_strength
        self.rank_weight = rank_weight
        self.rank_margin = rank_margin
        self.spearman_method = spearman_method
        self.spearman_adaptive = spearman_adaptive

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pairs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions: [batch, 1] or [batch] model predictions
            targets: [batch, 1] or [batch] ground truth ICF
            pairs: Optional [n_pairs, 2] indices for pairwise ranking

        Returns:
            (total_loss, component_losses)
        """
        # Ensure 1D for Spearman
        pred_1d = predictions.squeeze() if predictions.dim() > 1 else predictions
        target_1d = targets.squeeze() if targets.dim() > 1 else targets

        # Huber loss (absolute values)
        huber = F.smooth_l1_loss(predictions, targets, reduction="mean", beta=self.huber_delta)

        # Spearman loss (ranking correlation) with adaptive regularization and multiple methods
        if int(pred_1d.numel()) < 2 or int(target_1d.numel()) < 2:
            # No meaningful ranking signal for a singleton batch.
            spearman = torch.tensor(0.0, device=predictions.device)
        else:
            spearman = spearman_loss_tensor(
                pred_1d,
                target_1d,
                regularization_strength=self.spearman_reg_strength,
                method=self.spearman_method,
                adaptive=self.spearman_adaptive,
            )
            if spearman.dim() > 0:
                spearman = spearman.mean()

        # Pairwise ranking loss (if pairs provided)
        rank_loss = torch.tensor(0.0, device=predictions.device)
        if pairs is not None and len(pairs) > 0:
            idx1, idx2 = pairs[:, 0], pairs[:, 1]
            pred1 = predictions[idx1].squeeze()
            pred2 = predictions[idx2].squeeze()
            target1 = targets[idx1].squeeze()
            target2 = targets[idx2].squeeze()

            # Margin-based ranking loss
            target_diff = target1 - target2
            pred_diff = pred1 - pred2

            # Loss when ranking is wrong: max(0, margin - (pred_diff * sign(target_diff)))
            margin_loss = F.relu(self.rank_margin - pred_diff * torch.sign(target_diff))
            rank_loss = margin_loss.mean()

        total_loss = huber + self.spearman_weight * spearman + self.rank_weight * rank_loss

        components = {
            "huber": huber,
            "spearman": spearman,
            "rank": rank_loss,
        }

        return total_loss, components


class TextReductionLoss(nn.Module):
    """
    Loss for text reduction task: minimize embedding regret (path minimization).

    This task can be:
    1. **Coupled with ICF**: Uses ICF scores to rank words (rare words = important)
    2. **Disjoint from ICF**: Directly optimizes embedding similarity without ICF

    The objective is to find the minimal "path" of embedding regret - select words
    such that the embedding of the reduced text is as close as possible to the original.

    Uses soft ranking to rank words (by ICF if provided, or by embedding importance),
    then computes regret (embedding difference) for reduced text.
    """

    def __init__(
        self,
        regret_weight: float = 1.0,
        path_regret_weight: float = 0.3,  # Track cumulative embedding changes
        ranking_weight: float = 0.5,
        regularization_strength: float = 1.0,
        use_icf_ranking: bool = True,  # If False, rank by embedding importance directly
    ):
        super().__init__()
        self.regret_weight = regret_weight
        self.path_regret_weight = path_regret_weight
        self.ranking_weight = ranking_weight
        self.regularization_strength = regularization_strength
        self.use_icf_ranking = use_icf_ranking

    def forward(
        self,
        word_icf_scores: Optional[torch.Tensor] = None,  # Optional: can be None if disjoint
        original_embedding: torch.Tensor = None,
        reduced_embedding: torch.Tensor = None,
        word_embeddings: Optional[torch.Tensor] = None,  # [n_words, embed_dim] for direct ranking
        target_length: int = None,
        reduction_path: Optional[List[torch.Tensor]] = None,  # Optional: track path of embeddings
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            word_icf_scores: [n_words] ICF scores (optional, if use_icf_ranking=True)
            original_embedding: [embed_dim] original text embedding
            reduced_embedding: [embed_dim] reduced text embedding
            word_embeddings: [n_words, embed_dim] individual word embeddings (for direct ranking)
            target_length: Target number of words to keep
            reduction_path: Optional list of embeddings along reduction path (for path regret)

        Returns:
            (total_loss, component_losses)
        """
        # Embedding regret (cosine distance) - final regret
        cos_sim = F.cosine_similarity(
            original_embedding.unsqueeze(0),
            reduced_embedding.unsqueeze(0),
        )
        regret = 1.0 - cos_sim

        # Path regret: cumulative embedding change along reduction path
        path_regret = torch.tensor(0.0, device=original_embedding.device)
        if reduction_path is not None and len(reduction_path) > 1:
            # Compute cumulative cosine distance along path
            path_distances = []
            prev_embedding = original_embedding
            for step_embedding in reduction_path:
                step_cos_sim = F.cosine_similarity(
                    prev_embedding.unsqueeze(0),
                    step_embedding.unsqueeze(0),
                )
                path_distances.append(1.0 - step_cos_sim)
                prev_embedding = step_embedding
            path_regret = torch.stack(path_distances).mean()

        # Ranking loss: words kept should be important (by ICF or by embedding)
        ranking_loss = torch.tensor(0.0, device=original_embedding.device)

        if self.use_icf_ranking and word_icf_scores is not None:
            # Option 1: Rank by ICF (coupled with ICF prediction)
            icf_ranks = soft_rank_tensor(
                word_icf_scores, regularization_strength=self.regularization_strength
            )
            # Encourage keeping words with high ICF (high ranks)
            ranking_loss = -icf_ranks.mean()  # Negative because higher ranks are better

        elif word_embeddings is not None:
            # Option 2: Rank by embedding importance (disjoint from ICF)
            # Compute importance as similarity to original embedding
            word_importance = F.cosine_similarity(
                word_embeddings,
                original_embedding.unsqueeze(0).expand(word_embeddings.shape[0], -1),
            )
            importance_ranks = soft_rank_tensor(
                word_importance, regularization_strength=self.regularization_strength
            )
            # Encourage keeping words with high importance (high ranks)
            ranking_loss = -importance_ranks.mean()

        total_loss = (
            self.regret_weight * regret
            + self.path_regret_weight * path_regret
            + self.ranking_weight * ranking_loss
        )

        components = {
            "regret": regret,
            "path_regret": path_regret,
            "ranking": ranking_loss,
        }

        return total_loss, components


class TemporalICFLoss(nn.Module):
    """
    Loss for temporal ICF prediction: consistency across decades.

    Uses soft ranking to ensure predictions are consistent with historical trends.
    """

    def __init__(
        self,
        base_weight: float = 1.0,
        consistency_weight: float = 0.1,
        ranking_weight: float = 0.5,
        regularization_strength: float = 1.0,
    ):
        super().__init__()
        self.base_weight = base_weight
        self.consistency_weight = consistency_weight
        self.ranking_weight = ranking_weight
        self.regularization_strength = regularization_strength

    def forward(
        self,
        current_predictions: torch.Tensor,
        current_targets: torch.Tensor,
        historical_predictions: Optional[Dict[str, torch.Tensor]] = None,
        historical_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            current_predictions: [batch] current ICF predictions
            current_targets: [batch] current ICF targets
            historical_predictions: Dict mapping decade -> [batch] predictions
            historical_targets: Dict mapping decade -> [batch] targets

        Returns:
            (total_loss, component_losses)
        """
        # Normalize shapes to 1D where possible (prevents rank-relax shape surprises).
        cur_pred = (
            current_predictions.squeeze() if current_predictions.dim() > 1 else current_predictions
        )
        cur_tgt = current_targets.squeeze() if current_targets.dim() > 1 else current_targets

        # Base loss (current predictions)
        base_loss = F.mse_loss(cur_pred, cur_tgt)

        # Temporal consistency loss
        consistency_loss = torch.tensor(0.0, device=current_predictions.device)
        if historical_predictions and historical_targets:
            for decade in historical_predictions.keys():
                if decade in historical_targets:
                    hist_pred = historical_predictions[decade]
                    hist_pred = hist_pred.squeeze() if hist_pred.dim() > 1 else hist_pred

                    # Encourage smooth transitions
                    consistency_loss += F.mse_loss(cur_pred, hist_pred)

        # Ranking loss: predictions across decades should maintain relative order
        ranking_loss = torch.tensor(0.0, device=current_predictions.device)
        if historical_predictions:
            # Collect all predictions (current + historical)
            all_predictions = [cur_pred]
            all_targets = [cur_tgt]

            for decade in sorted(historical_predictions.keys()):
                hp = historical_predictions[decade]
                hp = hp.squeeze() if hp.dim() > 1 else hp
                all_predictions.append(hp)
                if historical_targets and decade in historical_targets:
                    ht = historical_targets[decade]
                    ht = ht.squeeze() if ht.dim() > 1 else ht
                    all_targets.append(ht)

            # Stack: [n_decades, batch]
            pred_stack = torch.stack(all_predictions)  # [n_decades, batch]
            target_stack = torch.stack(all_targets)  # [n_decades, batch]

            # For each word, rank predictions across decades
            # Should match ranking of targets
            for i in range(pred_stack.shape[1]):  # For each word in batch
                pred_ranks = soft_rank_tensor(
                    pred_stack[:, i].squeeze(), regularization_strength=self.regularization_strength
                )
                target_ranks = soft_rank_tensor(
                    target_stack[:, i].squeeze(),
                    regularization_strength=self.regularization_strength,
                )

                # Spearman loss between prediction ranks and target ranks
                ranking_loss += spearman_loss_tensor(
                    pred_ranks, target_ranks, regularization_strength=self.regularization_strength
                )

            ranking_loss = ranking_loss / pred_stack.shape[1]  # Average over batch

        total_loss = (
            self.base_weight * base_loss
            + self.consistency_weight * consistency_loss
            + self.ranking_weight * ranking_loss
        )

        components = {
            "base": base_loss,
            "consistency": consistency_loss,
            "ranking": ranking_loss,
        }

        return total_loss, components


class LanguageDetectionLoss(nn.Module):
    """
    Loss for language detection: multi-class classification with ranking.

    Uses soft ranking to rank language confidence scores.
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        ranking_weight: float = 0.5,
        regularization_strength: float = 1.0,
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.ranking_weight = ranking_weight
        self.regularization_strength = regularization_strength

    def forward(
        self,
        language_logits: torch.Tensor,
        language_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            language_logits: [batch, n_languages] language prediction logits
            language_targets: [batch] true language indices (or [batch, n_languages] one-hot)

        Returns:
            (total_loss, component_losses)
        """
        # Classification loss (cross-entropy)
        if language_targets.dim() == 1:
            # Class indices
            classification_loss = F.cross_entropy(language_logits, language_targets)
        else:
            # One-hot
            classification_loss = F.cross_entropy(language_logits, language_targets.argmax(dim=-1))

        # Ranking loss: top predicted language should match target
        # Use soft ranking to rank confidence scores
        language_probs = F.softmax(language_logits, dim=-1)

        ranking_loss = torch.tensor(0.0, device=language_logits.device)
        for i in range(language_logits.shape[0]):
            # Rank languages by confidence
            conf_ranks = soft_rank_tensor(
                language_probs[i], regularization_strength=self.regularization_strength
            )

            # Target: true language should have rank 0 (highest)
            if language_targets.dim() == 1:
                target_idx = language_targets[i].item()
            else:
                target_idx = language_targets[i].argmax().item()

            # Loss: true language should have highest rank (lowest rank value)
            # In soft ranking, highest confidence = highest rank value
            # So we want target_idx to have the highest rank value
            max_rank = conf_ranks.max()
            target_rank = conf_ranks[target_idx]

            # Loss: encourage target_rank to be close to max_rank
            ranking_loss += F.mse_loss(target_rank.unsqueeze(0), max_rank.unsqueeze(0))

        ranking_loss = ranking_loss / language_logits.shape[0]

        total_loss = (
            self.classification_weight * classification_loss + self.ranking_weight * ranking_loss
        )

        components = {
            "classification": classification_loss,
            "ranking": ranking_loss,
        }

        return total_loss, components


class EraClassificationLoss(nn.Module):
    """
    Loss for era classification: similar to language detection.

    Uses soft ranking to rank era confidence scores.
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        ranking_weight: float = 0.5,
        regularization_strength: float = 1.0,
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.ranking_weight = ranking_weight
        self.regularization_strength = regularization_strength

    def forward(
        self,
        era_logits: torch.Tensor,
        era_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            era_logits: [batch, n_eras] era prediction logits
            era_targets: [batch] true era indices (or [batch, n_eras] one-hot)

        Returns:
            (total_loss, component_losses)
        """
        # Same structure as LanguageDetectionLoss
        if era_targets.dim() == 1:
            classification_loss = F.cross_entropy(era_logits, era_targets)
        else:
            classification_loss = F.cross_entropy(era_logits, era_targets.argmax(dim=-1))

        era_probs = F.softmax(era_logits, dim=-1)

        ranking_loss = torch.tensor(0.0, device=era_logits.device)
        for i in range(era_logits.shape[0]):
            conf_ranks = soft_rank_tensor(
                era_probs[i], regularization_strength=self.regularization_strength
            )

            if era_targets.dim() == 1:
                target_idx = int(era_targets[i].item())
            else:
                target_idx = int(era_targets[i].argmax().item())

            max_rank = conf_ranks.max()
            target_rank = conf_ranks[target_idx]
            ranking_loss += F.mse_loss(target_rank.unsqueeze(0), max_rank.unsqueeze(0))

        ranking_loss = ranking_loss / era_logits.shape[0]

        total_loss = (
            self.classification_weight * classification_loss + self.ranking_weight * ranking_loss
        )

        components = {
            "classification": classification_loss,
            "ranking": ranking_loss,
        }

        return total_loss, components


# ============================================================================
# Token Hygiene (auxiliary classification)
# ============================================================================


class TokenHygieneLoss(nn.Module):
    """
    Loss for token hygiene classification (multi-class).

    This is meant to teach the shared trunk to separate clean lexical tokens from
    common contamination classes (URLs, code, numbers, mojibake, etc.).
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        ranking_weight: float = 0.0,
        regularization_strength: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.ranking_weight = ranking_weight
        self.regularization_strength = regularization_strength
        self.ignore_index = int(ignore_index)

    def forward(
        self, hygiene_logits: torch.Tensor, hygiene_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Classification loss (cross-entropy).
        classification_loss = F.cross_entropy(
            hygiene_logits, hygiene_targets, ignore_index=self.ignore_index
        )

        # Optional ranking loss: target class should be top-ranked.
        ranking_loss = torch.tensor(0.0, device=hygiene_logits.device)
        if self.ranking_weight != 0.0:
            probs = F.softmax(hygiene_logits, dim=-1)
            for i in range(hygiene_logits.shape[0]):
                tgt = int(hygiene_targets[i].item())
                if tgt == self.ignore_index:
                    continue
                conf_ranks = soft_rank_tensor(
                    probs[i], regularization_strength=self.regularization_strength
                )
                max_rank = conf_ranks.max()
                target_rank = conf_ranks[tgt]
                ranking_loss += F.mse_loss(target_rank.unsqueeze(0), max_rank.unsqueeze(0))
            ranking_loss = ranking_loss / max(1, hygiene_logits.shape[0])

        total_loss = (
            self.classification_weight * classification_loss + self.ranking_weight * ranking_loss
        )

        components = {
            "classification": classification_loss,
            "ranking": ranking_loss,
        }
        return total_loss, components


# ============================================================================
# Unified Multi-Task Loss
# ============================================================================


class UnifiedMultiTaskLoss(nn.Module):
    """
    Unified loss for all tasks using rank-relax and AMOO.

    Combines:
    - ICF prediction loss
    - Text reduction loss
    - Temporal ICF loss
    - Language detection loss
    - Era classification loss

    Uses Aligned Multi-Objective Optimization (AMOO) for adaptive weighting.
    """

    def __init__(
        self,
        # Task weights
        icf_weight: float = 1.0,
        text_reduction_weight: float = 0.5,
        temporal_weight: float = 0.3,
        language_weight: float = 0.2,
        era_weight: float = 0.2,
        hygiene_weight: float = 0.2,
        # AMOO settings
        use_amoo: bool = True,
        amoo_curvature_weight: float = 0.1,
        # Loss-specific settings
        icf_spearman_weight: float = 10.0,
        icf_spearman_reg_strength: float = 1.0,
        icf_spearman_method: str = "sigmoid",  # Research: try "neural_sort" for sharper rankings
        icf_spearman_adaptive: bool = False,  # Research: adaptive regularization matches data scale
        ranking_reg_strength: float = 1.0,
    ):
        super().__init__()
        self.use_amoo = use_amoo
        self.amoo_curvature_weight = amoo_curvature_weight

        # Initialize task-specific losses
        self.icf_loss = ICFPredictionLoss(
            spearman_weight=icf_spearman_weight,
            spearman_reg_strength=icf_spearman_reg_strength,
            spearman_method=icf_spearman_method,
            spearman_adaptive=icf_spearman_adaptive,
        )
        self.text_reduction_loss = TextReductionLoss(
            regularization_strength=ranking_reg_strength,
        )
        self.temporal_loss = TemporalICFLoss(
            regularization_strength=ranking_reg_strength,
        )
        self.language_loss = LanguageDetectionLoss(
            regularization_strength=ranking_reg_strength,
        )
        self.era_loss = EraClassificationLoss(
            regularization_strength=ranking_reg_strength,
        )
        self.hygiene_loss = TokenHygieneLoss(
            regularization_strength=ranking_reg_strength,
        )

        # Task weights (can be learned if use_amoo=True)
        if use_amoo:
            # Learnable weights (initialized to given values)
            self.register_parameter("icf_weight", nn.Parameter(torch.tensor(icf_weight)))
            self.register_parameter(
                "text_reduction_weight", nn.Parameter(torch.tensor(text_reduction_weight))
            )
            self.register_parameter("temporal_weight", nn.Parameter(torch.tensor(temporal_weight)))
            self.register_parameter("language_weight", nn.Parameter(torch.tensor(language_weight)))
            self.register_parameter("era_weight", nn.Parameter(torch.tensor(era_weight)))
            self.register_parameter("hygiene_weight", nn.Parameter(torch.tensor(hygiene_weight)))
        else:
            # Fixed weights
            self.register_buffer("icf_weight", torch.tensor(icf_weight))
            self.register_buffer("text_reduction_weight", torch.tensor(text_reduction_weight))
            self.register_buffer("temporal_weight", torch.tensor(temporal_weight))
            self.register_buffer("language_weight", torch.tensor(language_weight))
            self.register_buffer("era_weight", torch.tensor(era_weight))
            self.register_buffer("hygiene_weight", torch.tensor(hygiene_weight))

    def forward(
        self,
        # ICF prediction
        icf_predictions: Optional[torch.Tensor] = None,
        icf_targets: Optional[torch.Tensor] = None,
        icf_pairs: Optional[torch.Tensor] = None,
        # Text reduction
        word_icf_scores: Optional[torch.Tensor] = None,
        original_embedding: Optional[torch.Tensor] = None,
        reduced_embedding: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
        # Temporal
        current_predictions: Optional[torch.Tensor] = None,
        current_targets: Optional[torch.Tensor] = None,
        historical_predictions: Optional[Dict[str, torch.Tensor]] = None,
        historical_targets: Optional[Dict[str, torch.Tensor]] = None,
        # Language
        language_logits: Optional[torch.Tensor] = None,
        language_targets: Optional[torch.Tensor] = None,
        # Era
        era_logits: Optional[torch.Tensor] = None,
        era_targets: Optional[torch.Tensor] = None,
        # Hygiene
        hygiene_logits: Optional[torch.Tensor] = None,
        hygiene_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute unified multi-task loss.

        Returns:
            (total_loss, diagnostics) where diagnostics contains:
            - task_losses: Dict of individual task losses
            - task_weights: Dict of current task weights
            - components: Dict of loss components per task
        """
        task_losses = {}
        components = {}

        # ICF prediction
        if icf_predictions is not None and icf_targets is not None:
            icf_loss, icf_components = self.icf_loss(icf_predictions, icf_targets, icf_pairs)
            task_losses["icf"] = icf_loss
            components["icf"] = icf_components

        # Text reduction
        if (
            word_icf_scores is not None
            and original_embedding is not None
            and reduced_embedding is not None
            and target_length is not None
        ):
            tr_loss, tr_components = self.text_reduction_loss(
                word_icf_scores, original_embedding, reduced_embedding, target_length
            )
            task_losses["text_reduction"] = tr_loss
            components["text_reduction"] = tr_components

        # Temporal
        if current_predictions is not None and current_targets is not None:
            temp_loss, temp_components = self.temporal_loss(
                current_predictions, current_targets, historical_predictions, historical_targets
            )
            task_losses["temporal"] = temp_loss
            components["temporal"] = temp_components

        # Language
        if language_logits is not None and language_targets is not None:
            lang_loss, lang_components = self.language_loss(language_logits, language_targets)
            task_losses["language"] = lang_loss
            components["language"] = lang_components

        # Era
        if era_logits is not None and era_targets is not None:
            era_loss_val, era_components = self.era_loss(era_logits, era_targets)
            task_losses["era"] = era_loss_val
            components["era"] = era_components

        # Hygiene
        if hygiene_logits is not None and hygiene_targets is not None:
            hyg_loss, hyg_components = self.hygiene_loss(hygiene_logits, hygiene_targets)
            task_losses["hygiene"] = hyg_loss
            components["hygiene"] = hyg_components

        # Compute weighted sum
        if self.use_amoo and len(task_losses) > 1:
            # AMOO: adaptive weighting based on gradient alignment
            # Simplified: use current weights (can be enhanced with gradient analysis)
            weights = {
                "icf": self.icf_weight,
                "text_reduction": self.text_reduction_weight,
                "temporal": self.temporal_weight,
                "language": self.language_weight,
                "era": self.era_weight,
                "hygiene": self.hygiene_weight,
            }

            # Normalize weights
            total_weight = sum(w for k, w in weights.items() if k in task_losses)
            if total_weight > 0:
                weights = {k: w / total_weight for k, w in weights.items() if k in task_losses}

            total_loss = sum(weights[k] * task_losses[k] for k in task_losses.keys())
        else:
            # Fixed weights
            weights = {
                "icf": self.icf_weight,
                "text_reduction": self.text_reduction_weight,
                "temporal": self.temporal_weight,
                "language": self.language_weight,
                "era": self.era_weight,
                "hygiene": self.hygiene_weight,
            }

            # Compute total loss with proper type handling
            total_loss = torch.tensor(0.0, device=list(task_losses.values())[0].device)
            for k in task_losses.keys():
                weight = weights.get(k, 0.0)
                if isinstance(weight, torch.Tensor):
                    weight_val = weight.item()
                else:
                    weight_val = float(weight) if isinstance(weight, (int, float)) else 0.0
                total_loss = total_loss + weight_val * task_losses[k]

        diagnostics = {
            "task_losses": {k: v.item() for k, v in task_losses.items()},
            "task_weights": {
                k: w.item() if isinstance(w, torch.Tensor) else w for k, w in weights.items()
            },
            "components": {
                k: {ck: cv.item() if isinstance(cv, torch.Tensor) else cv for ck, cv in v.items()}
                for k, v in components.items()
            },
        }

        return total_loss, diagnostics
