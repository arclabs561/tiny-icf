"""Shared training utilities to reduce code duplication across training scripts."""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr

from tiny_icf.loss import CombinedLoss
from tiny_icf.loss_monitoring import (
    detect_loss_imbalance,
)


def generate_ranking_pairs(
    targets: torch.Tensor,
    n_pairs: int,
    min_diff: float = 0.05,
    use_weighted_sampling: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate pairs for ranking loss with sampling-based rewards.

    Uses weighted sampling: pairs with larger ICF differences are sampled
    with higher probability, providing stronger learning signal.

    Args:
        targets: [Batch, 1] or [Batch] ground truth ICF scores
        n_pairs: Number of pairs to generate
        min_diff: Minimum ICF difference required (default: 0.05)
        use_weighted_sampling: If True, sample pairs weighted by ICF difference

    Returns:
        (pairs, diffs) where:
        - pairs: [N_pairs, 2] tensor of indices (i, j) where target[i] < target[j]
        - diffs: [N_pairs] tensor of actual ICF differences for weighted loss
    """
    batch_size = len(targets)
    if batch_size < 2:
        empty_pairs = torch.empty((0, 2), dtype=torch.long, device=targets.device)
        empty_diffs = torch.empty((0,), dtype=targets.dtype, device=targets.device)
        return empty_pairs, empty_diffs

    # Handle both [Batch, 1] and [Batch] shapes
    if targets.dim() > 1 and targets.size(1) == 1:
        targets_flat = targets.squeeze(1)  # [Batch]
    else:
        targets_flat = targets  # Already [Batch]

    # Efficient sampling approach (avoid O(batch^2) diff matrices):
    # - sort targets
    # - sample candidate position pairs in sorted order
    # - filter by min_diff
    # - optionally weight by diff
    sorted_pos = torch.argsort(targets_flat)
    sorted_targets = targets_flat[sorted_pos]

    cur_min_diff = float(min_diff)
    for attempt in range(5):
        n_cand = max(int(n_pairs) * 50, 4096) * (2**attempt)

        a = torch.randint(0, batch_size, (n_cand,), device=targets.device)
        b = torch.randint(0, batch_size, (n_cand,), device=targets.device)
        lo = torch.minimum(a, b)
        hi = torch.maximum(a, b)
        neq = lo != hi
        lo = lo[neq]
        hi = hi[neq]
        if lo.numel() == 0:
            cur_min_diff *= 0.5
            continue

        diffs = sorted_targets[hi] - sorted_targets[lo]
        keep = diffs >= cur_min_diff
        if keep.any():
            lo_k = lo[keep]
            hi_k = hi[keep]
            diffs_k = diffs[keep]
            pairs_k = torch.stack([sorted_pos[lo_k], sorted_pos[hi_k]], dim=1)

            if pairs_k.size(0) > n_pairs:
                if use_weighted_sampling:
                    probs = torch.softmax(diffs_k * 5.0, dim=0)
                    idx = torch.multinomial(probs, num_samples=n_pairs, replacement=False)
                else:
                    idx = torch.randperm(pairs_k.size(0), device=targets.device)[:n_pairs]
                return pairs_k[idx], diffs_k[idx]

            return pairs_k, diffs_k

        cur_min_diff *= 0.5

    # Final fallback: monotone pairs with diff > 0.
    n_cand = max(int(n_pairs) * 10, 1024)
    a = torch.randint(0, batch_size, (n_cand,), device=targets.device)
    b = torch.randint(0, batch_size, (n_cand,), device=targets.device)
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    neq = lo != hi
    lo = lo[neq]
    hi = hi[neq]
    if lo.numel() > 0:
        diffs = sorted_targets[hi] - sorted_targets[lo]
        keep = diffs > 0
        if keep.any():
            lo = lo[keep]
            hi = hi[keep]
            diffs = diffs[keep]
            pairs = torch.stack([sorted_pos[lo], sorted_pos[hi]], dim=1)
            if pairs.size(0) > n_pairs:
                idx = torch.randperm(pairs.size(0), device=targets.device)[:n_pairs]
                return pairs[idx], diffs[idx]
            return pairs, diffs

    empty_pairs = torch.empty((0, 2), dtype=torch.long, device=targets.device)
    empty_diffs = torch.empty((0,), dtype=targets.dtype, device=targets.device)
    return empty_pairs, empty_diffs


def prepare_batch(batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare batch for training/validation.
    Handles both dict and tuple batch formats.

    Returns:
        (words, targets) tensors on device
    """
    if isinstance(batch, dict):
        words = batch["bytes"].to(device)
        targets = batch["icf"].to(device)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
    else:
        words, targets = batch
        words = words.to(device)
        targets = targets.to(device)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

    return words, targets


def compute_spearman_safe(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Compute Spearman correlation with safe handling of edge cases.

    Returns:
        Spearman correlation (0.0 if computation fails)
    """
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0

    pred_std = np.std(predictions)
    target_std = np.std(targets)

    if pred_std == 0 or target_std == 0:
        return 0.0

    try:
        corr, _ = spearmanr(predictions, targets)
        if np.isnan(corr) or np.isinf(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0


def train_epoch_unified(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    *,
    n_pairs: int = 16,
    min_diff: float = 0.05,
    use_weighted_sampling: bool = True,
    clip_grad_norm: Optional[float] = 1.0,
    check_collapse: bool = False,
    collapse_threshold: float = 0.01,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """
    Unified training epoch function with consistent behavior.

    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        n_pairs: Number of ranking pairs to generate
        min_diff: Minimum ICF difference for pairs
        use_weighted_sampling: Use weighted sampling for pairs
        clip_grad_norm: Gradient clipping norm (None to disable)
        check_collapse: Check for model collapse
        collapse_threshold: Threshold for collapse detection

    Returns:
        Dictionary with 'loss' and 'spearman_corr'
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    predictions_list = []
    targets_list = []

    print(f"Starting training loop, dataloader length: {len(dataloader)}", flush=True)
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)):
        if batch_idx == 0:
            print("Processing first batch...", flush=True)
        words, targets = prepare_batch(batch, device)
        if batch_idx == 0:
            print(
                f"Batch prepared: words shape={words.shape}, targets shape={targets.shape}",
                flush=True,
            )

        # Skip empty batches
        if len(words) == 0:
            continue

        optimizer.zero_grad()

        # Forward pass (with mixed precision if enabled)
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                predictions = model(words)

                # Collapse detection
                if check_collapse and len(predictions) > 1:
                    pred_std = predictions.std().item()
                    if pred_std < collapse_threshold:
                        raise RuntimeError(
                            f"Model collapsed: prediction std={pred_std:.6f} < {collapse_threshold}. "
                            "All predictions are too similar."
                        )

                # Generate ranking pairs
                pairs, pair_target_diffs = generate_ranking_pairs(
                    targets.squeeze(1),
                    n_pairs=min(len(targets), n_pairs),
                    min_diff=min_diff,
                    use_weighted_sampling=use_weighted_sampling,
                )

                if pairs is not None and len(pairs) > 0:
                    pairs = pairs.to(device)
                    pair_target_diffs = (
                        pair_target_diffs.to(device) if pair_target_diffs is not None else None
                    )

                # Compute loss
                loss = criterion(
                    predictions,
                    targets,
                    pairs=pairs,
                    pair_target_diffs=pair_target_diffs,
                )

            # Track loss components for monitoring
            loss_components = {}
            if hasattr(criterion, "get_component_stats"):
                component_stats = criterion.get_component_stats()
                if component_stats:
                    loss_components = {
                        "huber": component_stats.get("huber_mean", 0),
                        "ranking": component_stats.get("ranking_mean", 0),
                    }
                    if criterion.use_neural_ndcg:
                        loss_components["neural_ndcg"] = component_stats.get("neural_ndcg_mean", 0)
                    if criterion.use_listwise_ranking:
                        loss_components["listwise"] = component_stats.get("listwise_mean", 0)

            # Check for imbalance (every 10 batches)
            if n_batches % 10 == 0 and loss_components:
                is_imbalanced, dominant = detect_loss_imbalance(loss_components, threshold=0.7)
                if is_imbalanced:
                    print(f"⚠️  Warning: Loss imbalance detected. Dominant components: {dominant}")

            # Backward pass with scaler
            scaler.scale(loss).backward()

            # Gradient clipping (unscale first)
            if clip_grad_norm is not None and clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward/backward
            predictions = model(words)

            # Collapse detection
            if check_collapse and len(predictions) > 1:
                pred_std = predictions.std().item()
                if pred_std < collapse_threshold:
                    raise RuntimeError(
                        f"Model collapsed: prediction std={pred_std:.6f} < {collapse_threshold}. "
                        "All predictions are too similar."
                    )

            # Generate ranking pairs
            pairs, pair_target_diffs = generate_ranking_pairs(
                targets.squeeze(1),
                n_pairs=min(len(targets), n_pairs),
                min_diff=min_diff,
                use_weighted_sampling=use_weighted_sampling,
            )

            if pairs is not None and len(pairs) > 0:
                pairs = pairs.to(device)
                pair_target_diffs = (
                    pair_target_diffs.to(device) if pair_target_diffs is not None else None
                )

            # Compute loss
            loss = criterion(
                predictions,
                targets,
                pairs=pairs,
                pair_target_diffs=pair_target_diffs,
            )

            # Track loss components for monitoring
            loss_components = {}
            if hasattr(criterion, "get_component_stats"):
                component_stats = criterion.get_component_stats()
                if component_stats:
                    loss_components = {
                        "huber": component_stats.get("huber_mean", 0),
                        "ranking": component_stats.get("ranking_mean", 0),
                    }
                    if criterion.use_neural_ndcg:
                        loss_components["neural_ndcg"] = component_stats.get("neural_ndcg_mean", 0)
                    if criterion.use_listwise_ranking:
                        loss_components["listwise"] = component_stats.get("listwise_mean", 0)

            # Check for imbalance (every 10 batches)
            if n_batches % 10 == 0 and loss_components:
                is_imbalanced, dominant = detect_loss_imbalance(loss_components, threshold=0.7)
                if is_imbalanced:
                    print(f"⚠️  Warning: Loss imbalance detected. Dominant components: {dominant}")

            # Backward pass
            loss.backward()

            # Gradient clipping
            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        n_batches += 1

        predictions_list.append(predictions.detach().cpu().numpy())
        targets_list.append(targets.detach().cpu().numpy())

    if n_batches == 0:
        return {"loss": 0.0, "spearman_corr": 0.0}

    avg_loss = total_loss / n_batches

    # Compute Spearman correlation
    all_preds = np.concatenate(predictions_list).flatten()
    all_targets = np.concatenate(targets_list).flatten()

    spearman_corr = compute_spearman_safe(all_preds, all_targets)

    return {
        "loss": avg_loss,
        "spearman_corr": spearman_corr,
    }


def validate_unified(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    *,
    n_pairs: int = 16,
    min_diff: float = 0.05,
    use_weighted_sampling: bool = True,
) -> Dict[str, float]:
    """
    Unified validation function with consistent behavior.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use
        n_pairs: Number of ranking pairs to generate
        min_diff: Minimum ICF difference for pairs
        use_weighted_sampling: Use weighted sampling for pairs

    Returns:
        Dictionary with 'loss', 'spearman_corr', and 'mae'
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    predictions_list = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            words, targets = prepare_batch(batch, device)

            # Skip empty batches
            if len(words) == 0:
                continue

            predictions = model(words)

            # Generate ranking pairs
            pairs, pair_target_diffs = generate_ranking_pairs(
                targets.squeeze(1),
                n_pairs=min(len(targets), n_pairs),
                min_diff=min_diff,
                use_weighted_sampling=use_weighted_sampling,
            )

            if pairs is not None and len(pairs) > 0:
                pairs = pairs.to(device)
                pair_target_diffs = (
                    pair_target_diffs.to(device) if pair_target_diffs is not None else None
                )

            # Compute loss
            loss = criterion(
                predictions,
                targets,
                pairs=pairs,
                pair_target_diffs=pair_target_diffs,
            )

            total_loss += loss.item()
            n_batches += 1

            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    if n_batches == 0:
        return {"loss": 0.0, "spearman_corr": 0.0, "mae": 0.0}

    avg_loss = total_loss / n_batches

    # Compute metrics
    all_preds = np.concatenate(predictions_list).flatten()
    all_targets = np.concatenate(targets_list).flatten()

    spearman_corr = compute_spearman_safe(all_preds, all_targets)
    mae = float(np.mean(np.abs(all_preds - all_targets)))  # Convert numpy float to Python float

    return {
        "loss": float(avg_loss),
        "spearman_corr": float(spearman_corr),
        "mae": mae,
    }


def save_checkpoint(
    checkpoint_path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    best_spearman: float,
    best_model_state: Optional[Dict],
    history: Dict,
    args: Dict,
) -> None:
    """Save training checkpoint with all necessary state."""
    from pathlib import Path

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_spearman": best_spearman,
        "best_model_state": best_model_state,
        "history": history,
        "args": args,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Atomic write: save to temp file first, then rename
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_suffix(".tmp")

    try:
        torch.save(checkpoint, temp_path)
        # Atomic rename (works on same filesystem)
        temp_path.replace(checkpoint_path)
    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise e


def load_checkpoint(
    checkpoint_path,
    device: torch.device,
) -> Optional[Dict]:
    """Load checkpoint with validation and error handling."""
    from pathlib import Path

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Validate checkpoint structure
        required_keys = ["epoch", "model_state_dict", "optimizer_state_dict"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            print(f"⚠️  Checkpoint missing required keys: {missing_keys}")
            return None

        return checkpoint
    except Exception as e:
        print(f"⚠️  Could not load checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return None


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_type: str = "AdamW",
) -> torch.optim.Optimizer:
    """Create optimizer with consistent defaults."""
    if optimizer_type == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    mode: str = "max",
    factor: float = 0.5,
    patience: int = 10,
    min_lr: float = 1e-6,
    scheduler_type: str = "plateau",  # "plateau", "cosine", "onecycle"
    T_max: int = 100,  # For cosine annealing
    eta_min: float = 1e-6,  # For cosine annealing
    max_lr: float = 1e-3,  # For onecycle
    total_steps: int = 1000,  # For onecycle
) -> torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    Create learning rate scheduler with consistent defaults.

    Args:
        optimizer: Optimizer to schedule
        mode: For ReduceLROnPlateau: "max" or "min"
        factor: For ReduceLROnPlateau: LR reduction factor
        patience: For ReduceLROnPlateau: Patience epochs
        min_lr: Minimum learning rate
        scheduler_type: "plateau", "cosine", or "onecycle"
        T_max: For cosine: Maximum number of iterations
        eta_min: For cosine: Minimum learning rate
        max_lr: For onecycle: Maximum learning rate
        total_steps: For onecycle: Total number of training steps

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        return CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )
    elif scheduler_type == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR

        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.3,  # 30% warmup
            anneal_strategy="cos",
        )
    else:  # "plateau" (default)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
