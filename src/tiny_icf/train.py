"""Training script for Universal ICF model."""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiny_icf.checkpoint import load_model
from tiny_icf.data import (
    WordICFDataset,
    compute_normalized_icf,
    load_frequency_list,
    stratified_sample,
)
from tiny_icf.loss import CombinedLoss, ranking_loss
from tiny_icf.model import UniversalICF
from tiny_icf.predict import word_to_bytes
from tiny_icf.synthetic_oov import choose_bases, generate_composed_words, generate_gibberish_words
from tiny_icf.synthetic_training import SyntheticOOVConfig, generate_synthetic_oov_pairs


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    # Efficient sampling approach (avoid O(batch^2) Python loops):
    # - sort targets
    # - sample candidate position pairs in sorted order
    # - filter by min_diff
    # - optionally weight by diff
    sorted_pos = torch.argsort(targets_flat)
    sorted_targets = targets_flat[sorted_pos]

    cur_min_diff = float(min_diff)
    for attempt in range(5):
        n_cand = max(int(n_pairs) * 50, 4096) * (2**attempt)

        # Sample two positions and order them so lo < hi.
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


@dataclass(frozen=True)
class OOVAuxRanking:
    """
    Auxiliary raw-score ranking constraints for OOV calibration.

    We enforce a chain:
      raw_common < raw_composed < raw_gibberish

    using pairwise ranking_loss in *raw_output* space. Synthetic batches are
    optionally run with BatchNorm frozen (eval-mode) to avoid polluting running stats.
    """

    composed_pool: torch.Tensor  # [pool, max_length] on CPU
    gibberish_pool: torch.Tensor  # [pool, max_length] on CPU
    n_per_batch: int = 32
    weight: float = 0.05
    margin_common: float = 0.05
    margin_gibberish: float = 0.05
    n_common_anchors: int = 32
    every: int = 1
    freeze_bn: bool = True


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_loss_components: bool = False,
    check_collapse: bool = True,
    raw_overflow_penalty: float = 0.0,
    oov_aux: OOVAuxRanking | None = None,
    oov_aux_gen: torch.Generator | None = None,
) -> tuple[float, dict]:
    """
    Train for one epoch with ranking loss.

    Returns:
        (average_loss, metrics_dict) where metrics_dict contains:
        - huber_loss: Average Huber loss
        - ranking_loss: Average ranking loss (if pairs provided)
        - pred_std: Standard deviation of predictions (for collapse detection)
        - pred_range: [min, max] of predictions
    """
    model.train()
    total_loss = 0.0
    total_huber = 0.0
    total_ranking = 0.0
    n_batches = 0
    all_predictions = []
    total_oov_aux = 0.0
    n_oov_aux = 0
    batch_idx = 0

    for byte_tensors, icf_targets in tqdm(dataloader, desc="Training"):
        byte_tensors = byte_tensors.to(device)
        icf_targets = icf_targets.to(device)

        optimizer.zero_grad()
        need_features = raw_overflow_penalty > 0.0 or oov_aux is not None
        if need_features:
            predictions, features = model(byte_tensors, return_features=True)  # type: ignore[misc]
            raw_output = features.get("raw_output", predictions)
        else:
            predictions = model(byte_tensors)
            raw_output = None

        # Collapse detection: check prediction variance
        if check_collapse:
            pred_std = predictions.std().item()
            if pred_std < 0.01:
                raise RuntimeError(
                    f"Model collapsed: prediction std={pred_std:.6f} < 0.01. "
                    "All predictions are too similar. Check model initialization and loss function."
                )

        # Collect predictions for analysis
        all_predictions.append(predictions.detach().cpu())

        # Generate pairs for ranking loss with weighted sampling
        n_pairs = min(len(icf_targets), 32)
        pairs, pair_diffs = generate_ranking_pairs(
            icf_targets, n_pairs, min_diff=0.05, use_weighted_sampling=True
        )

        # Compute loss with ranking pairs and smooth rewards
        loss = criterion(
            predictions,
            icf_targets,
            pairs=pairs,
            pair_target_diffs=pair_diffs,
            smooth_ranking=True,
        )

        # Optional penalty: discourage raw_output from exceeding 1.0 (pre-clamp).
        # This reduces the model's tendency to saturate at exactly 1.0 for plausible OOV words.
        if raw_overflow_penalty > 0.0 and raw_output is not None:
            overflow = torch.relu(raw_output - 1.0)
            overflow_loss = (overflow * overflow).mean()
            loss = loss + raw_overflow_penalty * overflow_loss

        # Optional auxiliary OOV ranking constraints (raw_output space).
        if oov_aux is not None and raw_output is not None and oov_aux.weight > 0.0:
            if oov_aux_gen is None:
                oov_aux_gen = torch.Generator(device="cpu")
                oov_aux_gen.manual_seed(1337)

            if oov_aux.every <= 1 or (batch_idx % int(oov_aux.every) == 0):
                pool_size = int(oov_aux.composed_pool.size(0))
                n_aux = int(min(oov_aux.n_per_batch, pool_size))
                if n_aux > 0:
                    # Choose "common anchors" from this batch (lowest ICF targets).
                    n_common = int(min(oov_aux.n_common_anchors, len(icf_targets)))
                    common_idx = torch.argsort(icf_targets.squeeze(), dim=0)[:n_common]
                    # Sample with replacement to match n_aux.
                    common_pick = torch.randint(
                        0, n_common, (n_aux,), generator=oov_aux_gen, device="cpu"
                    )
                    raw_common = raw_output[common_idx][common_pick].to(device)

                    # Sample synthetic pairs from the precomputed pools.
                    pool_idx = torch.randint(
                        0, pool_size, (n_aux,), generator=oov_aux_gen, device="cpu"
                    )
                    x_comp = oov_aux.composed_pool[pool_idx].to(device)
                    x_gib = oov_aux.gibberish_pool[pool_idx].to(device)

                    # Freeze BN/dropout for synthetic pass if requested (avoids BN stat pollution).
                    was_training = model.training
                    if oov_aux.freeze_bn:
                        model.eval()
                    pred_comp, feat_comp = model(x_comp, return_features=True)  # type: ignore[misc]
                    pred_gib, feat_gib = model(x_gib, return_features=True)  # type: ignore[misc]
                    if was_training:
                        model.train()

                    raw_comp = feat_comp.get("raw_output", pred_comp)
                    raw_gib = feat_gib.get("raw_output", pred_gib)

                    # Chain constraints: raw_common < raw_comp < raw_gib
                    loss_common = ranking_loss(
                        raw_common,
                        raw_comp,
                        margin=float(oov_aux.margin_common),
                        smooth=True,
                    )
                    loss_gib = ranking_loss(
                        raw_comp,
                        raw_gib,
                        margin=float(oov_aux.margin_gibberish),
                        smooth=True,
                    )
                    aux_loss = 0.5 * (loss_common + loss_gib)
                    loss = loss + float(oov_aux.weight) * aux_loss

                    total_oov_aux += float(aux_loss.detach().cpu().item())
                    n_oov_aux += 1

        # Extract loss components if logging enabled
        if log_loss_components:
            # Compute Huber loss separately
            from tiny_icf.loss import huber_loss

            huber = huber_loss(predictions, icf_targets, delta=0.1)
            total_huber += huber.item()

            # Compute ranking loss separately if pairs exist
            if len(pairs) > 0:
                idx1, idx2 = pairs[:, 0], pairs[:, 1]
                rank = ranking_loss(
                    predictions[idx1],
                    predictions[idx2],
                    margin=0.1,
                    target_diff=pair_diffs,
                    smooth=True,
                )
                total_ranking += rank.item()

        loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        batch_idx += 1

    # Compute metrics
    all_preds = torch.cat(all_predictions, dim=0)
    metrics = {
        "pred_std": all_preds.std().item(),
        "pred_min": all_preds.min().item(),
        "pred_max": all_preds.max().item(),
        "pred_mean": all_preds.mean().item(),
    }

    if n_oov_aux > 0:
        metrics["oov_aux_loss"] = total_oov_aux / n_oov_aux

    if log_loss_components:
        metrics["huber_loss"] = total_huber / n_batches if n_batches > 0 else 0.0
        metrics["ranking_loss"] = total_ranking / n_batches if n_batches > 0 else 0.0

    return total_loss / n_batches if n_batches > 0 else 0.0, metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for byte_tensors, icf_targets in tqdm(dataloader, desc="Validating"):
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)

            predictions = model(byte_tensors)
            loss = criterion(predictions, icf_targets)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train Universal ICF model")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=20, help="Max word length")
    parser.add_argument("--augment-prob", type=float, default=0.1, help="Augmentation probability")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (0 disables multiprocessing).",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Optional: initialize from an existing checkpoint (fine-tune).",
    )
    parser.add_argument(
        "--raw-overflow-penalty",
        type=float,
        default=0.0,
        help="Optional penalty weight for relu(raw_output-1)^2 to reduce clamp saturation (0 disables).",
    )
    parser.add_argument(
        "--synthetic-oov",
        type=int,
        default=0,
        help="Optional: add N composed-OOV + N gibberish samples to the training set (0 disables).",
    )
    parser.add_argument(
        "--synthetic-oov-common-k",
        type=int,
        default=10000,
        help="How many most-common words to draw stems from when generating synthetic OOV.",
    )
    parser.add_argument(
        "--oov-aux-n",
        type=int,
        default=0,
        help="Auxiliary OOV ranking: number of synthetic pairs per batch (0 disables).",
    )
    parser.add_argument(
        "--oov-aux-weight",
        type=float,
        default=0.05,
        help="Auxiliary OOV ranking loss weight.",
    )
    parser.add_argument(
        "--oov-aux-margin-common",
        type=float,
        default=0.05,
        help="Auxiliary margin for common < composed (raw_output space).",
    )
    parser.add_argument(
        "--oov-aux-margin-gibberish",
        type=float,
        default=0.05,
        help="Auxiliary margin for composed < gibberish (raw_output space).",
    )
    parser.add_argument(
        "--oov-aux-common-anchors",
        type=int,
        default=32,
        help="How many low-ICF anchors to sample per batch for common < composed.",
    )
    parser.add_argument(
        "--oov-aux-pool",
        type=int,
        default=5000,
        help="Precomputed synthetic pool size (composed + gibberish) for aux ranking.",
    )
    parser.add_argument(
        "--oov-aux-every",
        type=int,
        default=1,
        help="Apply auxiliary OOV loss every N batches (>=1).",
    )
    parser.add_argument(
        "--oov-aux-no-freeze-bn",
        action="store_true",
        help="Do NOT freeze BatchNorm/dropout during synthetic aux forward pass (default freezes).",
    )
    parser.add_argument(
        "--icf-mode",
        type=str,
        default="log",
        choices=["log", "rank"],
        help="Target definition: 'log' (corpus ICF) or 'rank' (corpus-invariant quantile)",
    )
    parser.add_argument("--output", type=Path, default=Path("model.pt"), help="Output model path")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument(
        "--use-attention",
        action="store_true",
        help="Enable multi-head self-attention in UniversalICF (better for long-range composition)",
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=3,
        help="Number of attention heads (only used with --use-attention)",
    )
    parser.add_argument(
        "--output-activation",
        type=str,
        default="clamp",
        choices=["clamp", "clamp_ste", "sigmoid"],
        help="Output activation for UniversalICF",
    )
    parser.add_argument(
        "--sigmoid-temperature",
        type=float,
        default=1.0,
        help="Sigmoid temperature (only used with --output-activation=sigmoid)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(42)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print("Random seed: 42 (reproducible)")

    # Load data
    print("Loading frequency list...")
    try:
        word_counts, total_tokens = load_frequency_list(args.data)
        print(f"Loaded {len(word_counts)} words, {total_tokens:,} total tokens")
    except Exception as e:
        print(f"Error loading frequency list: {e}")
        raise

    # Compute ICF
    print("Computing normalized ICF...")
    try:
        word_icf = compute_normalized_icf(word_counts, total_tokens, mode=args.icf_mode)
        icf_vals = list(word_icf.values())
        if icf_vals:
            print(
                f"ICF stats ({args.icf_mode}): "
                f"min={min(icf_vals):.4f} max={max(icf_vals):.4f} mean={sum(icf_vals)/len(icf_vals):.4f}"
            )
    except Exception as e:
        print(f"Error computing ICF: {e}")
        raise

    # Stratified sampling
    print("Creating stratified sample...")
    try:
        samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
        print(f"Sampled {len(samples)} word-ICF pairs")
    except Exception as e:
        print(f"Error in stratified sampling: {e}")
        raise

    # Split train/val (80/20)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Precompute a synthetic OOV pool for auxiliary ranking constraints.
    oov_aux: OOVAuxRanking | None = None
    oov_aux_gen: torch.Generator | None = None
    if int(args.oov_aux_n) > 0 and float(args.oov_aux_weight) > 0.0:
        sorted_by_count = sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)
        common_k = min(int(args.synthetic_oov_common_k), len(sorted_by_count))
        common_words = [w for w, _ in sorted_by_count[:common_k]]
        bases = choose_bases(common_words, seed=42, max_bases=min(5000, len(common_words)))
        real_set = set(word_counts.keys())

        pool_n = int(max(1, args.oov_aux_pool))
        composed_words = generate_composed_words(bases=bases, real_set=real_set, n=pool_n, seed=42)
        gibberish_words = generate_gibberish_words(
            lengths=[len(w) for w in composed_words],
            real_set=real_set,
            n=len(composed_words),
            seed=43,
        )

        # Precompute byte tensors once (CPU) to keep training overhead bounded.
        composed_pool = torch.cat(
            [word_to_bytes(w, max_length=int(args.max_length)) for w in composed_words], dim=0
        )
        gibberish_pool = torch.cat(
            [word_to_bytes(w, max_length=int(args.max_length)) for w in gibberish_words], dim=0
        )
        oov_aux = OOVAuxRanking(
            composed_pool=composed_pool,
            gibberish_pool=gibberish_pool,
            n_per_batch=int(args.oov_aux_n),
            weight=float(args.oov_aux_weight),
            margin_common=float(args.oov_aux_margin_common),
            margin_gibberish=float(args.oov_aux_margin_gibberish),
            n_common_anchors=int(args.oov_aux_common_anchors),
            every=int(max(1, args.oov_aux_every)),
            freeze_bn=not bool(args.oov_aux_no_freeze_bn),
        )
        oov_aux_gen = torch.Generator(device="cpu")
        oov_aux_gen.manual_seed(1337)

        print(
            "Aux OOV ranking enabled: "
            f"pool={int(composed_pool.size(0)):,} "
            f"n_per_batch={oov_aux.n_per_batch} "
            f"weight={oov_aux.weight:g} "
            f"every={oov_aux.every} "
            f"freeze_bn={oov_aux.freeze_bn}"
        )

    # Optional synthetic OOV augmentation (training only).
    if int(args.synthetic_oov) > 0:
        synth_cfg = SyntheticOOVConfig(
            n_composed=int(args.synthetic_oov),
            common_k=int(args.synthetic_oov_common_k),
            seed=42,
            max_len=int(args.max_length),
        )
        synthetic_pairs = generate_synthetic_oov_pairs(
            word_counts=word_counts, word_icf=word_icf, config=synth_cfg
        )
        if synthetic_pairs:
            train_samples = list(train_samples) + synthetic_pairs
            synth_targets = [y for _, y in synthetic_pairs]
            print(
                f"Added synthetic OOV pairs: {len(synthetic_pairs):,} "
                f"(composed={int(args.synthetic_oov):,}, gibberish={int(args.synthetic_oov):,})"
            )
            print(
                f"  synthetic target stats: "
                f"min={min(synth_targets):.4f} p50={float(np.median(synth_targets)):.4f} max={max(synth_targets):.4f}"
            )

    # Datasets
    train_dataset = WordICFDataset(
        train_samples, max_length=args.max_length, augment_prob=args.augment_prob
    )
    val_dataset = WordICFDataset(val_samples, max_length=args.max_length, augment_prob=0.0)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(args.num_workers) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(args.num_workers) > 0,
    )

    # Model
    try:
        if args.init_from is not None:
            model, _checkpoint = load_model(args.init_from, device=device)
            if not isinstance(model, UniversalICF):
                raise ValueError(
                    f"--init-from must load a UniversalICF checkpoint, got: {type(model).__name__}"
                )
            print(f"Loaded init-from checkpoint: {args.init_from}")
            print(f"Model parameters: {model.count_parameters():,}")
            mean_icf = float("nan")
        else:
            model = UniversalICF(
                use_attention=args.use_attention,
                attention_heads=args.attention_heads,
                output_activation=args.output_activation,
                sigmoid_temperature=args.sigmoid_temperature,
            ).to(device)
            # Initialize weights properly
            # Estimate mean ICF from data for better initialization
            sample_icf_values = [icf for _, icf in samples]
            mean_icf = sum(sample_icf_values) / len(sample_icf_values) if sample_icf_values else 0.4
            model.init_weights(mean_icf=mean_icf)
            print(f"Model parameters: {model.count_parameters():,}")
            print(f"Initialized with mean ICF bias: {mean_icf:.4f}")
    except Exception as e:
        print(f"Error creating model: {e}")
        raise

    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            raw_overflow_penalty=float(args.raw_overflow_penalty),
            oov_aux=oov_aux,
            oov_aux_gen=oov_aux_gen,
        )
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        if train_metrics:
            print(
                "Train prediction stats: "
                f"std={train_metrics['pred_std']:.4f} "
                f"range=[{train_metrics['pred_min']:.4f}, {train_metrics['pred_max']:.4f}] "
                f"mean={train_metrics['pred_mean']:.4f}"
            )
            if "oov_aux_loss" in train_metrics:
                print(f"Train oov-aux loss: {train_metrics['oov_aux_loss']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                # Ensure output directory exists
                args.output.parent.mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "model_type": "UniversalICF",
                    "model_kwargs": {
                        "use_attention": bool(getattr(model, "use_attention", False)),
                        "attention_heads": int(
                            getattr(
                                getattr(model, "attention", None), "num_heads", args.attention_heads
                            )
                        ),
                        "output_activation": str(
                            getattr(model, "output_activation", args.output_activation)
                        ),
                        "sigmoid_temperature": float(
                            getattr(model, "sigmoid_temperature", args.sigmoid_temperature)
                        ),
                    },
                    "model_state_dict": model.state_dict(),
                    "train_args": {
                        "data": str(args.data),
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                        "lr": float(args.lr),
                        "max_length": int(args.max_length),
                        "augment_prob": float(args.augment_prob),
                        "init_from": str(args.init_from) if args.init_from is not None else None,
                        "icf_mode": str(args.icf_mode),
                        "raw_overflow_penalty": float(args.raw_overflow_penalty),
                        "synthetic_oov": int(args.synthetic_oov),
                        "synthetic_oov_common_k": int(args.synthetic_oov_common_k),
                        "oov_aux_n": int(args.oov_aux_n),
                        "oov_aux_weight": float(args.oov_aux_weight),
                        "oov_aux_margin_common": float(args.oov_aux_margin_common),
                        "oov_aux_margin_gibberish": float(args.oov_aux_margin_gibberish),
                        "oov_aux_common_anchors": int(args.oov_aux_common_anchors),
                        "oov_aux_pool": int(args.oov_aux_pool),
                        "oov_aux_every": int(args.oov_aux_every),
                        "oov_aux_freeze_bn": not bool(args.oov_aux_no_freeze_bn),
                        "seed": 42,
                    },
                    "init": {"mean_icf_bias": float(mean_icf)},
                    "best_val_loss": float(best_val_loss),
                }
                torch.save(checkpoint, args.output)
                print(f"Saved best model to {args.output}")
            except Exception as e:
                print(f"Error saving model: {e}")
                raise

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
