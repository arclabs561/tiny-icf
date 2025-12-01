"""Optimized training with mixed precision, better DataLoader, LR scheduling."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiny_icf.curriculum import create_curriculum_schedule, CurriculumSampler, get_stage_schedule
from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.data_universal import UniversalICFDataset, load_frequency_list_with_emojis
from tiny_icf.loss import CombinedLoss
from tiny_icf.model import UniversalICF
from tiny_icf.typo_augmentation import TypoAwareDataset

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # For reproducibility
    else:
        torch.backends.cudnn.benchmark = True  # Faster on CPU


def generate_ranking_pairs(targets: torch.Tensor, n_pairs: int) -> torch.Tensor:
    """Generate pairs for ranking loss."""
    batch_size = len(targets)
    if batch_size < 2:
        return torch.empty((0, 2), dtype=torch.long, device=targets.device)
    
    pairs = []
    if targets.dim() > 1 and targets.size(1) == 1:
        targets_flat = targets.squeeze(1)
    else:
        targets_flat = targets
    
    for _ in range(n_pairs):
        i, j = torch.randint(0, batch_size, (2,), device=targets.device)
        if i == j:
            continue
        if targets_flat[i] < targets_flat[j]:
            pairs.append([i.item(), j.item()])
        elif targets_flat[j] < targets_flat[i]:
            pairs.append([j.item(), i.item()])
    
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=targets.device)
    
    return torch.tensor(pairs, dtype=torch.long, device=targets.device)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool = True,
) -> float:
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for byte_tensors, icf_targets in tqdm(dataloader, desc="Training", leave=False):
        byte_tensors = byte_tensors.to(device, non_blocking=True)
        icf_targets = icf_targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            predictions = model(byte_tensors)
            n_pairs = len(icf_targets) // 2
            pairs = generate_ranking_pairs(icf_targets, n_pairs)
            loss = criterion(predictions, icf_targets, pairs=pairs)
        
        # Mixed precision backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    compute_metrics: bool = True,
) -> tuple[float, dict]:
    """Validate with mixed precision and compute metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for byte_tensors, icf_targets in tqdm(dataloader, desc="Validation", leave=False):
            byte_tensors = byte_tensors.to(device, non_blocking=True)
            icf_targets = icf_targets.to(device, non_blocking=True)
            
            with autocast(enabled=use_amp):
                predictions = model(byte_tensors)
                n_pairs = len(icf_targets) // 2
                pairs = generate_ranking_pairs(icf_targets, n_pairs)
                loss = criterion(predictions, icf_targets, pairs=pairs)
            
            total_loss += loss.item()
            n_batches += 1
            
            if compute_metrics:
                all_predictions.append(predictions.cpu().squeeze())
                all_targets.append(icf_targets.cpu().squeeze())
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    metrics = {}
    if compute_metrics and all_predictions:
        pred_tensor = torch.cat(all_predictions).numpy()
        target_tensor = torch.cat(all_targets).numpy()
        
        # MAE
        mae = np.mean(np.abs(pred_tensor - target_tensor))
        metrics['mae'] = mae
        
        # RMSE
        rmse = np.sqrt(np.mean((pred_tensor - target_tensor) ** 2))
        metrics['rmse'] = rmse
        
        # Spearman correlation
        if HAS_SCIPY and len(pred_tensor) > 1:
            corr, p_value = spearmanr(pred_tensor, target_tensor)
            metrics['spearman_corr'] = corr
            metrics['spearman_p'] = p_value
        else:
            metrics['spearman_corr'] = 0.0
            metrics['spearman_p'] = 1.0
        
        # Prediction range
        metrics['pred_min'] = float(pred_tensor.min())
        metrics['pred_max'] = float(pred_tensor.max())
        metrics['pred_std'] = float(pred_tensor.std())
        metrics['target_std'] = float(target_tensor.std())
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Optimized training with AMP and LR scheduling")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256, help="Larger batch for GPU")
    parser.add_argument("--lr", type=float, default=2e-3, help="Higher LR for larger batch")
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--augment-prob", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("model_optimized.pt"))
    parser.add_argument("--curriculum-stages", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--typo-corpus", type=Path)
    parser.add_argument("--emoji-freq", type=Path)
    parser.add_argument("--multilingual", action="store_true")
    parser.add_argument("--include-symbols", action="store_true", default=True)
    parser.add_argument("--include-emojis", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--early-stopping", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    
    args = parser.parse_args()
    
    set_seed(42)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    use_amp = not args.no_amp and device.type == "cuda"
    
    print(f"Using device: {device}")
    print(f"Mixed precision (AMP): {use_amp}")
    print(f"DataLoader workers: {args.num_workers}")
    print(f"Batch size: {args.batch_size}")
    
    # Load data
    print("Loading frequency list...")
    if args.emoji_freq:
        word_counts, total_tokens = load_frequency_list_with_emojis(args.data, args.emoji_freq)
    else:
        word_counts, total_tokens = load_frequency_list(args.data)
    print(f"Loaded {len(word_counts)} words, {total_tokens:,} total tokens")
    
    # Compute ICF
    print("Computing normalized ICF...")
    word_icf = compute_normalized_icf(word_counts, total_tokens, multilingual=args.multilingual)
    
    # Split train/val
    print("Splitting train/validation sets...")
    random.seed(42)
    all_samples = list(word_icf.items())
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train_samples_raw = all_samples[:split_idx]
    val_samples_raw = all_samples[split_idx:]
    print(f"Train: {len(train_samples_raw)} words, Val: {len(val_samples_raw)} words")
    
    # Stratified sampling
    print("Creating stratified sample from training set...")
    train_word_icf = dict(train_samples_raw)
    train_word_counts = {word: word_counts.get(word, 1) for word in train_word_icf.keys()}
    train_samples = stratified_sample(train_word_icf, word_counts=train_word_counts, use_token_frequency=False)
    print(f"Sampled {len(train_samples)} word-ICF pairs for training")
    
    # Create curriculum
    print(f"Creating curriculum schedule ({args.curriculum_stages} stages)...")
    stages = create_curriculum_schedule(train_samples, num_stages=args.curriculum_stages)
    for i, stage in enumerate(stages):
        avg_icf = sum(icf for _, icf in stage) / len(stage)
        print(f"  Stage {i}: {len(stage)} words, avg ICF: {avg_icf:.4f}")
    
    schedule = get_stage_schedule(args.epochs, args.curriculum_stages)
    curriculum = CurriculumSampler(stages, schedule, warmup_epochs=args.warmup_epochs)
    
    # Validation set
    val_samples = val_samples_raw
    val_dataset = WordICFDataset(val_samples, max_length=args.max_length, augment_prob=0.0)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    
    # Model
    model = UniversalICF().to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss, optimizer, scheduler, scaler
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = GradScaler(enabled=use_amp)
    
    # Training loop with checkpointing
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = args.output.parent / f"{args.output.stem}_checkpoint.pt"
    start_epoch = 0
    
    # Resume from checkpoint if exists
    if checkpoint_path.exists():
        print(f"📂 Found checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"   Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"   ⚠️  Could not load checkpoint: {e}")
            print("   Starting from scratch")
    
    for epoch in range(start_epoch, args.epochs):
        current_stage_words = curriculum.get_current_stage_words()
        stage_num, total_stages, progress = curriculum.get_progress()
        
        train_dataset = UniversalICFDataset(
            current_stage_words,
            max_length=args.max_length,
            augment_prob=args.augment_prob,
            typo_corpus_path=args.typo_corpus if args.typo_corpus else None,
            emoji_freq_path=args.emoji_freq if args.emoji_freq else None,
            include_symbols=args.include_symbols,
            include_emojis=args.include_emojis,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        
        print(f"\nEpoch {epoch + 1}/{args.epochs} (Stage {stage_num + 1}/{total_stages}, {progress*100:.1f}% progress)")
        print(f"  Training on {len(current_stage_words)} words")
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, use_amp, compute_metrics=True)
        
        scheduler.step()
        
        # Print metrics
        print(f"  Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        if val_metrics:
            print(f"  MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
            if val_metrics.get('spearman_corr', 0) != 0:
                print(f"  Spearman: {val_metrics['spearman_corr']:.4f} (p={val_metrics['spearman_p']:.4f})")
            print(f"  Pred range: [{val_metrics['pred_min']:.4f}, {val_metrics['pred_max']:.4f}], std: {val_metrics['pred_std']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)
            print(f"  ✓ Saved best model (val loss: {val_loss:.4f})")
            if val_metrics.get('spearman_corr', 0) > 0.8:
                print(f"  ⭐ Excellent correlation: {val_metrics['spearman_corr']:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping:
                print(f"  Early stopping after {args.early_stopping} epochs without improvement")
                break
        
        # Save checkpoint every epoch with metrics
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'curriculum_stage': stage_num,
            'val_metrics': val_metrics,
            'train_loss': train_loss,
        }, checkpoint_path)
        
        curriculum.advance_epoch()
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

