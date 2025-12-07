"""Training with curriculum learning and advanced augmentation."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from tiny_icf.curriculum import create_curriculum_schedule, CurriculumSampler, get_stage_schedule
from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.data_universal import UniversalICFDataset, load_frequency_list_with_emojis
from tiny_icf.loss import CombinedLoss
from tiny_icf.model import UniversalICF
from tiny_icf.typo_augmentation import TypoAwareDataset


# Import from train.py to use the improved weighted sampling version
from tiny_icf.train import generate_ranking_pairs


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch with ranking loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for byte_tensors, icf_targets in tqdm(dataloader, desc="Training", leave=False):
        byte_tensors = byte_tensors.to(device)
        icf_targets = icf_targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(byte_tensors)
        
        # Generate pairs for ranking loss with weighted sampling
        n_pairs = min(len(icf_targets), 32)  # More pairs for better ranking learning
        pairs, pair_diffs = generate_ranking_pairs(
            icf_targets, n_pairs, min_diff=0.05, use_weighted_sampling=True
        )
        
        # Compute loss with ranking pairs and smooth rewards
        loss = criterion(
            predictions, icf_targets,
            pairs=pairs,
            pair_target_diffs=pair_diffs,
            smooth_ranking=True,
        )
        loss.backward()
        
        # Gradient clipping for training stability
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
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for byte_tensors, icf_targets in tqdm(dataloader, desc="Validating", leave=False):
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            
            predictions = model(byte_tensors)
            loss = criterion(predictions, icf_targets)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train with curriculum learning")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=20, help="Max word length")
    parser.add_argument("--augment-prob", type=float, default=0.2, help="Augmentation probability")
    parser.add_argument("--output", type=Path, default=Path("model_curriculum.pt"), help="Output model path")
    parser.add_argument("--curriculum-stages", type=int, default=5, help="Number of curriculum stages")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs (easy words only)")
    parser.add_argument("--typo-corpus", type=Path, help="Path to real typo corpus CSV (typo,correction)")
    parser.add_argument("--emoji-freq", type=Path, help="Path to emoji frequency CSV")
    parser.add_argument("--multilingual", action="store_true", help="Use multilingual ICF computation")
    parser.add_argument("--include-symbols", action="store_true", default=True, help="Include symbol augmentation")
    parser.add_argument("--include-emojis", action="store_true", default=True, help="Include emoji augmentation")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Random seed: 42 (reproducible)")
    print(f"Curriculum learning: {args.curriculum_stages} stages, {args.warmup_epochs} warmup epochs")
    
    # Load data (with emojis if provided)
    print("Loading frequency list...")
    if args.emoji_freq:
        word_counts, total_tokens = load_frequency_list_with_emojis(args.data, args.emoji_freq)
    else:
        word_counts, total_tokens = load_frequency_list(args.data)
    print(f"Loaded {len(word_counts)} words, {total_tokens:,} total tokens")
    
    # Compute ICF
    print("Computing normalized ICF...")
    word_icf = compute_normalized_icf(
        word_counts, 
        total_tokens,
        multilingual=args.multilingual,
    )
    
    # Split train/val BEFORE creating curriculum (fixes validation leakage)
    print("Splitting train/validation sets...")
    import random
    random.seed(42)
    all_samples = list(word_icf.items())
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train_samples_raw = all_samples[:split_idx]
    val_samples_raw = all_samples[split_idx:]
    print(f"Train: {len(train_samples_raw)} words, Val: {len(val_samples_raw)} words")
    
    # Stratified sampling on training set only
    print("Creating stratified sample from training set...")
    train_word_icf = dict(train_samples_raw)
    # Get word_counts for training set only (for optional token-frequency weighting)
    train_word_counts = {word: word_counts.get(word, 1) for word in train_word_icf.keys()}
    train_samples = stratified_sample(
        train_word_icf, 
        word_counts=train_word_counts,
        use_token_frequency=False  # Can enable for better distribution matching
    )
    print(f"Sampled {len(train_samples)} word-ICF pairs for training")
    
    # Create curriculum schedule from training samples only
    print(f"Creating curriculum schedule ({args.curriculum_stages} stages)...")
    stages = create_curriculum_schedule(train_samples, num_stages=args.curriculum_stages)
    
    for i, stage in enumerate(stages):
        avg_icf = sum(icf for _, icf in stage) / len(stage)
        print(f"  Stage {i}: {len(stage)} words, avg ICF: {avg_icf:.4f}")
    
    # Create curriculum sampler
    schedule = get_stage_schedule(args.epochs, args.curriculum_stages)
    curriculum = CurriculumSampler(stages, schedule, warmup_epochs=args.warmup_epochs)
    
    # Validation set from held-out data (no leakage)
    val_samples = val_samples_raw
    val_dataset = WordICFDataset(val_samples, max_length=args.max_length, augment_prob=0.0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = UniversalICF().to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop with curriculum
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        # Get current stage words
        current_stage_words = curriculum.get_current_stage_words()
        stage_num, total_stages, progress = curriculum.get_progress()
        
        # Create dataset for current stage
        # Use universal dataset (symbols + emojis + multilingual + real typos)
        train_dataset = UniversalICFDataset(
            current_stage_words,
            max_length=args.max_length,
            augment_prob=args.augment_prob,
            typo_corpus_path=args.typo_corpus if args.typo_corpus else None,
            emoji_freq_path=args.emoji_freq if args.emoji_freq else None,
            include_symbols=args.include_symbols,
            include_emojis=args.include_emojis,
        )
        if epoch == 0:
            features = []
            if args.typo_corpus:
                features.append("real typos")
            if args.include_symbols:
                features.append("symbols")
            if args.include_emojis:
                features.append("emojis")
            if args.multilingual:
                features.append("multilingual")
            if features:
                print(f"  Using universal augmentation: {', '.join(features)}")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        print(f"\nEpoch {epoch + 1}/{args.epochs} (Stage {stage_num + 1}/{total_stages}, {progress*100:.1f}% progress)")
        print(f"  Training on {len(current_stage_words)} words")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"  Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)
            print(f"  ✓ Saved best model (val loss: {val_loss:.4f})")
        
        # Advance curriculum
        curriculum.advance_epoch()
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

