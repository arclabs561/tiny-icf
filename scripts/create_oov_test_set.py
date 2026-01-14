"""Create Out-of-Vocabulary (OOV) test set for generalization validation."""

import sys
from pathlib import Path
import csv
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tiny_icf.data import load_frequency_list, compute_normalized_icf


def create_oov_test_set(
    train_data_path: Path,
    output_path: Path,
    oov_ratio: float = 0.2,
    min_word_length: int = 3,
    max_word_length: int = 20,
    seed: int = 42,
):
    """
    Create OOV test set by:
    1. Loading training data
    2. Splitting into train/OOV (80/20)
    3. Ensuring OOV words are not in training set
    4. Saving OOV test set to CSV
    
    Args:
        train_data_path: Path to training frequency CSV
        output_path: Path to save OOV test set CSV
        oov_ratio: Ratio of data to use for OOV (default 0.2 = 20%)
        min_word_length: Minimum word length
        max_word_length: Maximum word length
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    print(f"Loading training data from: {train_data_path}")
    word_counts, total_tokens = load_frequency_list(train_data_path)
    
    # Filter by length
    filtered_words = {
        word: count for word, count in word_counts.items()
        if min_word_length <= len(word) <= max_word_length
    }
    
    print(f"Loaded {len(filtered_words):,} words (after length filtering)")
    
    # Compute ICF
    word_icf = compute_normalized_icf(filtered_words, total_tokens)
    
    # Split into train/OOV
    all_words = list(word_icf.items())
    random.shuffle(all_words)
    
    split_idx = int(len(all_words) * (1 - oov_ratio))
    train_words = dict(all_words[:split_idx])
    oov_words = dict(all_words[split_idx:])
    
    print(f"Train set: {len(train_words):,} words")
    print(f"OOV test set: {len(oov_words):,} words ({oov_ratio*100:.1f}%)")
    
    # Verify OOV words are not in training set
    overlap = set(train_words.keys()) & set(oov_words.keys())
    if overlap:
        print(f"⚠️  Warning: {len(overlap)} words overlap between train and OOV sets")
        # Remove overlap from OOV
        for word in overlap:
            del oov_words[word]
        print(f"OOV test set after removing overlap: {len(oov_words):,} words")
    else:
        print("✅ No overlap between train and OOV sets")
    
    # Save OOV test set
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'icf_score'])
        for word, icf in sorted(oov_words.items()):
            writer.writerow([word, f"{icf:.6f}"])
    
    print(f"✅ OOV test set saved to: {output_path}")
    print(f"   Format: word,icf_score (CSV)")
    
    # Also save train set for reference
    train_output_path = output_path.parent / f"{output_path.stem}_train.csv"
    with open(train_output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'icf_score'])
        for word, icf in sorted(train_words.items()):
            writer.writerow([word, f"{icf:.6f}"])
    
    print(f"✅ Train set saved to: {train_output_path}")
    
    return train_words, oov_words


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create OOV test set for generalization validation')
    parser.add_argument('--train-data', type=Path, default=Path('data/word_frequency.csv'),
                        help='Path to training frequency CSV')
    parser.add_argument('--output', type=Path, default=Path('data/oov_test_set.csv'),
                        help='Path to save OOV test set CSV')
    parser.add_argument('--oov-ratio', type=float, default=0.2,
                        help='Ratio of data to use for OOV (default: 0.2 = 20%%)')
    parser.add_argument('--min-length', type=int, default=3,
                        help='Minimum word length (default: 3)')
    parser.add_argument('--max-length', type=int, default=20,
                        help='Maximum word length (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if not args.train_data.exists():
        print(f"❌ Error: Training data file not found: {args.train_data}")
        return 1
    
    create_oov_test_set(
        train_data_path=args.train_data,
        output_path=args.output,
        oov_ratio=args.oov_ratio,
        min_word_length=args.min_length,
        max_word_length=args.max_length,
        seed=args.seed,
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

