"""Universal data handling: symbols, emojis, multilingual."""

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list
from tiny_icf.symbol_augmentation import UniversalAugmentation


def load_frequency_list_with_emojis(
    word_freq_path: Path,
    emoji_freq_path: Path | None = None,
) -> Tuple[Dict[str, int], int]:
    """
    Load frequency list including emojis/emoticons.
    
    Merges word frequencies with emoji frequencies for complete coverage.
    """
    word_counts, total_tokens = load_frequency_list(word_freq_path)
    
    # Add emoji frequencies if available
    if emoji_freq_path and emoji_freq_path.exists():
        with open(emoji_freq_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                emoji = row['emoji'].strip()
                count = int(row['count'])
                word_counts[emoji] = word_counts.get(emoji, 0) + count
                total_tokens += count
        
        # Count emojis added (re-read file to count)
        with open(emoji_freq_path, 'r', encoding='utf-8') as f:
            emoji_count = sum(1 for _ in csv.DictReader(f))
        print(f"Added {emoji_count} emoji/emoticon frequencies")
    
    return word_counts, total_tokens


def filter_symbol_heavy_words(
    word_icf_pairs: List[Tuple[str, float]],
    max_symbol_ratio: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Filter out words that are mostly symbols (not useful for frequency estimation).
    
    Keeps words that are primarily alphanumeric with some symbols.
    """
    filtered = []
    
    for word, icf in word_icf_pairs:
        # Count alphanumeric vs non-alphanumeric
        alpha_count = sum(1 for c in word if c.isalnum())
        total_count = len(word)
        
        if total_count == 0:
            continue
        
        alpha_ratio = alpha_count / total_count
        
        # Keep if mostly alphanumeric, or if it's a known emoji/emoticon
        if alpha_ratio >= (1 - max_symbol_ratio) or len(word) <= 3:
            filtered.append((word, icf))
    
    return filtered


class UniversalICFDataset(WordICFDataset):
    """
    Universal dataset handling:
    - Words (all languages)
    - Symbols and punctuation
    - Emojis/emoticons
    - Multilingual typo patterns
    """
    
    def __init__(
        self,
        word_icf_pairs: List[Tuple[str, float]],
        max_length: int = 20,
        augment_prob: float = 0.0,
        typo_corpus_path: Path | None = None,
        emoji_freq_path: Path | None = None,
        include_symbols: bool = True,
        include_emojis: bool = True,
    ):
        """
        Args:
            word_icf_pairs: List of (word, icf_score) tuples
            max_length: Maximum byte length
            augment_prob: Probability of augmentation
            typo_corpus_path: Path to real typo corpus
            emoji_freq_path: Path to emoji frequency list
            include_symbols: Whether to augment with symbols
            include_emojis: Whether to augment with emojis
        """
        # Filter symbol-heavy words if needed
        if not include_symbols:
            word_icf_pairs = filter_symbol_heavy_words(word_icf_pairs)
        
        # Use universal augmentation
        augmentation_fn = UniversalAugmentation(
            typo_corpus_path=typo_corpus_path,
            symbol_prob=0.05 if include_symbols else 0.0,
            emoji_prob=0.02 if include_emojis else 0.0,
            multilingual_prob=0.1,
            keyboard_prob=0.15,
        )
        
        super().__init__(
            word_icf_pairs,
            max_length=max_length,
            augment_prob=augment_prob,
            augmentation_fn=augmentation_fn,
        )

