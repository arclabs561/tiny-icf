"""Data loading and ICF normalization utilities."""

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from tiny_icf.augmentation import AdvancedAugmentation
from tiny_icf.preprocessing import filter_frequency_list


def compute_normalized_icf(
    word_counts: Dict[str, int], total_tokens: int, min_count: int = 5, multilingual: bool = False
) -> Dict[str, float]:
    """
    Compute normalized ICF for all words with edge case handling.
    
    Formula: y = (log(Total_Tokens + 1) - log(Count + 1)) / log(Total_Tokens + 1)
    
    Uses add-1 smoothing to handle edge cases (zero division, count = total).
    
    For multilingual data (with language prefixes like "en:word"), computes ICF
    per language to avoid mixing corpora.
    
    Args:
        word_counts: Dictionary mapping words to their frequency counts
        total_tokens: Total number of tokens in the corpus
        min_count: Minimum count threshold (words below this get ICF=1.0)
        multilingual: If True, compute ICF per language separately
    
    Returns:
        Dictionary mapping words to normalized ICF scores (0.0=common, 1.0=rare)
    """
    if multilingual:
        # Use per-language ICF computation
        from tiny_icf.data_multilingual import compute_icf_per_language
        icf_scores, _ = compute_icf_per_language(word_counts, min_count)
        return icf_scores
    
    # Single-corpus computation with smoothing
    # Add 1 to prevent edge cases: log(1) = 0, count = total_tokens
    log_total = math.log(total_tokens + 1)
    icf_scores = {}
    
    for word, count in word_counts.items():
        if count < min_count:
            # Treat as effectively unknown/rare
            icf_scores[word] = 1.0
        elif count >= total_tokens:
            # Most common word (shouldn't happen, but handle gracefully)
            icf_scores[word] = 0.0
        else:
            # Normalized ICF with smoothing: higher score = rarer word
            icf = math.log((total_tokens + 1) / (count + 1)) / log_total
            # Clip to [0, 1] range
            icf_scores[word] = max(0.0, min(1.0, icf))
    
    return icf_scores


def load_frequency_list(
    filepath: Path,
    filter_noise: bool = True,
    min_length: int = 2,
    max_length: int = 50,
) -> Tuple[Dict[str, int], int]:
    """
    Load word frequency list from CSV file.
    
    Expected format: word,count (one per line)
    
    Returns:
        Tuple of (word_counts dict, total_tokens)
    """
    word_counts = {}
    total_tokens = 0
    
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Skip header if present
        first_row = next(reader, None)
        if first_row and (first_row[0].lower() in ['word', 'token', 'text'] or 
                          first_row[0].lower().startswith('word') or
                          not first_row[1].isdigit()):
            # This looks like a header, skip it
            pass
        else:
            # First row is data, process it
            if first_row and len(first_row) >= 2:
                try:
                    word = first_row[0].strip().lower()
                    count = int(first_row[1])
                    # Accumulate counts (handle duplicate words)
                    word_counts[word] = word_counts.get(word, 0) + count
                    total_tokens += count
                except (ValueError, IndexError):
                    pass
        
        # Process remaining rows
        for row in reader:
            if len(row) < 2:
                continue
            try:
                word = row[0].strip().lower()
                count = int(row[1])
                word_counts[word] = word_counts.get(word, 0) + count
                total_tokens += count
            except (ValueError, IndexError):
                continue
    
    # Filter noise if requested
    if filter_noise:
        word_counts = filter_frequency_list(
            word_counts,
            min_length=min_length,
            max_length=max_length,
        )
        # Recalculate total (filtered)
        total_tokens = sum(word_counts.values())
    
    return word_counts, total_tokens


def stratified_sample(
    word_icf: Dict[str, float],
    word_counts: Dict[str, int] | None = None,
    head_size: int = 10000,
    body_size: int = 100000,
    head_prob: float = 0.4,
    body_prob: float = 0.3,
    use_token_frequency: bool = False,
) -> List[Tuple[str, float]]:
    """
    Create stratified sample from word ICF dictionary.
    
    Handles Zipfian distribution by sampling from:
    - Head (top 10k): 40% probability
    - Body (10k-100k): 30% probability  
    - Tail (100k+): 30% probability
    
    Args:
        word_icf: Dictionary mapping words to ICF scores
        word_counts: Optional dictionary mapping words to token counts (for frequency-weighted sampling)
        use_token_frequency: If True, sample weighted by token frequency (matches real distribution)
    
    Returns:
        List of (word, icf_score) tuples
    """
    # Sort by ICF (ascending = most common first)
    sorted_words = sorted(word_icf.items(), key=lambda x: x[1])
    
    head = sorted_words[:head_size]
    body = sorted_words[head_size:body_size] if len(sorted_words) > head_size else []
    tail = sorted_words[body_size:] if len(sorted_words) > body_size else []
    
    def sample_with_weights(items: List[Tuple[str, float]], n_samples: int, weights: List[float] | None = None) -> List[Tuple[str, float]]:
        """Sample items, optionally weighted by token frequency."""
        if not items:
            return []
        
        if weights is not None and use_token_frequency:
            # Weighted sampling by token frequency
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            indices = np.random.choice(len(items), size=min(n_samples, len(items)), replace=False, p=weights)
        else:
            # Uniform sampling
            indices = np.random.choice(len(items), size=min(n_samples, len(items)), replace=False)
        
        return [items[i] for i in indices]
    
    # Calculate sample sizes
    n_total = int(len(sorted_words) * (head_prob + body_prob + (1 - head_prob - body_prob)))
    n_head = int(n_total * head_prob)
    n_body = int(n_total * body_prob)
    n_tail = n_total - n_head - n_body
    
    # Get weights if using token frequency
    head_weights = None
    body_weights = None
    tail_weights = None
    
    if use_token_frequency and word_counts is not None:
        head_weights = [word_counts.get(word, 1) for word, _ in head]
        body_weights = [word_counts.get(word, 1) for word, _ in body]
        tail_weights = [word_counts.get(word, 1) for word, _ in tail]
    
    # Sample from each stratum
    head_samples = sample_with_weights(head, n_head, head_weights)
    body_samples = sample_with_weights(body, n_body, body_weights)
    tail_samples = sample_with_weights(tail, n_tail, tail_weights)
    
    return head_samples + body_samples + tail_samples


class WordICFDataset(Dataset):
    """PyTorch Dataset for word-ICF pairs."""
    
    def __init__(
        self,
        word_icf_pairs: List[Tuple[str, float]],
        max_length: int = 20,
        augment_prob: float = 0.0,
        augmentation_fn: Callable[[str], str] | None = None,
    ):
        """
        Args:
            word_icf_pairs: List of (word, icf_score) tuples
            max_length: Maximum character length (padding/truncation)
            augment_prob: Probability of applying augmentation
            augmentation_fn: Custom augmentation function (None = use default)
        """
        self.pairs = word_icf_pairs
        self.max_length = max_length
        self.augment_prob = augment_prob
        self.augmentation_fn = augmentation_fn or AdvancedAugmentation()
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def _word_to_bytes(self, word: str) -> torch.Tensor:
        """
        Convert word to byte tensor with character-boundary aware truncation.
        
        Truncates at character boundaries (not byte boundaries) to preserve UTF-8 validity.
        Similar to Rust's bstr approach - ensures multi-byte characters aren't corrupted.
        """
        import unicodedata
        
        # Normalize to NFC (canonical composition) for consistency
        word = unicodedata.normalize('NFC', word)
        
        # Truncate characters first (preserves UTF-8 validity)
        chars = list(word)[:self.max_length]
        byte_seq = ''.join(chars).encode("utf-8")
        
        # Truncate bytes if needed (multi-byte chars can exceed max_length)
        if len(byte_seq) > self.max_length:
            byte_seq = byte_seq[:self.max_length]
        
        # Pad to max_length bytes (may be < max_length if multi-byte chars)
        # This is acceptable - model handles variable-length via padding
        pad_length = max(0, self.max_length - len(byte_seq))
        padded = byte_seq + bytes(pad_length)
        return torch.tensor(list(padded), dtype=torch.long)
    
    def _augment(self, word: str) -> str:
        """Apply advanced augmentation."""
        if np.random.random() > self.augment_prob:
            return word
        
        # Use advanced augmentation function
        return self.augmentation_fn(word)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        word, icf = self.pairs[idx]
        
        # Apply augmentation during training
        word = self._augment(word)
        
        byte_tensor = self._word_to_bytes(word)
        icf_tensor = torch.tensor(icf, dtype=torch.float32)
        
        return byte_tensor, icf_tensor

