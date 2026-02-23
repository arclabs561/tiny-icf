"""Data loading and ICF normalization utilities."""

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from tiny_icf.augmentation import AdvancedAugmentation
from tiny_icf.preprocessing import filter_frequency_list


def compute_normalized_icf(
    word_counts: Dict[str, int],
    total_tokens: int,
    min_count: int = 5,
    multilingual: bool = False,
    *,
    mode: str = "log",
) -> Dict[str, float]:
    """
    Compute normalized ICF for all words with edge case handling.

    Modes:
    - mode="log" (default): corpus-ICF style normalization
        y = log((Total_Tokens + 1) / (Count + 1)) / log(Total_Tokens + 1)
    - mode="rank": corpus-invariant quantile target
        y = rank(count, descending) / (N - 1)

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
    if mode not in {"log", "rank"}:
        raise ValueError("mode must be 'log' or 'rank'")

    if multilingual:
        # Use per-language ICF computation
        from tiny_icf.data_multilingual import compute_icf_per_language

        icf_scores, _ = compute_icf_per_language(word_counts, min_count)
        return icf_scores

    if mode == "rank":
        # Corpus-invariant target: rank-normalized rarity.
        items = list(word_counts.items())
        n = len(items)
        if n == 0:
            return {}
        if n == 1:
            word, count = items[0]
            return {word: 1.0 if count < min_count else 0.5}

        counts = np.array([c for _, c in items], dtype=np.float64)
        # Descending order: most common => rank 0 (ICF 0.0), rarest => rank n-1 (ICF 1.0).
        order = np.argsort(-counts, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        icf_vals = ranks / float(n - 1)

        out: Dict[str, float] = {}
        for i, (word, count) in enumerate(items):
            if count < min_count:
                out[word] = 1.0
            else:
                out[word] = float(icf_vals[i])
        return out

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
        if first_row and (
            first_row[0].lower() in ["word", "token", "text"]
            or first_row[0].lower().startswith("word")
            or not first_row[1].isdigit()
        ):
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
    word_counts: Optional[Dict[str, int]] = None,
    head_size: int = 10000,
    body_size: int = 100000,
    head_prob: float = 0.4,
    body_prob: float = 0.3,
    use_token_frequency: bool = False,
    max_samples: Optional[int] = None,
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
        max_samples: Optional cap on total returned samples (helps on very large corpora)

    Returns:
        List of (word, icf_score) tuples
    """
    # Sort by ICF (ascending = most common first)
    sorted_words = sorted(word_icf.items(), key=lambda x: x[1])

    head = sorted_words[:head_size]
    body = sorted_words[head_size:body_size] if len(sorted_words) > head_size else []
    tail = sorted_words[body_size:] if len(sorted_words) > body_size else []

    def sample_with_weights(
        items: List[Tuple[str, float]], n_samples: int, weights: Optional[List[float]] = None
    ) -> List[Tuple[str, float]]:
        """Sample items, optionally weighted by token frequency."""
        if not items:
            return []

        if weights is not None and use_token_frequency:
            # Weighted sampling by token frequency: draw with replacement so high-count
            # words (e.g. "the") appear many times per epoch and get proper gradient.
            weights = np.array(weights, dtype=np.float64)
            weights = weights / weights.sum()
            size = n_samples  # allow repeated draws so head words are seen often
            indices = np.random.choice(len(items), size=size, replace=True, p=weights)
        else:
            # Uniform sampling (each word at most once per stratum)
            indices = np.random.choice(len(items), size=min(n_samples, len(items)), replace=False)

        return [items[i] for i in indices]

    # Calculate sample sizes.
    #
    # - If max_samples is unset, preserve the historical behavior: the sample budget
    #   is implicitly tied to the full vocab size, and per-stratum sampling is then
    #   clamped by pool sizes via `min(n_samples, len(items))`.
    # - If max_samples is set, treat it as a hard cap and push any leftover budget
    #   (after head/body clamping) into the tail so total samples ~= max_samples.
    if max_samples is None or int(max_samples) <= 0:
        n_total = int(len(sorted_words))
        n_head = int(n_total * head_prob)
        n_body = int(n_total * body_prob)
        n_tail = n_total - n_head - n_body
    else:
        n_total = int(min(int(max_samples), len(sorted_words)))
        n_head_desired = int(n_total * head_prob)
        n_body_desired = int(n_total * body_prob)
        n_head = min(n_head_desired, len(head))
        n_body = min(n_body_desired, len(body))
        n_tail = max(0, n_total - n_head - n_body)

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
    """PyTorch Dataset for word-ICF pairs with efficient caching."""

    def __init__(
        self,
        word_icf_pairs: List[Tuple[str, float]],
        max_length: int = 20,
        augment_prob: float = 0.0,
        augmentation_fn: Optional[Callable[[str], str]] = None,
        cache_byte_tensors: bool = True,
        return_words: bool = False,  # For distillation: return word strings
    ):
        """
        Args:
            word_icf_pairs: List of (word, icf_score) tuples
            max_length: Maximum character length (padding/truncation)
            augment_prob: Probability of applying augmentation
            augmentation_fn: Custom augmentation function (None = use default)
            cache_byte_tensors: If True, cache byte tensor conversions for efficiency
            return_words: If True, return word strings (for distillation)
        """
        self.pairs = word_icf_pairs
        self.max_length = max_length
        self.augment_prob = augment_prob
        self.augmentation_fn = augmentation_fn or AdvancedAugmentation()
        self.cache_byte_tensors = cache_byte_tensors
        self.return_words = return_words

        # Pre-compute byte tensors for validation (no augmentation) or when augment_prob=0
        # This avoids repeated UTF-8 encoding/decoding - major bottleneck fix
        if cache_byte_tensors and augment_prob == 0.0:
            # Pre-compute ALL byte tensors upfront for validation set
            # This eliminates __getitem__ overhead entirely
            self._byte_cache = {}
            self._precomputed_tensors = []
            self._precomputed_icfs = []

            for word, icf in self.pairs:
                byte_tensor = self._word_to_bytes_impl(word)
                self._byte_cache[word] = byte_tensor
                self._precomputed_tensors.append(byte_tensor)
                self._precomputed_icfs.append(icf)

            # Convert to tensors for faster indexing
            self._precomputed_tensors = torch.stack(self._precomputed_tensors)
            self._precomputed_icfs = torch.tensor(self._precomputed_icfs, dtype=torch.float32)
            self._use_precomputed = True
            print(
                f"✅ Pre-computed {len(self._precomputed_tensors)} validation samples (zero __getitem__ overhead)"
            )
        else:
            self._byte_cache = {}
            self._use_precomputed = False

    def __len__(self) -> int:
        return len(self.pairs)

    def _word_to_bytes_impl(self, word: str) -> torch.Tensor:
        """
        Internal implementation: Convert word to byte tensor.
        Cached separately for efficiency.
        """
        import unicodedata

        # Handle empty or None words
        if not word:
            return torch.zeros(self.max_length, dtype=torch.long)

        # Normalize to NFC (canonical composition) for consistency
        try:
            word = unicodedata.normalize("NFC", word)
        except (UnicodeError, TypeError):
            # Fallback: use word as-is if normalization fails
            pass

        # Truncate characters first (preserves UTF-8 validity for most cases)
        chars = list(word)[: self.max_length]
        try:
            byte_seq = "".join(chars).encode("utf-8")
        except (UnicodeEncodeError, UnicodeError):
            # Fallback: encode with error handling
            byte_seq = "".join(chars).encode("utf-8", errors="replace")

        # Truncate bytes if needed (multi-byte chars can exceed max_length)
        if len(byte_seq) > self.max_length:
            byte_seq = byte_seq[: self.max_length]

        # Pad to max_length bytes (may be < max_length if multi-byte chars)
        # This is acceptable - model handles variable-length via padding
        pad_length = max(0, self.max_length - len(byte_seq))
        padded = byte_seq + bytes(pad_length)
        return torch.tensor(list(padded), dtype=torch.long)

    def _word_to_bytes(self, word: str) -> torch.Tensor:
        """
        Convert word to byte tensor with caching for efficiency.

        Uses cache when augment_prob=0 (validation) or when word hasn't been augmented.
        """
        # Check cache first (only valid when no augmentation)
        if self.cache_byte_tensors and word in self._byte_cache:
            return self._byte_cache[word]

        # Compute and optionally cache
        result = self._word_to_bytes_impl(word)
        if self.cache_byte_tensors and self.augment_prob == 0.0:
            self._byte_cache[word] = result
        return result

    def _augment(self, word: str) -> str:
        """Apply advanced augmentation."""
        if np.random.random() > self.augment_prob:
            return word

        # Use advanced augmentation function
        return self.augmentation_fn(word)

    def __getitem__(self, idx: int):
        # Fast path: use precomputed tensors for validation (no augmentation)
        if self._use_precomputed:
            if self.return_words:
                word, _ = self.pairs[idx]
                return self._precomputed_tensors[idx], self._precomputed_icfs[idx], word
            return self._precomputed_tensors[idx], self._precomputed_icfs[idx]

        # Standard path: training with augmentation
        word, icf = self.pairs[idx]
        original_word = word  # Keep original for distillation

        # Apply augmentation during training
        word = self._augment(word)

        byte_tensor = self._word_to_bytes(word)
        icf_tensor = torch.tensor(icf, dtype=torch.float32)

        if self.return_words:
            return byte_tensor, icf_tensor, original_word
        return byte_tensor, icf_tensor
