# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24.0",
# ]
# ///
"""
Baseline methods for ICF prediction.

Implements simple baselines to compare against neural models:
- Character unigram frequency
- Character bigram frequency
- Character trigram frequency
- Word length heuristic
- TFIDF (if scikit-learn available)
"""

from typing import Dict, List, Optional
from collections import Counter
import numpy as np


def _rank_normalize_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """
    Map a 1D array to [0, 1] by rank (min -> 0, max -> 1).

    This is robust to arbitrary score scales and preserves ordering, which is
    what matters for Spearman-style comparisons.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    n = len(values)
    if n == 0:
        return values
    if n == 1:
        return np.array([0.5], dtype=float)
    if np.allclose(values, values[0]):
        return np.full(n, 0.5, dtype=float)

    order = np.argsort(values)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    return ranks / float(n - 1)


def character_unigram_baseline(
    words: List[str],
    word_counts: Dict[str, int],
    total_tokens: int,
) -> Dict[str, float]:
    """
    Predict ICF using character unigram frequencies.

    Heuristic: words whose characters are rare in the corpus tend to be rarer.
    We compute a per-word rarity score via mean(-log p(char)), then rank-normalize
    those scores into the ICF range [0, 1].

    Args:
        words: List of words to predict
        word_counts: Word frequency counts
        total_tokens: Total tokens in corpus

    Returns:
        Dictionary mapping words to predicted ICF scores
    """
    # Compute character frequencies (token-weighted)
    char_counts = Counter()
    for word, count in word_counts.items():
        for char in word.lower():
            char_counts[char] += count

    total_chars = sum(char_counts.values())
    if total_chars <= 0:
        return {word: 0.5 for word in words}
    char_freqs = {char: count / total_chars for char, count in char_counts.items()}

    # Score each word by average character surprisal.
    eps = 1e-12
    scores = np.zeros(len(words), dtype=float)
    for i, word in enumerate(words):
        if not word:
            scores[i] = float("inf")
            continue
        freqs = [char_freqs.get(ch, eps) for ch in word.lower()]
        scores[i] = float(np.mean(-np.log(np.maximum(freqs, eps))))

    icf_vals = _rank_normalize_to_unit_interval(scores)
    return {word: float(icf_vals[i]) for i, word in enumerate(words)}


def character_bigram_baseline(
    words: List[str],
    word_counts: Dict[str, int],
    total_tokens: int,
) -> Dict[str, float]:
    """
    Predict ICF using character bigram frequencies.

    Heuristic: words whose character *bigrams* are rare in the corpus tend to be rarer.
    We compute a per-word rarity score via mean(-log p(bigram)), then rank-normalize
    those scores into the ICF range [0, 1].

    Args:
        words: List of words to predict
        word_counts: Word frequency counts
        total_tokens: Total tokens in corpus

    Returns:
        Dictionary mapping words to predicted ICF scores
    """
    # Compute bigram frequencies (token-weighted)
    bigram_counts = Counter()
    for word, count in word_counts.items():
        word_lower = word.lower()
        for i in range(len(word_lower) - 1):
            bigram = word_lower[i : i + 2]
            bigram_counts[bigram] += count

    total_bigrams = sum(bigram_counts.values())
    if total_bigrams <= 0:
        return {word: 0.5 for word in words}
    bigram_freqs = {bigram: count / total_bigrams for bigram, count in bigram_counts.items()}

    # Also compute unigram freqs for short-word fallback.
    char_counts = Counter()
    for word, count in word_counts.items():
        for ch in word.lower():
            char_counts[ch] += count
    total_chars = sum(char_counts.values())
    char_freqs = {ch: c / total_chars for ch, c in char_counts.items()} if total_chars > 0 else {}

    eps = 1e-12
    scores = np.zeros(len(words), dtype=float)
    for i, word in enumerate(words):
        if not word:
            scores[i] = float("inf")
            continue

        w = word.lower()
        if len(w) < 2:
            # No bigrams; fall back to unigram surprisal.
            freqs = [char_freqs.get(ch, eps) for ch in w]
            scores[i] = float(np.mean(-np.log(np.maximum(freqs, eps))))
            continue

        freqs = [bigram_freqs.get(w[j : j + 2], eps) for j in range(len(w) - 1)]
        scores[i] = float(np.mean(-np.log(np.maximum(freqs, eps))))

    icf_vals = _rank_normalize_to_unit_interval(scores)
    return {word: float(icf_vals[i]) for i, word in enumerate(words)}


def word_length_baseline(
    words: List[str],
    word_counts: Dict[str, int],
    total_tokens: int,
) -> Dict[str, float]:
    """
    Predict ICF using word length heuristic.

    Simple heuristic: longer words are rarer.
    Normalized by corpus statistics.

    Args:
        words: List of words to predict
        word_counts: Word frequency counts
        total_tokens: Total tokens in corpus

    Returns:
        Dictionary mapping words to predicted ICF scores
    """
    # Compute length statistics
    lengths = [len(word) for word in word_counts.keys()]
    if not lengths:
        return {word: 0.5 for word in words}

    min_len = min(lengths)
    max_len = max(lengths)

    # Predict ICF based on length
    predictions = {}
    for word in words:
        word_len = len(word)
        if max_len > min_len:
            # Normalize length to [0, 1]
            normalized_len = (word_len - min_len) / (max_len - min_len)
            # Longer = rarer, but use sigmoid for smoother transition
            icf = 1.0 / (1.0 + np.exp(-(normalized_len - 0.5) * 4))
        else:
            icf = 0.5

        predictions[word] = max(0.0, min(1.0, icf))

    return predictions


def tfidf_baseline(
    words: List[str],
    word_counts: Dict[str, int],
    total_tokens: int,
    documents: Optional[List[str]] = None,
) -> Optional[Dict[str, float]]:
    """
    Predict ICF using TFIDF (if scikit-learn available).

    TFIDF = Term Frequency × Inverse Document Frequency
    Higher TFIDF = more informative = higher ICF

    Args:
        words: List of words to predict
        word_counts: Word frequency counts
        total_tokens: Total tokens in corpus
        documents: Optional list of documents (if None, treats each word as a document)

    Returns:
        Dictionary mapping words to predicted ICF scores, or None if scikit-learn unavailable
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return None

    # If no documents provided, create pseudo-documents
    if documents is None:
        # Create one document per word (simple approach)
        documents = list(word_counts.keys())

    if not documents:
        return None

    # Compute TFIDF
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        max_features=1000,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)

        # For each word, compute average TFIDF
        predictions = {}
        for word in words:
            if word not in documents:
                predictions[word] = 0.5  # Default
                continue

            doc_idx = documents.index(word)
            tfidf_scores = tfidf_matrix[doc_idx].toarray()[0]
            avg_tfidf = np.mean(tfidf_scores)

            # Normalize to [0, 1] range
            # Higher TFIDF = rarer word = higher ICF
            if avg_tfidf > 0:
                icf = min(1.0, avg_tfidf * 2.0)  # Scale factor
            else:
                icf = 0.0

            predictions[word] = max(0.0, min(1.0, icf))

        return predictions
    except Exception:
        return None


def evaluate_baselines(
    words: List[str],
    true_icf: Dict[str, float],
    word_counts: Dict[str, int],
    total_tokens: int,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all baseline methods.

    Args:
        words: List of words to evaluate
        true_icf: True ICF scores
        word_counts: Word frequency counts
        total_tokens: Total tokens in corpus

    Returns:
        Dictionary mapping baseline name to metrics
    """
    from scipy.stats import spearmanr

    results = {}

    # Unigram baseline
    unigram_preds = character_unigram_baseline(words, word_counts, total_tokens)
    unigram_preds_list = [unigram_preds.get(w, 0.5) for w in words]
    true_list = [true_icf.get(w, 0.5) for w in words]

    if len(unigram_preds_list) > 1:
        spearman, _ = spearmanr(unigram_preds_list, true_list)
        mae = np.mean(np.abs(np.array(unigram_preds_list) - np.array(true_list)))
    else:
        spearman, mae = 0.0, 0.0

    results["unigram"] = {
        "spearman": float(spearman),
        "mae": float(mae),
    }

    # Bigram baseline
    bigram_preds = character_bigram_baseline(words, word_counts, total_tokens)
    bigram_preds_list = [bigram_preds.get(w, 0.5) for w in words]

    if len(bigram_preds_list) > 1:
        spearman, _ = spearmanr(bigram_preds_list, true_list)
        mae = np.mean(np.abs(np.array(bigram_preds_list) - np.array(true_list)))
    else:
        spearman, mae = 0.0, 0.0

    results["bigram"] = {
        "spearman": float(spearman),
        "mae": float(mae),
    }

    # Length baseline
    length_preds = word_length_baseline(words, word_counts, total_tokens)
    length_preds_list = [length_preds.get(w, 0.5) for w in words]

    if len(length_preds_list) > 1:
        spearman, _ = spearmanr(length_preds_list, true_list)
        mae = np.mean(np.abs(np.array(length_preds_list) - np.array(true_list)))
    else:
        spearman, mae = 0.0, 0.0

    results["length"] = {
        "spearman": float(spearman),
        "mae": float(mae),
    }

    # TFIDF baseline (if available)
    tfidf_preds = tfidf_baseline(words, word_counts, total_tokens)
    if tfidf_preds:
        tfidf_preds_list = [tfidf_preds.get(w, 0.5) for w in words]
        if len(tfidf_preds_list) > 1:
            spearman, _ = spearmanr(tfidf_preds_list, true_list)
            mae = np.mean(np.abs(np.array(tfidf_preds_list) - np.array(true_list)))
        else:
            spearman, mae = 0.0, 0.0

        results["tfidf"] = {
            "spearman": float(spearman),
            "mae": float(mae),
        }

    return results
