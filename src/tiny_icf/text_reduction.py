"""Optimal text reduction using ICF to minimize embedding regret.

The idea: Drop words in optimal order (lowest ICF first) to minimize
the difference between original and reduced text embeddings.
"""

import torch
from typing import List, Tuple
import numpy as np


def compute_embedding_difference(
    original_embedding: torch.Tensor,
    reduced_embedding: torch.Tensor,
) -> float:
    """
    Compute embedding difference (regret) between original and reduced text.
    
    Uses cosine similarity or L2 distance to measure how much information
    is lost when dropping words.
    
    Args:
        original_embedding: [dim] embedding of original text
        reduced_embedding: [dim] embedding of reduced text
    
    Returns:
        Regret score (higher = more information lost)
    """
    # Cosine distance: 1 - cosine_similarity
    # Higher = more different
    cos_sim = torch.nn.functional.cosine_similarity(
        original_embedding.unsqueeze(0),
        reduced_embedding.unsqueeze(0),
    ).item()
    regret = 1.0 - cos_sim
    
    # Alternative: L2 distance
    # regret = torch.norm(original_embedding - reduced_embedding).item()
    
    return regret


def greedy_icf_reduction(
    words: List[str],
    icf_scores: List[float],
    target_length: int,
    embedding_fn: callable,
) -> Tuple[List[str], float]:
    """
    Greedy reduction: Drop lowest ICF words first.
    
    Simple heuristic: Common words (low ICF) contribute less to semantics,
    so drop them first to preserve rare, informative words.
    
    Args:
        words: List of words in text
        icf_scores: ICF score for each word
        target_length: Target number of words to keep
        embedding_fn: Function that takes word list -> embedding
    
    Returns:
        (reduced_words, regret)
    """
    # Sort by ICF (ascending: common first)
    word_icf_pairs = list(zip(words, icf_scores))
    word_icf_pairs.sort(key=lambda x: x[1])  # Sort by ICF
    
    # Keep top N words (highest ICF = most informative)
    keep_pairs = word_icf_pairs[-target_length:]
    reduced_words = [word for word, _ in keep_pairs]
    
    # Compute regret
    original_embedding = embedding_fn(words)
    reduced_embedding = embedding_fn(reduced_words)
    regret = compute_embedding_difference(original_embedding, reduced_embedding)
    
    return reduced_words, regret


def optimal_reduction_dynamic(
    words: List[str],
    icf_scores: List[float],
    target_length: int,
    embedding_fn: callable,
    max_iterations: int = 100,
) -> Tuple[List[str], float]:
    """
    Optimal reduction using dynamic programming or greedy with regret.
    
    Goal: Find subset of words that minimizes embedding regret.
    Uses ICF scores to guide search (prefer dropping low ICF words).
    
    Args:
        words: List of words
        icf_scores: ICF for each word
        target_length: Target words to keep
        embedding_fn: words -> embedding
        max_iterations: Max iterations for optimization
    
    Returns:
        (optimal_words, min_regret)
    """
    n = len(words)
    if n <= target_length:
        return words, 0.0
    
    # Compute original embedding once
    original_embedding = embedding_fn(words)
    
    # Greedy approach: Iteratively drop word that causes least regret
    current_words = words.copy()
    current_icf = icf_scores.copy()
    
    while len(current_words) > target_length:
        min_regret = float('inf')
        best_idx = -1
        
        # Try dropping each word, find one with least regret
        for i in range(len(current_words)):
            # Skip if this is a high-ICF word (we want to keep those)
            if current_icf[i] > 0.7:
                continue  # Don't drop rare words
            
            # Try dropping word i
            test_words = current_words[:i] + current_words[i+1:]
            test_embedding = embedding_fn(test_words)
            regret = compute_embedding_difference(original_embedding, test_embedding)
            
            # Weight by ICF: prefer dropping low ICF words
            # Low ICF = common = less informative
            weighted_regret = regret * (1.0 - current_icf[i])
            
            if weighted_regret < min_regret:
                min_regret = weighted_regret
                best_idx = i
        
        # Drop the word with least regret
        if best_idx >= 0:
            current_words.pop(best_idx)
            current_icf.pop(best_idx)
        else:
            # Fallback: drop lowest ICF word
            min_icf_idx = min(range(len(current_icf)), key=lambda i: current_icf[i])
            current_words.pop(min_icf_idx)
            current_icf.pop(min_icf_idx)
    
    # Compute final regret
    reduced_embedding = embedding_fn(current_words)
    final_regret = compute_embedding_difference(original_embedding, reduced_embedding)
    
    return current_words, final_regret


def optimal_reduction_beam_search(
    words: List[str],
    icf_scores: List[float],
    target_length: int,
    embedding_fn: callable,
    beam_size: int = 10,
) -> Tuple[List[str], float]:
    """
    Beam search for optimal reduction.
    
    Maintains top-K candidates at each step, exploring multiple paths.
    More thorough than greedy but slower.
    
    Args:
        words: List of words
        icf_scores: ICF for each word
        target_length: Target words to keep
        embedding_fn: words -> embedding
        beam_size: Number of candidates to maintain
    
    Returns:
        (optimal_words, min_regret)
    """
    n = len(words)
    if n <= target_length:
        return words, 0.0
    
    original_embedding = embedding_fn(words)
    
    # Beam: list of (word_list, icf_list, regret)
    beam = [(words.copy(), icf_scores.copy(), 0.0)]
    
    # Iteratively reduce
    while len(beam[0][0]) > target_length:
        candidates = []
        
        for word_list, icf_list, _ in beam:
            # Try dropping each word
            for i in range(len(word_list)):
                # Prefer dropping low ICF words
                if icf_list[i] > 0.7:
                    continue
                
                test_words = word_list[:i] + word_list[i+1:]
                test_icf = icf_list[:i] + icf_list[i+1:]
                test_embedding = embedding_fn(test_words)
                regret = compute_embedding_difference(original_embedding, test_embedding)
                
                # Weight by ICF
                weighted_regret = regret * (1.0 - icf_list[i])
                candidates.append((test_words, test_icf, weighted_regret))
        
        # Keep top-K candidates (lowest regret)
        candidates.sort(key=lambda x: x[2])
        beam = candidates[:beam_size]
    
    # Return best candidate
    best_words, _, best_regret = beam[0]
    final_embedding = embedding_fn(best_words)
    final_regret = compute_embedding_difference(original_embedding, final_embedding)
    
    return best_words, final_regret


def reduce_text_with_icf(
    text: str,
    icf_model,
    embedding_model,
    target_ratio: float = 0.5,
    method: str = "greedy",
) -> Tuple[str, float, dict]:
    """
    Reduce text using ICF-guided optimal reduction.
    
    Args:
        text: Input text to reduce
        icf_model: Model that predicts ICF for words
        embedding_model: Model that computes text embeddings
        target_ratio: Fraction of words to keep (0.5 = keep 50%)
        method: "greedy", "dynamic", or "beam"
    
    Returns:
        (reduced_text, regret, stats)
    """
    # Tokenize
    words = text.split()
    n_words = len(words)
    target_length = max(1, int(n_words * target_ratio))
    
    # Predict ICF for each word
    icf_scores = []
    for word in words:
        # Clean word (remove punctuation)
        clean_word = ''.join(c for c in word if c.isalnum())
        if clean_word:
            byte_seq = clean_word.encode('utf-8')[:20]
            padded = byte_seq + bytes(20 - len(byte_seq))
            byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                icf = icf_model(byte_tensor).item()
            icf_scores.append(icf)
        else:
            icf_scores.append(0.0)  # Punctuation-only
    
    # Embedding function
    def embedding_fn(word_list: List[str]) -> torch.Tensor:
        text = ' '.join(word_list)
        # Simple: average word embeddings (or use actual embedding model)
        # For now, placeholder - would use actual embedding model
        return torch.randn(384)  # Placeholder
    
    # Reduce
    if method == "greedy":
        reduced_words, regret = greedy_icf_reduction(
            words, icf_scores, target_length, embedding_fn
        )
    elif method == "dynamic":
        reduced_words, regret = optimal_reduction_dynamic(
            words, icf_scores, target_length, embedding_fn
        )
    elif method == "beam":
        reduced_words, regret = optimal_reduction_beam_search(
            words, icf_scores, target_length, embedding_fn
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced_text = ' '.join(reduced_words)
    
    stats = {
        'original_length': n_words,
        'reduced_length': len(reduced_words),
        'reduction_ratio': 1.0 - (len(reduced_words) / n_words),
        'avg_icf_dropped': np.mean([icf for i, icf in enumerate(icf_scores) if words[i] not in reduced_words]) if len(reduced_words) < n_words else 0.0,
        'avg_icf_kept': np.mean([icf for i, icf in enumerate(icf_scores) if words[i] in reduced_words]),
        'regret': regret,
    }
    
    return reduced_text, regret, stats

