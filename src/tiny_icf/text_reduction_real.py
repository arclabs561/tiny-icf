"""Optimal text reduction using ICF with real embeddings (sentence-transformers).

Uses actual embedding models to compute regret, making the reduction
truly optimal for preserving semantic information.
"""

import torch
from typing import List, Tuple, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")


def compute_embedding_difference(
    original_embedding: torch.Tensor,
    reduced_embedding: torch.Tensor,
    metric: str = "cosine",
) -> float:
    """
    Compute embedding difference (regret) between original and reduced text.
    
    Args:
        original_embedding: [dim] embedding of original text
        reduced_embedding: [dim] embedding of reduced text
        metric: "cosine" or "l2"
    
    Returns:
        Regret score (higher = more information lost)
    """
    if metric == "cosine":
        # Cosine distance: 1 - cosine_similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            original_embedding.unsqueeze(0),
            reduced_embedding.unsqueeze(0),
        ).item()
        regret = 1.0 - cos_sim
    else:  # l2
        regret = torch.norm(original_embedding - reduced_embedding).item()
    
    return regret


class ICFTextReducer:
    """
    Text reducer using ICF scores to minimize embedding regret.
    
    Uses real embeddings (sentence-transformers) to compute actual
    semantic similarity and guide optimal word dropping.
    """
    
    def __init__(
        self,
        icf_model,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            icf_model: Trained ICF prediction model
            embedding_model_name: Sentence transformer model name
            device: Device for computation
        """
        self.icf_model = icf_model
        self.icf_model.eval()
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.device = device or torch.device("cpu")
        
    def predict_icf(self, word: str) -> float:
        """Predict ICF score for a word."""
        # Clean word
        clean_word = ''.join(c for c in word.lower() if c.isalnum())
        if not clean_word:
            return 0.0
        
        byte_seq = clean_word.encode('utf-8')[:20]
        padded = byte_seq + bytes(20 - len(byte_seq))
        byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            icf = self.icf_model(byte_tensor).item()
        
        return icf
    
    def compute_embedding(self, text: str) -> torch.Tensor:
        """Compute embedding for text."""
        embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        return embedding
    
    def reduce_greedy_icf(
        self,
        text: str,
        target_ratio: float = 0.5,
    ) -> Tuple[str, float, dict]:
        """
        Greedy reduction: Drop lowest ICF words first.
        
        Simple but effective: common words (low ICF) contribute less
        to semantics, so drop them first.
        
        Args:
            text: Input text
            target_ratio: Fraction of words to keep
        
        Returns:
            (reduced_text, regret, stats)
        """
        words = text.split()
        n_words = len(words)
        target_length = max(1, int(n_words * target_ratio))
        
        if n_words <= target_length:
            return text, 0.0, {
                'original_length': n_words,
                'reduced_length': n_words,
                'reduction_ratio': 0.0,
                'regret': 0.0,
            }
        
        # Predict ICF for each word
        icf_scores = [self.predict_icf(word) for word in words]
        
        # Sort by ICF (ascending: common first)
        word_icf_pairs = list(zip(words, icf_scores))
        word_icf_pairs.sort(key=lambda x: x[1])  # Sort by ICF
        
        # Keep top N words (highest ICF = most informative)
        keep_pairs = word_icf_pairs[-target_length:]
        reduced_words = [word for word, _ in keep_pairs]
        reduced_text = ' '.join(reduced_words)
        
        # Compute regret using real embeddings
        original_embedding = self.compute_embedding(text)
        reduced_embedding = self.compute_embedding(reduced_text)
        regret = compute_embedding_difference(original_embedding, reduced_embedding)
        
        stats = {
            'original_length': n_words,
            'reduced_length': len(reduced_words),
            'reduction_ratio': 1.0 - (len(reduced_words) / n_words),
            'avg_icf_kept': np.mean([icf for _, icf in keep_pairs]),
            'avg_icf_dropped': np.mean([icf for _, icf in word_icf_pairs[:-target_length]]) if target_length < n_words else 0.0,
            'regret': regret,
        }
        
        return reduced_text, regret, stats
    
    def reduce_optimal_regret(
        self,
        text: str,
        target_ratio: float = 0.5,
        max_iterations: int = 100,
    ) -> Tuple[str, float, dict]:
        """
        Optimal reduction: Iteratively drop word that causes least regret.
        
        More thorough than greedy: tries dropping each word and picks
        the one that minimizes embedding regret.
        
        Args:
            text: Input text
            target_ratio: Fraction of words to keep
            max_iterations: Max iterations (safety limit)
        
        Returns:
            (reduced_text, regret, stats)
        """
        words = text.split()
        n_words = len(words)
        target_length = max(1, int(n_words * target_ratio))
        
        if n_words <= target_length:
            return text, 0.0, {
                'original_length': n_words,
                'reduced_length': n_words,
                'reduction_ratio': 0.0,
                'regret': 0.0,
            }
        
        # Compute original embedding once
        original_embedding = self.compute_embedding(text)
        
        # Predict ICF for all words
        icf_scores = [self.predict_icf(word) for word in words]
        
        # Greedy iterative reduction
        current_words = words.copy()
        current_icf = icf_scores.copy()
        iterations = 0
        
        while len(current_words) > target_length and iterations < max_iterations:
            min_regret = float('inf')
            best_idx = -1
            
            # Try dropping each word, find one with least regret
            for i in range(len(current_words)):
                # Skip high-ICF words (we want to keep those)
                if current_icf[i] > 0.7:
                    continue
                
                # Try dropping word i
                test_words = current_words[:i] + current_words[i+1:]
                test_text = ' '.join(test_words)
                test_embedding = self.compute_embedding(test_text)
                regret = compute_embedding_difference(original_embedding, test_embedding)
                
                # Weight by ICF: prefer dropping low ICF words
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
            
            iterations += 1
        
        reduced_text = ' '.join(current_words)
        
        # Compute final regret
        reduced_embedding = self.compute_embedding(reduced_text)
        final_regret = compute_embedding_difference(original_embedding, reduced_embedding)
        
        stats = {
            'original_length': n_words,
            'reduced_length': len(current_words),
            'reduction_ratio': 1.0 - (len(current_words) / n_words),
            'avg_icf_kept': np.mean(current_icf) if current_icf else 0.0,
            'regret': final_regret,
            'iterations': iterations,
        }
        
        return reduced_text, final_regret, stats


def reduce_text_with_real_embeddings(
    text: str,
    icf_model,
    target_ratio: float = 0.5,
    method: str = "greedy",
    embedding_model: str = "all-MiniLM-L6-v2",
    device: Optional[torch.device] = None,
) -> Tuple[str, float, dict]:
    """
    Reduce text using ICF-guided optimal reduction with real embeddings.
    
    Args:
        text: Input text to reduce
        icf_model: ICF prediction model
        target_ratio: Fraction of words to keep (0.5 = keep 50%)
        method: "greedy" or "optimal"
        embedding_model: Sentence transformer model name
        device: Device for computation
    
    Returns:
        (reduced_text, regret, stats)
    """
    reducer = ICFTextReducer(icf_model, embedding_model, device)
    
    if method == "greedy":
        return reducer.reduce_greedy_icf(text, target_ratio)
    elif method == "optimal":
        return reducer.reduce_optimal_regret(text, target_ratio)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'greedy' or 'optimal'")

