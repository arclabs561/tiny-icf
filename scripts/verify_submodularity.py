#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.20.0",
#   "sentence-transformers>=2.2.0",
# ]
# ///

"""
Verify if embedding regret is submodular (or approximately submodular).

Submodularity test: f(S ∪ {v}) - f(S) ≥ f(T ∪ {v}) - f(T) for S ⊆ T

If embedding similarity is submodular, greedy algorithms have (1-1/e) guarantee.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import argparse
from pathlib import Path
import json

def compute_embedding_similarity(
    embeddings: torch.Tensor,
    original_embedding: torch.Tensor,
    word_indices: List[int],
) -> float:
    """Compute cosine similarity between average of selected embeddings and original."""
    if len(word_indices) == 0:
        return 0.0
    
    selected_embeddings = embeddings[word_indices]
    avg_embedding = selected_embeddings.mean(dim=0)
    
    cos_sim = torch.nn.functional.cosine_similarity(
        avg_embedding.unsqueeze(0),
        original_embedding.unsqueeze(0),
    )
    return cos_sim.item()

def test_submodularity(
    embeddings: torch.Tensor,
    original_embedding: torch.Tensor,
    n_samples: int = 100,
    k_max: int = 10,
) -> Dict[str, float]:
    """
    Test submodularity of embedding similarity function.
    
    For random subsets S ⊆ T and word v, check:
    f(S ∪ {v}) - f(S) ≥ f(T ∪ {v}) - f(T)
    
    Returns:
        - violation_rate: Fraction of tests that violate submodularity
        - avg_marginal_gain_S: Average marginal gain for smaller set
        - avg_marginal_gain_T: Average marginal gain for larger set
        - is_submodular: True if violation_rate < 0.05 (approximately submodular)
    """
    n_words = embeddings.shape[0]
    violations = 0
    marginal_gains_S = []
    marginal_gains_T = []
    
    for _ in range(n_samples):
        # Random subsets: S ⊆ T
        k_S = np.random.randint(1, min(k_max, n_words - 2))
        k_T = np.random.randint(k_S + 1, min(k_max + 1, n_words - 1))
        
        # Random word v not in T
        all_indices = list(range(n_words))
        T_indices = np.random.choice(all_indices, size=k_T, replace=False).tolist()
        S_indices = np.random.choice(T_indices, size=k_S, replace=False).tolist()
        
        remaining = [i for i in all_indices if i not in T_indices]
        if len(remaining) == 0:
            continue
        v = np.random.choice(remaining)
        
        # Compute marginal gains
        f_S = compute_embedding_similarity(embeddings, original_embedding, S_indices)
        f_S_plus_v = compute_embedding_similarity(embeddings, original_embedding, S_indices + [v])
        marginal_gain_S = f_S_plus_v - f_S
        
        f_T = compute_embedding_similarity(embeddings, original_embedding, T_indices)
        f_T_plus_v = compute_embedding_similarity(embeddings, original_embedding, T_indices + [v])
        marginal_gain_T = f_T_plus_v - f_T
        
        marginal_gains_S.append(marginal_gain_S)
        marginal_gains_T.append(marginal_gain_T)
        
        # Check submodularity: marginal_gain_S ≥ marginal_gain_T
        if marginal_gain_S < marginal_gain_T - 1e-6:  # Small tolerance for numerical errors
            violations += 1
    
    violation_rate = violations / n_samples
    avg_marginal_gain_S = np.mean(marginal_gains_S) if marginal_gains_S else 0.0
    avg_marginal_gain_T = np.mean(marginal_gains_T) if marginal_gains_T else 0.0
    
    return {
        'violation_rate': violation_rate,
        'avg_marginal_gain_S': avg_marginal_gain_S,
        'avg_marginal_gain_T': avg_marginal_gain_T,
        'is_submodular': violation_rate < 0.05,  # Approximately submodular if < 5% violations
        'is_approximately_submodular': violation_rate < 0.20,  # Approximately if < 20% violations
        'n_samples': n_samples,
        'n_violations': violations,
    }

def test_with_real_text(
    text: str,
    embedding_model: SentenceTransformer,
    n_samples: int = 100,
) -> Dict[str, any]:
    """Test submodularity with real text."""
    words = text.split()
    n_words = len(words)
    
    if n_words < 3:
        return {'error': 'Text too short for submodularity test'}
    
    # Get word embeddings
    word_embeddings = []
    for word in words:
        emb = embedding_model.encode(word, convert_to_tensor=True)
        word_embeddings.append(emb)
    embeddings = torch.stack(word_embeddings)
    
    # Get original text embedding
    original_embedding = embedding_model.encode(text, convert_to_tensor=True)
    
    # Test submodularity
    results = test_submodularity(embeddings, original_embedding, n_samples=n_samples)
    
    results['text_length'] = n_words
    results['embedding_dim'] = embeddings.shape[1]
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Verify submodularity of embedding regret')
    parser.add_argument('--text', type=str, default='the quick brown fox jumps over the lazy dog',
                       help='Text to test (default: example sentence)')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                       help='Sentence transformer model (default: all-MiniLM-L6-v2)')
    parser.add_argument('--n-samples', type=int, default=100,
                       help='Number of random samples for test (default: 100)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    print('🔬 Submodularity Verification\n')
    print('='*80)
    print(f'\n📝 Text: {args.text}')
    print(f'🤖 Model: {args.model}')
    print(f'📊 Samples: {args.n_samples}')
    print('\n⏳ Loading model and computing embeddings...')
    
    embedding_model = SentenceTransformer(args.model)
    
    print('⏳ Testing submodularity...')
    results = test_with_real_text(args.text, embedding_model, n_samples=args.n_samples)
    
    if 'error' in results:
        print(f'❌ Error: {results["error"]}')
        return
    
    print('\n📊 Results:')
    print('-'*80)
    print(f'Violation Rate: {results["violation_rate"]:.4f} ({results["n_violations"]}/{results["n_samples"]})')
    print(f'Avg Marginal Gain (Small Set): {results["avg_marginal_gain_S"]:.6f}')
    print(f'Avg Marginal Gain (Large Set): {results["avg_marginal_gain_T"]:.6f}')
    print(f'Is Submodular: {results["is_submodular"]}')
    print(f'Is Approximately Submodular: {results["is_approximately_submodular"]}')
    
    print('\n💡 Interpretation:')
    if results['is_submodular']:
        print('   ✅ Embedding similarity is submodular!')
        print('   ✅ Greedy algorithm has (1-1/e) ≈ 0.63 approximation guarantee')
    elif results['is_approximately_submodular']:
        print('   ⚠️  Embedding similarity is approximately submodular')
        print('   ⚠️  Greedy algorithm may work well, but no theoretical guarantee')
    else:
        print('   ❌ Embedding similarity is NOT submodular')
        print('   ❌ Greedy algorithm may not work well')
        print('   💡 Consider: Optimal Transport, coreset algorithms, or other methods')
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\n💾 Results saved to: {output_path}')

if __name__ == '__main__':
    main()

