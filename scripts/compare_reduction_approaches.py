#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.20.0",
#   "sentence-transformers>=2.2.0",
#   "pandas>=2.0.0",
# ]
# ///

"""
Compare ICF-based vs embedding-based vs hybrid text reduction approaches.

Tests:
1. ICF-based: Rank by ICF scores, keep top k
2. Embedding-based greedy: Greedily add words maximizing embedding similarity
3. Embedding-based optimal: Try all combinations (small k only)
4. Hybrid: Use ICF for initial ranking, embedding for fine-tuning
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
import argparse
from pathlib import Path
import json
import pandas as pd
import sys

# Add src to path for model loading
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def compute_embedding_regret(
    original_embedding: torch.Tensor,
    reduced_embedding: torch.Tensor,
) -> float:
    """Compute embedding regret (1 - cosine similarity)."""
    cos_sim = torch.nn.functional.cosine_similarity(
        original_embedding.unsqueeze(0),
        reduced_embedding.unsqueeze(0),
    )
    return (1.0 - cos_sim.item())

def icf_based_reduction(
    words: List[str],
    icf_scores: List[float],
    k: int,
    embeddings: torch.Tensor,
    original_embedding: torch.Tensor,
) -> Tuple[List[str], float]:
    """ICF-based: Rank by ICF, keep top k."""
    # Sort by ICF (descending: high ICF = important)
    word_icf_pairs = list(zip(words, icf_scores, range(len(words))))
    word_icf_pairs.sort(key=lambda x: -x[1])  # Sort by ICF descending
    
    # Keep top k
    selected_pairs = word_icf_pairs[:k]
    selected_words = [w for w, _, _ in selected_pairs]
    selected_indices = [i for _, _, i in selected_pairs]
    
    # Compute regret
    selected_embeddings = embeddings[selected_indices]
    reduced_embedding = selected_embeddings.mean(dim=0)
    regret = compute_embedding_regret(original_embedding, reduced_embedding)
    
    return selected_words, regret

def embedding_based_greedy(
    words: List[str],
    k: int,
    embeddings: torch.Tensor,
    original_embedding: torch.Tensor,
) -> Tuple[List[str], float]:
    """Embedding-based greedy: Iteratively add word maximizing similarity."""
    selected_indices = []
    remaining_indices = list(range(len(words)))
    
    for _ in range(k):
        best_idx = None
        best_similarity = -1.0
        
        for idx in remaining_indices:
            candidate_indices = selected_indices + [idx]
            candidate_embeddings = embeddings[candidate_indices]
            candidate_embedding = candidate_embeddings.mean(dim=0)
            
            similarity = torch.nn.functional.cosine_similarity(
                candidate_embedding.unsqueeze(0),
                original_embedding.unsqueeze(0),
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    selected_words = [words[i] for i in selected_indices]
    selected_embeddings = embeddings[selected_indices]
    reduced_embedding = selected_embeddings.mean(dim=0)
    regret = compute_embedding_regret(original_embedding, reduced_embedding)
    
    return selected_words, regret

def hybrid_reduction(
    words: List[str],
    icf_scores: List[float],
    k: int,
    embeddings: torch.Tensor,
    original_embedding: torch.Tensor,
    icf_top_n: int = None,
) -> Tuple[List[str], float]:
    """Hybrid: Use ICF for initial ranking, embedding greedy for fine-tuning."""
    if icf_top_n is None:
        icf_top_n = min(2 * k, len(words))  # Consider top 2k by ICF
    
    # Step 1: Rank by ICF, get top candidates
    word_icf_pairs = list(zip(words, icf_scores, range(len(words))))
    word_icf_pairs.sort(key=lambda x: -x[1])  # Sort by ICF descending
    candidate_pairs = word_icf_pairs[:icf_top_n]
    candidate_indices = [i for _, _, i in candidate_pairs]
    
    # Step 2: Greedy selection from candidates
    selected_indices = []
    remaining_candidates = candidate_indices.copy()
    
    for _ in range(k):
        best_idx = None
        best_similarity = -1.0
        
        for idx in remaining_candidates:
            candidate_indices = selected_indices + [idx]
            candidate_embeddings = embeddings[candidate_indices]
            candidate_embedding = candidate_embeddings.mean(dim=0)
            
            similarity = torch.nn.functional.cosine_similarity(
                candidate_embedding.unsqueeze(0),
                original_embedding.unsqueeze(0),
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
        
        selected_indices.append(best_idx)
        remaining_candidates.remove(best_idx)
    
    selected_words = [words[i] for i in selected_indices]
    selected_embeddings = embeddings[selected_indices]
    reduced_embedding = selected_embeddings.mean(dim=0)
    regret = compute_embedding_regret(original_embedding, reduced_embedding)
    
    return selected_words, regret

def compare_approaches(
    text: str,
    icf_scores: List[float],
    embedding_model: SentenceTransformer,
    k: int,
) -> Dict[str, any]:
    """Compare all reduction approaches."""
    words = text.split()
    n_words = len(words)
    
    if n_words < k:
        return {'error': f'Text too short: {n_words} words, need at least {k}'}
    
    # Get embeddings
    word_embeddings = []
    for word in words:
        emb = embedding_model.encode(word, convert_to_tensor=True)
        word_embeddings.append(emb)
    embeddings = torch.stack(word_embeddings)
    
    original_embedding = embedding_model.encode(text, convert_to_tensor=True)
    
    # Test each approach
    results = {}
    
    # 1. ICF-based
    icf_words, icf_regret = icf_based_reduction(
        words, icf_scores, k, embeddings, original_embedding
    )
    results['icf_based'] = {
        'words': icf_words,
        'regret': icf_regret,
        'method': 'ICF ranking (top k by ICF)',
    }
    
    # 2. Embedding-based greedy
    greedy_words, greedy_regret = embedding_based_greedy(
        words, k, embeddings, original_embedding
    )
    results['embedding_greedy'] = {
        'words': greedy_words,
        'regret': greedy_regret,
        'method': 'Greedy embedding optimization',
    }
    
    # 3. Hybrid
    hybrid_words, hybrid_regret = hybrid_reduction(
        words, icf_scores, k, embeddings, original_embedding
    )
    results['hybrid'] = {
        'words': hybrid_words,
        'regret': hybrid_regret,
        'method': 'ICF pre-filter + embedding greedy',
    }
    
    # Summary
    best_method = min(results.items(), key=lambda x: x[1]['regret'])
    results['summary'] = {
        'best_method': best_method[0],
        'best_regret': best_method[1]['regret'],
        'icf_regret': icf_regret,
        'greedy_regret': greedy_regret,
        'hybrid_regret': hybrid_regret,
        'improvement_icf_vs_greedy': (icf_regret - greedy_regret) / icf_regret if icf_regret > 0 else 0.0,
        'improvement_icf_vs_hybrid': (icf_regret - hybrid_regret) / icf_regret if icf_regret > 0 else 0.0,
    }
    
    return results

def load_icf_model(model_path: Optional[Path], device: torch.device) -> Optional[torch.nn.Module]:
    """Load ICF model from checkpoint (supports both .pt and .ckpt formats)."""
    if model_path is None or not model_path.exists():
        return None
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle PyTorch Lightning checkpoints (.ckpt)
        if 'state_dict' in checkpoint:
            # Lightning checkpoint - extract model state
            state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                # Remove 'model.' prefix if present (Lightning wraps model)
                new_key = key.replace('model.', '') if key.startswith('model.') else key
                state_dict[new_key] = value
            checkpoint = state_dict
        
        # Try to infer model type
        model_type = None
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
        elif 'hyper_parameters' in checkpoint:
            # Lightning checkpoint may have hyper_parameters
            hparams = checkpoint.get('hyper_parameters', {})
            model_type = hparams.get('model_type')
        
        # Create model
        if model_type == 'ResidualICF':
            from tiny_icf.model_residual import ResidualICF
            model = ResidualICF().to(device)
        elif model_type == 'NanoICF':
            from tiny_icf.nano_model import NanoICF
            model = NanoICF().to(device)
        else:
            # Default to UniversalICF
            from tiny_icf.model import UniversalICF
            model = UniversalICF().to(device)
        
        # Load state dict (handle both direct state_dict and wrapped)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            # Assume checkpoint is the state dict itself
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️  Warning: Failed to load ICF model: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_icf_batch(
    words: List[str],
    model: torch.nn.Module,
    device: torch.device,
    max_length: int = 20,
) -> List[float]:
    """Predict ICF scores for a batch of words."""
    model.eval()
    scores = []
    
    with torch.no_grad():
        for word in words:
            # Convert word to byte tensor
            byte_seq = word.encode('utf-8')[:max_length]
            padded = byte_seq + bytes(max_length - len(byte_seq))
            byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0).to(device)
            
            # Predict
            icf = model(byte_tensor).item()
            scores.append(float(icf))
    
    return scores

def main():
    parser = argparse.ArgumentParser(description='Compare text reduction approaches')
    parser.add_argument('--text', type=str, default='the quick brown fox jumps over the lazy dog',
                       help='Text to reduce')
    parser.add_argument('--icf-scores', type=str, default=None,
                       help='Comma-separated ICF scores (or use synthetic/model)')
    parser.add_argument('--icf-model', type=str, default=None,
                       help='Path to trained ICF model checkpoint (optional)')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of words to keep (default: 5)')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                       help='Sentence transformer model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parse or generate ICF scores
    if args.icf_scores:
        icf_scores = [float(x) for x in args.icf_scores.split(',')]
        icf_source = "user-provided"
    elif args.icf_model:
        # Load ICF model and predict
        model_path = Path(args.icf_model)
        icf_model = load_icf_model(model_path, device)
        if icf_model is None:
            print("⚠️  Failed to load ICF model, falling back to synthetic scores")
            words = args.text.split()
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            icf_scores = [0.1 if w.lower() in common_words else 0.5 + np.random.random() * 0.4 for w in words]
            icf_source = "synthetic (fallback)"
        else:
            words = args.text.split()
            icf_scores = predict_icf_batch(words, icf_model, device)
            icf_source = f"ICF model ({model_path.name})"
    else:
        # Synthetic: common words = low ICF, rare words = high ICF
        words = args.text.split()
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        icf_scores = [0.1 if w.lower() in common_words else 0.5 + np.random.random() * 0.4 for w in words]
        icf_source = "synthetic"
    
    print('🔬 Text Reduction Comparison\n')
    print('='*80)
    print(f'\n📝 Text: {args.text}')
    print(f'📊 Words: {len(args.text.split())}')
    print(f'🎯 Keep: {args.k} words')
    print(f'🤖 Embedding Model: {args.model}')
    print(f'📈 ICF Scores: {icf_source}')
    
    embedding_model = SentenceTransformer(args.model)
    
    print('\n⏳ Comparing approaches...')
    results = compare_approaches(args.text, icf_scores, embedding_model, args.k)
    
    if 'error' in results:
        print(f'❌ Error: {results["error"]}')
        return
    
    print('\n📊 Results:')
    print('-'*80)
    print(f"ICF-Based Regret:      {results['icf_based']['regret']:.6f}")
    print(f"Greedy Regret:         {results['embedding_greedy']['regret']:.6f}")
    print(f"Hybrid Regret:         {results['hybrid']['regret']:.6f}")
    print(f"\nBest Method:           {results['summary']['best_method']}")
    print(f"Best Regret:           {results['summary']['best_regret']:.6f}")
    
    improvement_icf = results['summary']['improvement_icf_vs_greedy'] * 100
    improvement_hybrid = results['summary']['improvement_icf_vs_hybrid'] * 100
    print(f"\nGreedy vs ICF:         {improvement_icf:+.1f}% improvement")
    print(f"Hybrid vs ICF:         {improvement_hybrid:+.1f}% improvement")
    
    print('\n💡 Selected Words:')
    print(f"  ICF-Based:  {' '.join(results['icf_based']['words'])}")
    print(f"  Greedy:     {' '.join(results['embedding_greedy']['words'])}")
    print(f"  Hybrid:     {' '.join(results['hybrid']['words'])}")
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\n💾 Results saved to: {output_path}')

if __name__ == '__main__':
    main()

