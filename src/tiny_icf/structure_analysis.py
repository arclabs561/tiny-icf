"""Analyze structure in ICF function to validate compression hypothesis.

This module measures:
1. Correlation between character patterns and ICF
2. Mutual information: I(character_patterns; ICF)
3. Structure strength (can we compress?)
4. Generalization potential (can model learn structure?)
"""

import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
from scipy.stats import entropy


def compute_ngram_icf_correlation(
    word_icf: Dict[str, float],
    n: int = 3,
) -> Dict[str, float]:
    """
    Compute correlation between n-grams and ICF scores.
    
    For each n-gram, compute average ICF of words containing it.
    Then compute correlation: n-gram frequency vs average ICF.
    
    Args:
        word_icf: Dictionary mapping words to ICF scores
        n: N-gram size (default: 3 for trigrams)
    
    Returns:
        Dictionary mapping n-grams to (avg_icf, correlation)
    """
    ngram_to_words = defaultdict(list)
    ngram_to_icfs = defaultdict(list)
    
    # Collect n-grams and their ICFs
    for word, icf in word_icf.items():
        # Character n-grams
        for i in range(len(word) - n + 1):
            ngram = word[i:i+n].lower()
            ngram_to_words[ngram].append(word)
            ngram_to_icfs[ngram].append(icf)
    
    # Compute average ICF per n-gram
    ngram_avg_icf = {}
    ngram_freq = {}
    for ngram, icfs in ngram_to_icfs.items():
        ngram_avg_icf[ngram] = np.mean(icfs)
        ngram_freq[ngram] = len(icfs)
    
    # Compute correlation: n-gram frequency vs average ICF
    if len(ngram_freq) < 2:
        return {}
    
    freqs = list(ngram_freq.values())
    avg_icfs = [ngram_avg_icf[ngram] for ngram in ngram_freq.keys()]
    
    correlation, p_value = stats.pearsonr(freqs, avg_icfs)
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'ngram_count': len(ngram_freq),
        'top_ngrams': sorted(
            ngram_avg_icf.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20],
    }


def compute_mutual_information(
    word_icf: Dict[str, float],
    n: int = 3,
    bins: int = 10,
) -> float:
    """
    Compute mutual information I(character_patterns; ICF).
    
    Measures how much information character patterns provide about ICF.
    High MI = strong structure (patterns predict ICF well).
    Low MI = weak structure (patterns don't predict ICF).
    
    Args:
        word_icf: Dictionary mapping words to ICF scores
        n: N-gram size
        bins: Number of bins for ICF discretization
    
    Returns:
        Mutual information in bits
    """
    # Discretize ICF into bins
    icf_values = np.array(list(word_icf.values()))
    icf_bins = np.digitize(icf_values, np.linspace(0, 1, bins))
    
    # Create feature vectors: presence of n-grams
    ngram_to_idx = {}
    feature_matrix = []
    icf_bin_list = []
    
    for word, icf_bin in zip(word_icf.keys(), icf_bins):
        # Extract n-grams
        ngrams = set()
        for i in range(len(word) - n + 1):
            ngram = word[i:i+n].lower()
            ngrams.add(ngram)
            if ngram not in ngram_to_idx:
                ngram_to_idx[ngram] = len(ngram_to_idx)
        
        # Create binary feature vector
        features = np.zeros(len(ngram_to_idx))
        for ngram in ngrams:
            features[ngram_to_idx[ngram]] = 1.0
        
        feature_matrix.append(features)
        icf_bin_list.append(icf_bin)
    
    if len(feature_matrix) == 0:
        return 0.0
    
    feature_matrix = np.array(feature_matrix)
    icf_bin_array = np.array(icf_bin_list)
    
    # Compute MI: I(X; Y) = H(X) - H(X|Y)
    # Approximate using binning
    # H(ICF)
    icf_probs = np.bincount(icf_bin_array) / len(icf_bin_array)
    h_icf = entropy(icf_probs[icf_probs > 0], base=2)
    
    # H(ICF | patterns) - approximate by averaging conditional entropies
    # For simplicity, use feature similarity as proxy
    # More sophisticated: use clustering or density estimation
    h_icf_given_patterns = 0.0
    
    # Approximate: group similar feature vectors, compute entropy within groups
    # Simple approximation: use k-means or threshold-based grouping
    # For now, use a simple threshold: similar feature vectors (cosine similarity > 0.8)
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Group similar patterns
    similarity_matrix = cosine_similarity(feature_matrix)
    groups = []
    used = set()
    
    for i in range(len(feature_matrix)):
        if i in used:
            continue
        
        group = [i]
        used.add(i)
        
        for j in range(i+1, len(feature_matrix)):
            if j in used:
                continue
            if similarity_matrix[i, j] > 0.8:  # Similar patterns
                group.append(j)
                used.add(j)
        
        groups.append(group)
    
    # Compute conditional entropy: H(ICF | group)
    for group in groups:
        if len(group) < 2:
            continue
        
        group_icfs = icf_bin_array[group]
        group_probs = np.bincount(group_icfs) / len(group_icfs)
        h_group = entropy(group_probs[group_probs > 0], base=2)
        h_icf_given_patterns += h_group * (len(group) / len(feature_matrix))
    
    mi = h_icf - h_icf_given_patterns
    
    return max(0.0, mi)  # MI is non-negative


def estimate_structure_strength(
    word_icf: Dict[str, float],
) -> Dict[str, float]:
    """
    Estimate structure strength in ICF function.
    
    Measures:
    1. N-gram correlation (how well patterns predict ICF)
    2. Mutual information (information content of patterns about ICF)
    3. Compression potential (can we compress better than dict?)
    
    Args:
        word_icf: Dictionary mapping words to ICF scores
    
    Returns:
        Dictionary with structure strength metrics
    """
    results = {}
    
    # 1. N-gram correlation
    for n in [2, 3, 4]:
        corr_result = compute_ngram_icf_correlation(word_icf, n=n)
        if corr_result:
            results[f'ngram_{n}_correlation'] = corr_result['correlation']
            results[f'ngram_{n}_p_value'] = corr_result['p_value']
    
    # 2. Mutual information
    try:
        mi = compute_mutual_information(word_icf, n=3, bins=10)
        results['mutual_information'] = mi
    except Exception as e:
        results['mutual_information'] = None
        results['mutual_information_error'] = str(e)
    
    # 3. Compression potential
    # Estimate: H(ICF) vs actual size
    icf_values = np.array(list(word_icf.values()))
    icf_probs = np.histogram(icf_values, bins=20, density=True)[0]
    icf_probs = icf_probs / icf_probs.sum()  # Normalize
    h_icf = entropy(icf_probs[icf_probs > 0], base=2)  # Bits per word
    
    # Actual size: V × 32 bits (if stored as float32)
    v = len(word_icf)
    actual_size_bits = v * 32
    theoretical_min_bits = v * h_icf
    
    compression_potential = actual_size_bits / theoretical_min_bits if theoretical_min_bits > 0 else 0.0
    
    results['shannon_entropy'] = h_icf
    results['compression_potential'] = compression_potential
    results['vocabulary_size'] = v
    
    # 4. Structure strength score (0-1)
    # Combine correlation and MI
    avg_correlation = np.mean([
        results.get('ngram_2_correlation', 0),
        results.get('ngram_3_correlation', 0),
        results.get('ngram_4_correlation', 0),
    ])
    
    mi_normalized = results.get('mutual_information', 0) / 10.0  # Normalize (max MI ≈ 10 bits)
    mi_normalized = min(1.0, max(0.0, mi_normalized))
    
    structure_strength = (abs(avg_correlation) + mi_normalized) / 2.0
    
    results['structure_strength'] = structure_strength
    results['interpretation'] = (
        'strong' if structure_strength > 0.7 else
        'moderate' if structure_strength > 0.4 else
        'weak'
    )
    
    return results


def test_generalization(
    word_icf: Dict[str, float],
    train_ratio: float = 0.8,
) -> Dict[str, float]:
    """
    Test if structure generalizes to unseen words.
    
    Split words into train/test, compute structure metrics on both.
    If structure is real, metrics should be similar.
    If structure is spurious, metrics will differ.
    
    Args:
        word_icf: Dictionary mapping words to ICF scores
        train_ratio: Fraction of words for training
    
    Returns:
        Dictionary with generalization metrics
    """
    words = list(word_icf.keys())
    np.random.shuffle(words)
    
    split_idx = int(len(words) * train_ratio)
    train_words = words[:split_idx]
    test_words = words[split_idx:]
    
    train_icf = {w: word_icf[w] for w in train_words}
    test_icf = {w: word_icf[w] for w in test_words}
    
    # Compute structure on train
    train_structure = estimate_structure_strength(train_icf)
    
    # Compute structure on test
    test_structure = estimate_structure_strength(test_icf)
    
    # Compare
    results = {
        'train_structure_strength': train_structure.get('structure_strength', 0),
        'test_structure_strength': test_structure.get('structure_strength', 0),
        'generalization_gap': abs(
            train_structure.get('structure_strength', 0) -
            test_structure.get('structure_strength', 0)
        ),
        'generalizes': abs(
            train_structure.get('structure_strength', 0) -
            test_structure.get('structure_strength', 0)
        ) < 0.1,  # Gap < 0.1 = generalizes
    }
    
    return results


if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from tiny_icf.data import load_frequency_list, compute_normalized_icf
    
    # Try multiple data paths
    data_paths = [
        Path('data/word_frequency.csv'),
        Path('../data/word_frequency.csv'),
        Path('word_frequency.csv'),
    ]
    
    data_path = None
    for path in data_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path and data_path.exists():
        print(f"Loading data from: {data_path}")
        word_counts, total_tokens = load_frequency_list(data_path)
        word_icf = compute_normalized_icf(word_counts, total_tokens)
        
        print(f"\nLoaded {len(word_icf):,} words, {total_tokens:,} total tokens")
        
        # Analyze structure
        print("\n" + "="*70)
        print("Analyzing ICF Structure Strength")
        print("="*70)
        structure = estimate_structure_strength(word_icf)
        
        print("\nStructure Strength Metrics:")
        for key, value in structure.items():
            if key != 'top_ngrams':
                print(f"  {key}: {value}")
        
        if 'top_ngrams' in structure:
            print(f"\n  Top 10 n-grams by average ICF:")
            for ngram, avg_icf in structure['top_ngrams'][:10]:
                print(f"    {ngram}: {avg_icf:.3f}")
        
        # Test generalization
        print("\n" + "="*70)
        print("Testing Generalization")
        print("="*70)
        gen_results = test_generalization(word_icf)
        
        print("\nGeneralization Results:")
        for key, value in gen_results.items():
            print(f"  {key}: {value}")
        
        # Summary
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        strength = structure.get('structure_strength', 0)
        interpretation = structure.get('interpretation', 'unknown')
        generalizes = gen_results.get('generalizes', False)
        
        print(f"\nStructure Strength: {strength:.3f} ({interpretation})")
        print(f"Generalizes: {generalizes}")
        
        if strength > 0.7 and generalizes:
            print("\n✓ Strong structure detected - compression is feasible")
            print("  Model should be able to learn structure and generalize")
        elif strength > 0.4:
            print("\n⚠️  Moderate structure - compression may be marginal")
            print("  Model may struggle to learn structure")
        else:
            print("\n✗ Weak structure - compression may not be possible")
            print("  Model may need to memorize (no compression advantage)")
    else:
        print("Data file not found. Tried:")
        for path in data_paths:
            print(f"  - {path}")
        print("\nPlease provide a frequency list CSV file.")
        print("Expected format: word,count (one per line)")

