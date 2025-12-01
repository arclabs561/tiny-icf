"""Multilingual-aware data loading and ICF computation."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def compute_icf_per_language(
    word_counts: Dict[str, int], min_count: int = 5
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Compute ICF separately for each language.
    
    For multilingual data with language prefixes (en:word, es:palabra),
    computes ICF per language to avoid mixing corpora.
    
    Returns:
        Tuple of (icf_scores dict, language_totals dict)
    """
    # Group by language
    languages = defaultdict(dict)
    language_totals = {}
    
    for word, count in word_counts.items():
        if ':' in word:
            lang, base_word = word.split(':', 1)
            languages[lang][base_word] = count
            language_totals[lang] = language_totals.get(lang, 0) + count
        else:
            # No language prefix - treat as default language
            languages['default'][word] = count
            language_totals['default'] = language_totals.get('default', 0) + count
    
    # Compute ICF per language with smoothing
    icf_scores = {}
    
    for lang, lang_counts in languages.items():
        total_tokens = language_totals[lang]
        # Add 1 smoothing to prevent edge cases
        log_total = math.log(total_tokens + 1)
        
        for base_word, count in lang_counts.items():
            # Reconstruct full key with language prefix
            key = f"{lang}:{base_word}" if lang != 'default' else base_word
            
            if count < min_count:
                icf_scores[key] = 1.0
            elif count >= total_tokens:
                # Most common word (handle gracefully)
                icf_scores[key] = 0.0
            else:
                # Normalized ICF with smoothing
                icf = math.log((total_tokens + 1) / (count + 1)) / log_total
                icf_scores[key] = max(0.0, min(1.0, icf))
    
    return icf_scores, language_totals


def stratified_sample_multilingual(
    word_icf: Dict[str, float],
    head_size: int = 10000,
    body_size: int = 100000,
    head_prob: float = 0.4,
    body_prob: float = 0.3,
    balance_languages: bool = True,
) -> List[Tuple[str, float]]:
    """
    Stratified sampling with language balancing for multilingual data.
    
    If balance_languages=True, ensures each language is represented
    proportionally in the sample.
    """
    # Group by language
    languages = defaultdict(list)
    
    for word, icf in word_icf.items():
        if ':' in word:
            lang = word.split(':', 1)[0]
        else:
            lang = 'default'
        languages[lang].append((word, icf))
    
    # Sample per language, then combine
    if balance_languages and len(languages) > 1:
        samples = []
        samples_per_lang = {}
        
        # Calculate samples per language (proportional to size)
        total_words = len(word_icf)
        for lang, lang_words in languages.items():
            lang_ratio = len(lang_words) / total_words
            samples_per_lang[lang] = int(total_words * lang_ratio * (head_prob + body_prob + (1 - head_prob - body_prob)))
        
        # Sample from each language
        for lang, lang_words in languages.items():
            lang_samples = stratified_sample_single_language(
                dict(lang_words),
                head_size=head_size,
                body_size=body_size,
                head_prob=head_prob,
                body_prob=body_prob,
            )
            # Limit to proportional amount
            n_samples = min(samples_per_lang.get(lang, len(lang_samples)), len(lang_samples))
            samples.extend(lang_samples[:n_samples])
        
        return samples
    else:
        # Single language or no balancing
        return stratified_sample_single_language(
            word_icf,
            head_size=head_size,
            body_size=body_size,
            head_prob=head_prob,
            body_prob=body_prob,
        )


def stratified_sample_single_language(
    word_icf: Dict[str, float],
    head_size: int = 10000,
    body_size: int = 100000,
    head_prob: float = 0.4,
    body_prob: float = 0.3,
) -> List[Tuple[str, float]]:
    """Stratified sampling for single language (original implementation)."""
    sorted_words = sorted(word_icf.items(), key=lambda x: x[1])
    
    head = sorted_words[:head_size]
    body = sorted_words[head_size:body_size] if len(sorted_words) > head_size else []
    tail = sorted_words[body_size:] if len(sorted_words) > body_size else []
    
    # Sample according to probabilities
    n_total = len(sorted_words)
    n_head = int(n_total * head_prob)
    n_body = int(n_total * body_prob)
    n_tail = n_total - n_head - n_body
    
    head_samples = []
    if head:
        head_indices = np.random.choice(len(head), size=min(n_head, len(head)), replace=False)
        head_samples = [head[i] for i in head_indices]
    
    body_samples = []
    if body:
        body_indices = np.random.choice(len(body), size=min(n_body, len(body)), replace=False)
        body_samples = [body[i] for i in body_indices]
    
    tail_samples = []
    if tail:
        tail_indices = np.random.choice(len(tail), size=min(n_tail, len(tail)), replace=False)
        tail_samples = [tail[i] for i in tail_indices]
    
    return head_samples + body_samples + tail_samples

