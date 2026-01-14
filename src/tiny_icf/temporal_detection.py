"""Temporal/era detection: predict when words were used, historical period, usage span."""

import re
from typing import List, Dict, Tuple, Optional
from collections import Counter


# Historical era patterns
ERA_PATTERNS = {
    'archaic': {
        'suffixes': ['-eth', '-est', '-th', '-st'],
        'prefixes': ['ye', 'thou', 'thy', 'thee'],
        'patterns': [r'^ye\w+', r'\w+eth$', r'\w+est$', r'^thou\w*'],
        'examples': ['thou', 'thee', 'thy', 'hast', 'doth', 'hath', 'art', 'wilt', 'shalt'],
    },
    'early_modern': {
        'suffixes': ['-ed', '-ing', '-tion'],
        'patterns': [r'\w+ed$', r'\w+ing$', r'\w+tion$'],
        'examples': ['hath', 'doth', 'wherefore', 'hence', 'whence'],
    },
    'modern': {
        'suffixes': ['-ing', '-ed', '-ly', '-tion', '-sion'],
        'patterns': [r'\w+ing$', r'\w+ed$', r'\w+ly$'],
        'examples': ['computer', 'internet', 'technology'],
    },
    'contemporary': {
        'suffixes': ['-ing', '-ed'],
        'patterns': [r'\w+ing$', r'\w+ed$'],
        'examples': ['selfie', 'tweet', 'blog', 'app', 'emoji', 'meme'],
    },
    'neologism': {
        'patterns': [r'^[a-z]+[0-9]+', r'^[a-z]+[A-Z]', r'^[A-Z]+[a-z]'],
        'examples': ['iPhone', 'YouTube', 'eBay', 'iPod', 'WiFi'],
    },
}


# Temporal usage indicators
TEMPORAL_INDICATORS = {
    'very_old': {
        'score_range': (0.9, 1.0),
        'indicators': ['archaic', 'early_modern'],
        'description': 'Archaic or very old usage (pre-1800s)',
    },
    'old': {
        'score_range': (0.7, 0.9),
        'indicators': ['early_modern'],
        'description': 'Older usage (1800s-early 1900s)',
    },
    'classic': {
        'score_range': (0.4, 0.7),
        'indicators': ['modern'],
        'description': 'Classic/modern usage (1900s-2000s)',
    },
    'recent': {
        'score_range': (0.2, 0.4),
        'indicators': ['contemporary', 'neologism'],
        'description': 'Recent usage (2000s-2010s)',
    },
    'very_recent': {
        'score_range': (0.0, 0.2),
        'indicators': ['contemporary', 'neologism'],
        'description': 'Very recent usage (2010s-present)',
    },
}


def detect_era_patterns(word: str) -> Dict[str, float]:
    """
    Detect historical era patterns in word.
    
    Returns:
        Dictionary mapping era names to confidence scores
    """
    word_lower = word.lower()
    scores = {}
    
    for era, patterns in ERA_PATTERNS.items():
        score = 0.0
        
        # Check suffixes
        suffixes = patterns.get('suffixes', [])
        for suffix in suffixes:
            if word_lower.endswith(suffix):
                score += 0.3
        
        # Check prefixes
        prefixes = patterns.get('prefixes', [])
        for prefix in prefixes:
            if word_lower.startswith(prefix):
                score += 0.3
        
        # Check regex patterns
        regex_patterns = patterns.get('patterns', [])
        for pattern in regex_patterns:
            if re.match(pattern, word_lower):
                score += 0.2
        
        # Check examples
        examples = patterns.get('examples', [])
        if word_lower in [ex.lower() for ex in examples]:
            score += 0.5
        
        if score > 0:
            scores[era] = score
    
    # Normalize
    if scores:
        total = sum(scores.values())
        if total > 0:
            scores = {era: score / total for era, score in scores.items()}
    
    return scores


def detect_temporal_usage(word: str, icf_score: Optional[float] = None) -> Dict[str, any]:
    """
    Detect temporal usage characteristics of a word.
    
    Args:
        word: Word to analyze
        icf_score: Optional ICF score (rare words often newer/older)
    
    Returns:
        Dictionary with temporal analysis
    """
    era_patterns = detect_era_patterns(word)
    
    # Determine primary era
    primary_era = max(era_patterns.items(), key=lambda x: x[1])[0] if era_patterns else 'modern'
    
    # Estimate usage span based on patterns
    usage_span = 'unknown'
    if primary_era == 'archaic':
        usage_span = 'pre-1800s'
    elif primary_era == 'early_modern':
        usage_span = '1800s-early_1900s'
    elif primary_era == 'modern':
        usage_span = '1900s-2000s'
    elif primary_era == 'contemporary':
        usage_span = '2000s-present'
    elif primary_era == 'neologism':
        usage_span = '2010s-present'
    
    # Estimate temporal category
    temporal_category = 'classic'
    if icf_score is not None:
        for category, info in TEMPORAL_INDICATORS.items():
            min_score, max_score = info['score_range']
            if min_score <= icf_score <= max_score:
                if primary_era in info['indicators']:
                    temporal_category = category
                    break
    
    # Estimate approximate era
    approximate_era = 'unknown'
    if primary_era == 'archaic':
        approximate_era = 'pre-1800s'
    elif primary_era == 'early_modern':
        approximate_era = '1800s'
    elif primary_era == 'modern':
        approximate_era = '1900s-2000s'
    elif primary_era == 'contemporary':
        approximate_era = '2000s-2010s'
    elif primary_era == 'neologism':
        approximate_era = '2010s-present'
    
    return {
        'primary_era': primary_era,
        'era_confidence': float(era_patterns.get(primary_era, 0.0)),
        'all_eras': {era: float(conf) for era, conf in era_patterns.items()},
        'usage_span': usage_span,
        'temporal_category': temporal_category,
        'approximate_era': approximate_era,
        'is_archaic': primary_era == 'archaic',
        'is_neologism': primary_era == 'neologism',
        'is_contemporary': primary_era in ['contemporary', 'neologism'],
    }


def estimate_usage_period(word: str, icf_score: Optional[float] = None) -> Dict[str, any]:
    """
    Estimate when a word was most commonly used.
    
    Returns:
        Dictionary with usage period estimates
    """
    temporal = detect_temporal_usage(word, icf_score)
    
    # Refine based on word characteristics
    word_lower = word.lower()
    
    # Check for modern technology terms
    tech_indicators = ['app', 'web', 'net', 'tech', 'digital', 'online', 'cyber', 'e-', 'i-']
    is_tech_term = any(indicator in word_lower for indicator in tech_indicators)
    
    # Check for social media terms
    social_indicators = ['tweet', 'post', 'share', 'like', 'follow', 'friend', 'unfriend']
    is_social_term = any(indicator in word_lower for indicator in social_indicators)
    
    # Adjust estimates
    if is_tech_term or is_social_term:
        temporal['approximate_era'] = '2000s-present'
        temporal['usage_span'] = '2000s-present'
        temporal['temporal_category'] = 'very_recent'
        temporal['is_contemporary'] = True
    
    # Check for compound words (often modern)
    if '-' in word or '_' in word:
        temporal['is_contemporary'] = True
        if temporal['approximate_era'] == 'unknown':
            temporal['approximate_era'] = '2000s-present'
    
    return temporal


# Era name mappings
ERA_NAMES = {
    'archaic': 'Archaic (pre-1800s)',
    'early_modern': 'Early Modern (1800s)',
    'modern': 'Modern (1900s-2000s)',
    'contemporary': 'Contemporary (2000s-present)',
    'neologism': 'Neologism (2010s-present)',
}


def format_temporal_analysis(temporal: Dict[str, any]) -> Dict[str, any]:
    """
    Format temporal analysis for output.
    
    Returns:
        Formatted dictionary with human-readable fields
    """
    return {
        'primary_era': temporal['primary_era'],
        'era_name': ERA_NAMES.get(temporal['primary_era'], temporal['primary_era']),
        'era_confidence': temporal['era_confidence'],
        'usage_span': temporal['usage_span'],
        'approximate_era': temporal['approximate_era'],
        'temporal_category': temporal['temporal_category'],
        'is_archaic': temporal['is_archaic'],
        'is_neologism': temporal['is_neologism'],
        'is_contemporary': temporal['is_contemporary'],
        'all_eras': {
            ERA_NAMES.get(era, era): float(conf)
            for era, conf in temporal.get('all_eras', {}).items()
        },
    }

