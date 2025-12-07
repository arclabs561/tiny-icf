"""Advanced data augmentation for word frequency estimation."""

import random
import string
from typing import List, Optional

# Common misspelling patterns
MISSPELLING_PATTERNS = {
    "double_letter": [
        ("l", "ll"), ("e", "ee"), ("o", "oo"), ("t", "tt"), ("s", "ss"),
        ("n", "nn"), ("r", "rr"), ("m", "mm"), ("p", "pp"), ("c", "cc"),
    ],
    "single_letter": [
        ("ll", "l"), ("ee", "e"), ("oo", "o"), ("tt", "t"), ("ss", "s"),
    ],
    "vowel_swap": [
        ("a", "e"), ("e", "i"), ("i", "e"), ("o", "u"), ("u", "o"),
        ("a", "o"), ("e", "a"), ("i", "a"), ("ou", "o"), ("ie", "ei"),
    ],
    "common_typos": [
        ("ie", "ei"), ("ei", "ie"), ("tion", "sion"), ("sion", "tion"),
        ("ph", "f"), ("f", "ph"), ("ck", "k"), ("k", "ck"),
        ("qu", "kw"), ("x", "ks"), ("z", "s"), ("s", "z"),
    ],
    "adjacent_swap": True,  # Swap adjacent characters
    "char_drop": True,  # Random character deletion
    "char_insert": True,  # Random character insertion
}


def apply_misspelling(word: str, pattern: str, prob: float = 0.5) -> str:
    """Apply a specific misspelling pattern."""
    if random.random() > prob:
        return word
    
    if pattern == "double_letter" and MISSPELLING_PATTERNS["double_letter"]:
        old, new = random.choice(MISSPELLING_PATTERNS["double_letter"])
        if old in word:
            return word.replace(old, new, 1)
    
    elif pattern == "single_letter" and MISSPELLING_PATTERNS["single_letter"]:
        old, new = random.choice(MISSPELLING_PATTERNS["single_letter"])
        if old in word:
            return word.replace(old, new, 1)
    
    elif pattern == "vowel_swap" and MISSPELLING_PATTERNS["vowel_swap"]:
        old, new = random.choice(MISSPELLING_PATTERNS["vowel_swap"])
        if old in word:
            return word.replace(old, new, 1)
    
    elif pattern == "common_typos" and MISSPELLING_PATTERNS["common_typos"]:
        old, new = random.choice(MISSPELLING_PATTERNS["common_typos"])
        if old in word:
            return word.replace(old, new, 1)
    
    elif pattern == "adjacent_swap" and len(word) >= 2:
        idx = random.randint(0, len(word) - 2)
        chars = list(word)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    
    elif pattern == "char_drop" and len(word) > 1:
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx + 1 :]
    
    elif pattern == "char_insert" and len(word) < 20:
        idx = random.randint(0, len(word))
        char = random.choice(string.ascii_lowercase)
        return word[:idx] + char + word[idx:]
    
    return word


def augment_word(word: str, num_augmentations: int = 1, patterns: Optional[List[str]] = None) -> List[str]:
    """
    Generate augmented versions of a word.
    
    Args:
        word: Original word
        num_augmentations: Number of augmented versions to generate
        patterns: List of patterns to use (None = use all)
    
    Returns:
        List of augmented words (including original if num_augmentations > 0)
    """
    if patterns is None:
        patterns = [
            "double_letter",
            "single_letter",
            "vowel_swap",
            "common_typos",
            "adjacent_swap",
            "char_drop",
            "char_insert",
        ]
    
    augmented = []
    for _ in range(num_augmentations):
        pattern = random.choice(patterns)
        aug_word = apply_misspelling(word, pattern, prob=0.7)
        if aug_word != word:  # Only add if actually changed
            augmented.append(aug_word)
    
    return augmented if augmented else [word]


def morphological_augment(word: str) -> str:
    """Apply morphological variations (plurals, verb forms, etc.)."""
    # Simple morphological rules
    if word.endswith("s") and len(word) > 3:
        # Try removing plural
        if random.random() < 0.3:
            return word[:-1]
    
    if word.endswith("ed") and len(word) > 4:
        # Try removing past tense
        if random.random() < 0.3:
            return word[:-2]
    
    if word.endswith("ing") and len(word) > 5:
        # Try removing gerund
        if random.random() < 0.3:
            base = word[:-3]
            # Add 'e' if needed (e.g., "running" -> "run")
            if base + "e" in [word[:-3] + "e"]:
                return base + "e"
            return base
    
    if word.endswith("ly") and len(word) > 4:
        # Try removing adverb suffix
        if random.random() < 0.3:
            return word[:-2]
    
    return word


class AdvancedAugmentation:
    """Advanced augmentation with multiple strategies."""
    
    def __init__(
        self,
        misspelling_prob: float = 0.15,
        morphological_prob: float = 0.1,
        noise_prob: float = 0.05,
    ):
        self.misspelling_prob = misspelling_prob
        self.morphological_prob = morphological_prob
        self.noise_prob = noise_prob
    
    def __call__(self, word: str) -> str:
        """Apply augmentation to a word."""
        # Misspellings
        if random.random() < self.misspelling_prob:
            patterns = [
                "double_letter",
                "single_letter",
                "vowel_swap",
                "common_typos",
                "adjacent_swap",
                "char_drop",
            ]
            word = apply_misspelling(word, random.choice(patterns), prob=0.8)
        
        # Morphological variations
        if random.random() < self.morphological_prob:
            word = morphological_augment(word)
        
        # Character-level noise
        if random.random() < self.noise_prob and len(word) > 1:
            # Small chance of character swap
            if random.random() < 0.5:
                word = apply_misspelling(word, "adjacent_swap", prob=1.0)
            else:
                word = apply_misspelling(word, "char_drop", prob=1.0)
        
        return word

