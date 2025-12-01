"""Keyboard-aware augmentation based on QWERTY layout."""

import random
from typing import Dict, List

# QWERTY keyboard layout adjacency
QWERTY_LAYOUT = {
    'q': ['w', 'a'], 'w': ['q', 'e', 'a', 's'], 'e': ['w', 'r', 'd', 's'],
    'r': ['e', 't', 'd', 'f'], 't': ['r', 'y', 'f', 'g'], 'y': ['t', 'u', 'g', 'h'],
    'u': ['y', 'i', 'h', 'j'], 'i': ['u', 'o', 'j', 'k'], 'o': ['i', 'p', 'k', 'l'],
    'p': ['o', 'l'],
    'a': ['q', 'w', 's', 'z'], 's': ['a', 'w', 'e', 'd', 'x', 'z'],
    'd': ['s', 'e', 'r', 'f', 'c', 'x'], 'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'], 'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'j': ['h', 'u', 'i', 'k', 'm', 'n'], 'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'z': ['a', 's', 'x'], 'x': ['z', 's', 'd', 'c'], 'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'], 'b': ['v', 'g', 'h', 'n'], 'n': ['b', 'h', 'j', 'm'],
    'm': ['n', 'j', 'k'],
}

# Typo pattern frequencies (from GitHub Typo Corpus - real data)
# Updated from analysis of 82k typo-correction pairs
TYPO_PATTERN_FREQUENCIES = {
    'char_drop': 0.443,      # Most common: character omission (44.3%)
    'char_insert': 0.391,    # Character insertion (39.1%)
    'substitution': 0.061,   # Single character substitution (6.1%)
    'adjacent_swap': 0.047,  # Adjacent character swap (4.7%)
    'distant_swap': 0.005,   # Non-adjacent swaps (0.5%)
    'vowel_swap': 0.03,      # Vowel confusion (estimated)
    'double_letter': 0.02,  # Double/single letter errors (estimated)
}


def get_adjacent_keys(char: str) -> List[str]:
    """Get keys adjacent to character on QWERTY keyboard."""
    return QWERTY_LAYOUT.get(char.lower(), [])


def keyboard_adjacent_swap(word: str) -> str:
    """Swap character with adjacent QWERTY key (most common typo)."""
    if len(word) < 2:
        return word
    
    # Find a character with adjacent keys
    candidates = []
    for i, char in enumerate(word):
        if char.lower() in QWERTY_LAYOUT:
            candidates.append(i)
    
    if not candidates:
        return word
    
    # Swap with adjacent key
    idx = random.choice(candidates)
    char = word[idx].lower()
    adjacent = get_adjacent_keys(char)
    
    if adjacent:
        new_char = random.choice(adjacent)
        # Preserve case
        if word[idx].isupper():
            new_char = new_char.upper()
        return word[:idx] + new_char + word[idx + 1 :]
    
    return word


def keyboard_char_drop(word: str) -> str:
    """Drop character (common typo, especially at word boundaries)."""
    if len(word) <= 1:
        return word
    
    # More likely to drop at boundaries
    if random.random() < 0.4:
        # Drop from end (40% chance)
        return word[:-1]
    elif random.random() < 0.3:
        # Drop from start (30% chance)
        return word[1:]
    else:
        # Drop from middle (30% chance)
        if len(word) <= 2:
            # Word too short for middle drop, drop from end
            return word[:-1]
        idx = random.randint(1, len(word) - 2)
        return word[:idx] + word[idx + 1 :]


def keyboard_char_insert(word: str) -> str:
    """Insert character (less common, but happens)."""
    if len(word) >= 20:  # Max length
        return word
    
    # Insert adjacent key character
    if len(word) > 0:
        # Choose position (slightly biased toward boundaries)
        if random.random() < 0.3:
            pos = 0  # Start
        elif random.random() < 0.3:
            pos = len(word)  # End
        else:
            # Middle - handle short words
            if len(word) <= 1:
                pos = len(word)  # Just append
            else:
                pos = random.randint(1, len(word) - 1)  # Middle
        
        # Insert character adjacent to nearby key
        if pos > 0:
            nearby_char = word[pos - 1].lower()
            adjacent = get_adjacent_keys(nearby_char)
            if adjacent:
                insert_char = random.choice(adjacent)
                # Match case of nearby character
                if word[pos - 1].isupper():
                    insert_char = insert_char.upper()
                return word[:pos] + insert_char + word[pos:]
    
    return word


class KeyboardAwareAugmentation:
    """Augmentation that matches real keyboard typo distributions."""
    
    def __init__(
        self,
        typo_prob: float = 0.15,
        pattern_frequencies: Dict[str, float] | None = None,
    ):
        self.typo_prob = typo_prob
        self.pattern_frequencies = pattern_frequencies or TYPO_PATTERN_FREQUENCIES
    
    def __call__(self, word: str) -> str:
        """Apply keyboard-aware augmentation."""
        if random.random() > self.typo_prob:
            return word
        
        # Sample typo pattern based on real frequencies
        rand = random.random()
        cumulative = 0.0
        
        for pattern, freq in self.pattern_frequencies.items():
            cumulative += freq
            if rand <= cumulative:
                if pattern == 'adjacent_swap':
                    return keyboard_adjacent_swap(word)
                elif pattern == 'char_drop':
                    return keyboard_char_drop(word)
                elif pattern == 'char_insert':
                    return keyboard_char_insert(word)
                elif pattern == 'vowel_swap':
                    # Use vowel swap from original augmentation
                    from tiny_icf.augmentation import apply_misspelling
                    return apply_misspelling(word, 'vowel_swap', prob=0.8)
                elif pattern == 'double_letter':
                    from tiny_icf.augmentation import apply_misspelling
                    return apply_misspelling(word, 'double_letter', prob=0.8)
                else:
                    # Fallback to simple swap
                    return keyboard_adjacent_swap(word)
        
        return word

