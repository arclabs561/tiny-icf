"""Augmentation for symbols, emojis, and special characters.

Handles:
- Random symbols (punctuation, special chars)
- Emojis/emoticons
- Multilingual typo patterns
"""

import random
import re
from typing import List

# Common symbols that appear in web text
COMMON_SYMBOLS = [
    '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
    '-', '_', '=', '+', '[', ']', '{', '}', '\\', '|',
    ';', ':', "'", '"', ',', '.', '<', '>', '/', '?',
    '~', '`',
]

# Common emojis/emoticons (most frequent on web)
COMMON_EMOJIS = [
    '😀', '😃', '😄', '😁', '😆', '😅', '😂', '🤣', '😊', '😇',
    '🙂', '🙃', '😉', '😌', '😍', '🥰', '😘', '😗', '😙', '😚',
    '😋', '😛', '😝', '😜', '🤪', '🤨', '🧐', '🤓', '😎', '🤩',
    '🥳', '😏', '😒', '😞', '😔', '😟', '😕', '🙁', '☹️', '😣',
    '😖', '😫', '😩', '🥺', '😢', '😭', '😤', '😠', '😡', '🤬',
    '❤️', '🧡', '💛', '💚', '💙', '💜', '🖤', '🤍', '🤎', '💔',
    '👍', '👎', '👌', '✌️', '🤞', '🤟', '🤘', '👏', '🙌', '👐',
    '🙏', '✍️', '💪', '🦵', '🦶', '👂', '👃', '👀', '👁️', '🧠',
]

# Text-based emoticons (old school)
TEXT_EMOTICONS = [
    ':)', ':(', ';)', ':D', ':P', ':/', ':|', ':(', ':)', ':o',
    ':-)', ':-(', ';-)', ':-D', ':-P', ':-/', ':-|', ':-)', ':-(',
    '=)', '=(', ';)', '=D', '=P', '=/', '=|', '=)', '=(',
    '<3', '</3', 'xD', 'XD', 'x)', 'X)', 'o_O', 'O_O', 'o_o',
]

# Multilingual keyboard layouts (common substitutions)
MULTILINGUAL_PATTERNS = {
    'es': {  # Spanish
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n',
        'ü': 'u', '¿': '?', '¡': '!',
    },
    'fr': {  # French
        'à': 'a', 'â': 'a', 'ä': 'a', 'é': 'e', 'è': 'e', 'ê': 'e',
        'ë': 'e', 'î': 'i', 'ï': 'i', 'ô': 'o', 'ö': 'o', 'ù': 'u',
        'û': 'u', 'ü': 'u', 'ÿ': 'y', 'ç': 'c',
    },
    'de': {  # German
        'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss',
    },
    'ru': {  # Russian (Cyrillic to Latin common mistakes)
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y',
        'х': 'x', 'к': 'k', 'м': 'm', 'т': 't',
    },
}


def augment_with_symbols(word: str, prob: float = 0.1) -> str:
    """
    Augment with symbols - only character replacement, no additions.
    
    Adding symbols at word boundaries breaks word frequency estimation.
    Only replace characters with symbols (typo-like), don't add new tokens.
    """
    if random.random() > prob:
        return word
    
    # Only replace characters with symbols (typo-like), don't add
    # This preserves word boundaries while adding noise
    if len(word) > 0:
        idx = random.randint(0, len(word) - 1)
        symbol = random.choice(COMMON_SYMBOLS)
        return word[:idx] + symbol + word[idx + 1:]
    
    return word


def augment_with_emoji(word: str, prob: float = 0.05) -> str:
    """
    Augment with emoji/emoticon - DISABLED to preserve word boundaries.
    
    Adding emojis creates multi-token strings which breaks the word frequency
    estimation task. Model is trained on single words, not phrases.
    
    If emoji augmentation is needed, it should be handled at a different level
    (phrase/sentence frequency estimation, not word frequency).
    """
    # Disabled: breaks word boundaries
    # Original implementation added ' ' + emoji which creates 2 tokens
    return word


def augment_multilingual(word: str, language: str, prob: float = 0.1) -> str:
    """Apply language-specific typo patterns."""
    if random.random() > prob:
        return word
    
    patterns = MULTILINGUAL_PATTERNS.get(language, {})
    if not patterns:
        return word
    
    # Replace accented chars with non-accented (common typo)
    for accented, unaccented in patterns.items():
        if accented in word and random.random() < 0.3:
            return word.replace(accented, unaccented, 1)
    
    return word


def detect_language(word: str) -> str:
    """Simple language detection based on character patterns."""
    # Check for language-specific characters
    if any(c in word for c in 'áéíóúñü¿¡'):
        return 'es'  # Spanish
    elif any(c in word for c in 'àâäéèêëîïôöùûüÿç'):
        return 'fr'  # French
    elif any(c in word for c in 'äöüß'):
        return 'de'  # German
    elif any(ord(c) >= 0x0400 and ord(c) <= 0x04FF for c in word):
        return 'ru'  # Cyrillic
    else:
        return 'en'  # Default to English


class UniversalAugmentation:
    """
    Universal augmentation handling:
    - Real typos (from corpus)
    - Keyboard-aware typos
    - Symbols and punctuation
    - Emojis/emoticons
    - Multilingual patterns
    """
    
    def __init__(
        self,
        typo_corpus_path=None,
        symbol_prob: float = 0.05,
        emoji_prob: float = 0.02,
        multilingual_prob: float = 0.1,
        keyboard_prob: float = 0.15,
    ):
        from tiny_icf.typo_augmentation import RealTypoAugmentation
        
        self.typo_aug = RealTypoAugmentation(typo_corpus_path) if typo_corpus_path else None
        self.symbol_prob = symbol_prob
        self.emoji_prob = emoji_prob
        self.multilingual_prob = multilingual_prob
        self.keyboard_prob = keyboard_prob
    
    def __call__(self, word: str) -> str:
        """Apply universal augmentation."""
        # Skip if word is mostly symbols/emojis already
        if self._is_symbol_heavy(word):
            return word
        
        # 1. Real typo (if available)
        if self.typo_aug and random.random() < 0.2:
            word = self.typo_aug.augment_word(word)
        
        # 2. Multilingual patterns
        lang = detect_language(word)
        if lang != 'en':
            word = augment_multilingual(word, lang, self.multilingual_prob)
        
        # 3. Symbols (web text common)
        word = augment_with_symbols(word, self.symbol_prob)
        
        # 4. Emojis (web text common)
        word = augment_with_emoji(word, self.emoji_prob)
        
        return word
    
    def _is_symbol_heavy(self, word: str) -> bool:
        """Check if word is mostly symbols/emojis."""
        # Count non-alphanumeric
        non_alpha = sum(1 for c in word if not c.isalnum() and c not in ' ')
        return non_alpha / max(len(word), 1) > 0.5

