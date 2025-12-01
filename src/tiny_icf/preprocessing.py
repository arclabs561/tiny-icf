"""Preprocessing utilities for cleaning text before frequency estimation.

Handles:
- HTML entity decoding
- URL extraction
- Code detection
- Encoding error detection
- Non-linguistic filtering
"""

import re
import html
from typing import List, Set, Tuple
from urllib.parse import urlparse


# Common HTML entities
HTML_ENTITIES = {
    '&nbsp;', '&amp;', '&lt;', '&gt;', '&quot;', '&apos;',
    '&copy;', '&reg;', '&trade;', '&mdash;', '&ndash;',
    '&hellip;', '&bull;', '&rsquo;', '&lsquo;', '&rdquo;', '&ldquo;',
}

# Code-like patterns
CODE_PATTERNS = [
    r'function\s*\([^)]*\)\s*\{',  # JavaScript function
    r'def\s+\w+\s*\([^)]*\):',     # Python function
    r'class\s+\w+',                 # Class definition
    r'import\s+\w+',                # Import statement
    r'#include\s*<[^>]+>',          # C include
    r'<\?php',                      # PHP tag
    r'<script',                     # HTML script tag
    r'<style',                      # HTML style tag
]

# Encoding error patterns (mojibake)
ENCODING_ERROR_PATTERNS = [
    r'â€™',      # Corrupted apostrophe
    r'â€œ',      # Corrupted quote
    r'â€"',      # Corrupted dash
    r'Ã©',       # Corrupted é
    r'Ã¡',       # Corrupted á
    r'\x00',     # Null bytes
    r'[\x80-\x9F]',  # Invalid UTF-8 continuation bytes
]


def is_html_entity(text: str) -> bool:
    """Check if text is an HTML entity."""
    return text.strip() in HTML_ENTITIES or text.strip().startswith('&') and text.strip().endswith(';')


def is_url(text: str) -> bool:
    """Check if text looks like a URL."""
    # Simple heuristic: contains protocol or common URL patterns
    url_patterns = [
        r'^https?://',
        r'^www\.',
        r'^[a-z0-9-]+\.(com|org|net|edu|gov|io|co|uk|de|fr|es|it|ru|jp|cn)',
        r'^[a-z0-9-]+\.[a-z]{2,}/',  # Domain with path
    ]
    return any(re.match(pattern, text.lower()) for pattern in url_patterns)


def is_email(text: str) -> bool:
    """Check if text looks like an email address."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, text))


def is_code_like(text: str) -> bool:
    """Check if text looks like code."""
    text_lower = text.lower()
    # Check for code patterns
    if any(re.search(pattern, text_lower) for pattern in CODE_PATTERNS):
        return True
    # Check for high ratio of special characters (code-like)
    special_chars = sum(1 for c in text if c in '{}[]();=<>+-*/%&|!~^')
    if len(text) > 0 and special_chars / len(text) > 0.3:
        return True
    # Check for common code keywords
    code_keywords = ['function', 'def', 'class', 'import', 'return', 'var', 'let', 'const']
    if any(text_lower.startswith(kw) or text_lower.endswith(kw) for kw in code_keywords):
        return True
    return False


def has_encoding_errors(text: str) -> bool:
    """Check if text contains encoding errors (mojibake)."""
    # Check for common mojibake patterns
    if any(re.search(pattern, text) for pattern in ENCODING_ERROR_PATTERNS):
        return True
    # Check for invalid UTF-8 sequences
    try:
        text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return True
    return False


def is_pure_number(text: str) -> bool:
    """Check if text is purely numeric."""
    # Remove common number separators
    cleaned = text.replace(',', '').replace('.', '').replace('-', '').replace(' ', '')
    return cleaned.isdigit()


def is_gibberish(text: str, min_ratio: float = 0.5) -> bool:
    """Check if text looks like gibberish (high ratio of non-alphabetic)."""
    if len(text) == 0:
        return True
    # Count alphabetic characters
    alpha_count = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_count / len(text)
    return alpha_ratio < min_ratio


def extract_words_from_url(url: str) -> List[str]:
    """Extract potential words from URL path/domain."""
    try:
        parsed = urlparse(url)
        words = []
        # Domain parts
        domain_parts = parsed.netloc.split('.')
        words.extend([p for p in domain_parts if len(p) > 2 and p.isalpha()])
        # Path parts
        path_parts = parsed.path.split('/')
        words.extend([p for p in path_parts if len(p) > 2 and p.isalpha()])
        return words
    except:
        return []


def is_valid_word(text: str, min_length: int = 2, max_length: int = 50) -> bool:
    """
    Check if text is a valid word for frequency estimation.
    
    Criteria:
    - Length within bounds
    - Contains alphabetic characters
    - Not HTML entity
    - Not URL
    - Not email
    - Not code-like
    - No encoding errors
    - Not pure number
    """
    # Length check
    if len(text) < min_length or len(text) > max_length:
        return False
    
    # Must contain at least one alphabetic character
    if not any(c.isalpha() for c in text):
        return False
    
    # Filter out non-linguistic content
    if is_html_entity(text):
        return False
    if is_url(text):
        return False
    if is_email(text):
        return False
    if is_code_like(text):
        return False
    if has_encoding_errors(text):
        return False
    if is_pure_number(text):
        return False
    
    return True


def clean_text_for_frequency(text: str) -> List[str]:
    """
    Clean text and extract valid words for frequency estimation.
    
    Steps:
    1. Decode HTML entities
    2. Split into tokens (whitespace + punctuation)
    3. Filter valid words
    4. Return list of clean words
    """
    # Decode HTML entities
    text = html.unescape(text)
    
    # Simple tokenization: split on whitespace and punctuation
    # Keep words with alphanumeric + accents
    tokens = re.findall(r'\b[\w\'-]+\b', text)
    
    # Filter valid words
    valid_words = [t.lower() for t in tokens if is_valid_word(t)]
    
    return valid_words


def filter_frequency_list(
    word_counts: dict,
    min_length: int = 2,
    max_length: int = 50,
    min_alpha_ratio: float = 0.5,
) -> dict:
    """
    Filter frequency list to keep only valid words.
    
    Removes:
    - HTML entities
    - URLs
    - Emails
    - Code-like strings
    - Encoding errors
    - Pure numbers
    - Gibberish
    """
    filtered = {}
    removed = {
        'html_entities': 0,
        'urls': 0,
        'emails': 0,
        'code': 0,
        'encoding_errors': 0,
        'numbers': 0,
        'gibberish': 0,
        'length': 0,
    }
    
    for word, count in word_counts.items():
        # Length check
        if len(word) < min_length or len(word) > max_length:
            removed['length'] += 1
            continue
        
        # Must contain alphabetic characters
        if not any(c.isalpha() for c in word):
            removed['gibberish'] += 1
            continue
        
        # Check alpha ratio
        alpha_count = sum(1 for c in word if c.isalpha())
        if alpha_count / len(word) < min_alpha_ratio:
            removed['gibberish'] += 1
            continue
        
        # Filter non-linguistic
        if is_html_entity(word):
            removed['html_entities'] += 1
            continue
        if is_url(word):
            removed['urls'] += 1
            continue
        if is_email(word):
            removed['emails'] += 1
            continue
        if is_code_like(word):
            removed['code'] += 1
            continue
        if has_encoding_errors(word):
            removed['encoding_errors'] += 1
            continue
        if is_pure_number(word):
            removed['numbers'] += 1
            continue
        
        # Valid word
        filtered[word] = count
    
    print(f"Filtered frequency list:")
    print(f"  Kept: {len(filtered):,} words")
    print(f"  Removed: {sum(removed.values()):,} words")
    for reason, count in removed.items():
        if count > 0:
            print(f"    {reason}: {count:,}")
    
    return filtered

