"""Synthetic OOV word generation utilities.

These helpers are used for:
- evaluation: composed-but-plausible vs gibberish comparisons
- (optional) training augmentation: adding OOV-shaped samples

We intentionally keep generation simple and deterministic (seeded).
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass


def is_simple_english_token(w: str) -> bool:
    """Conservative filter: lowercase ascii alphabetic tokens only."""

    return w.isascii() and w.islower() and w.isalpha()


def choose_bases(common_words: list[str], *, seed: int, max_bases: int) -> list[str]:
    """Pick candidate stems from common words."""

    rng = random.Random(seed)
    bases = [w for w in common_words if is_simple_english_token(w) and 3 <= len(w) <= 8]
    rng.shuffle(bases)
    return bases[:max_bases]


@dataclass(frozen=True)
class CompositionConfig:
    max_len: int = 20
    max_attempts_factor: int = 200


DEFAULT_COMPOSITION_CONFIG = CompositionConfig()


def generate_composed_words(
    *,
    bases: list[str],
    real_set: set[str],
    n: int,
    seed: int,
    config: CompositionConfig = DEFAULT_COMPOSITION_CONFIG,
) -> list[str]:
    """Generate prefix/suffix compositions that are *not* in the real_set."""

    rng = random.Random(seed)

    # Bias towards no-prefix so we get a mix.
    prefixes = [
        "",
        "",
        "un",
        "re",
        "pre",
        "post",
        "anti",
        "non",
        "over",
        "under",
        "sub",
        "super",
        "inter",
        "trans",
        "micro",
        "macro",
    ]
    suffixes = [
        "s",
        "es",
        "ed",
        "ing",
        "er",
        "est",
        "ness",
        "less",
        "ly",
        "ment",
        "tion",
        "able",
        "ible",
        "ity",
        "ism",
        "ist",
        "ship",
        "hood",
        "ful",
        "ish",
    ]

    out: set[str] = set()
    attempts = 0
    max_attempts = max(1, n) * int(config.max_attempts_factor)
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        base = rng.choice(bases)
        pref = rng.choice(prefixes)
        s1 = rng.choice(suffixes)
        s2 = rng.choice(suffixes) if rng.random() < 0.25 else ""
        w = f"{pref}{base}{s1}{s2}"
        if not is_simple_english_token(w):
            continue
        if len(w) > config.max_len:
            continue
        if w in real_set:
            continue
        out.add(w)

    return sorted(out)


def generate_gibberish_words(
    *,
    lengths: list[int],
    real_set: set[str],
    n: int,
    seed: int,
) -> list[str]:
    """Generate random lowercase gibberish strings not in real_set."""

    rng = random.Random(seed)
    letters = string.ascii_lowercase
    out: set[str] = set()
    attempts = 0
    max_attempts = max(1, n) * 200
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        L = rng.choice(lengths)
        w = "".join(rng.choice(letters) for _ in range(L))
        if w in real_set:
            continue
        out.add(w)
    return sorted(out)
