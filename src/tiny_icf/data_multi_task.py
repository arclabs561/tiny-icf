"""Multi-task dataset supporting ICF + auxiliary labels.

This is designed to be used with `model_multi_task.MultiTaskICF` and the unified
loss / Lightning module paths, but it is also usable standalone.

Key design points:
- We can mix "ICF-labeled" samples (clean words with meaningful counts) and
  "hygiene-only" samples (URLs, code, numbers, etc.) in the same dataset.
- We expose `icf_mask` to indicate which rows should contribute to the ICF loss.
- Temporal labels are optional and exposed as `historical_targets` + `historical_mask`.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional

from tiny_icf.language_detection import detect_language_simple
from tiny_icf.temporal_detection import detect_era_patterns
from tiny_icf.preprocessing import (
    has_encoding_errors,
    is_code_like,
    is_email,
    is_gibberish,
    is_html_entity,
    is_pure_number,
    is_url,
)
from tiny_icf.predict import word_to_bytes

# Optional augmentation
try:
    from tiny_icf.augmentation import AdvancedAugmentation

    HAS_AUGMENTATION = True
except ImportError:
    HAS_AUGMENTATION = False


# Language code to index mapping
LANGUAGE_CODES = ["en", "es", "fr", "de", "it", "pt", "ru", "ko", "zh", "ja"]
LANGUAGE_TO_INDEX = {lang: idx for idx, lang in enumerate(LANGUAGE_CODES)}
NUM_LANGUAGES = len(LANGUAGE_CODES)

# Era to index mapping
ERA_CODES = ["archaic", "early_modern", "modern", "contemporary", "neologism"]
ERA_TO_INDEX = {era: idx for idx, era in enumerate(ERA_CODES)}
NUM_ERAS = len(ERA_CODES)

# Token hygiene classes (useful for downstream filtering and “junk token” detection)
HYGIENE_CODES = [
    "clean_word",
    "url",
    "email",
    "code",
    "html_entity",
    "encoding_error",
    "number",
    "gibberish",
]
HYGIENE_TO_INDEX = {k: i for i, k in enumerate(HYGIENE_CODES)}
NUM_HYGIENE = len(HYGIENE_CODES)


def _label_hygiene(token: str) -> int:
    """
    Heuristic token hygiene label.

    Priority order matters: e.g. URLs often contain ':' and '.' and should be labeled as URL.
    """
    t = token.strip()
    if not t:
        return HYGIENE_TO_INDEX["gibberish"]
    if is_html_entity(t):
        return HYGIENE_TO_INDEX["html_entity"]
    if is_url(t):
        return HYGIENE_TO_INDEX["url"]
    if is_email(t):
        return HYGIENE_TO_INDEX["email"]
    if is_code_like(t):
        return HYGIENE_TO_INDEX["code"]
    if has_encoding_errors(t):
        return HYGIENE_TO_INDEX["encoding_error"]
    if is_pure_number(t):
        return HYGIENE_TO_INDEX["number"]
    if is_gibberish(t):
        return HYGIENE_TO_INDEX["gibberish"]
    return HYGIENE_TO_INDEX["clean_word"]


class MultiTaskICFDataset(Dataset):
    """
    Multi-task dataset that provides:
    - ICF targets (primary task)
    - Language labels (classification)
    - Era labels (classification)
    - Temporal ICF (optional, if historical data available)
    """

    def __init__(
        self,
        word_icf_pairs: List[Tuple[str, float] | Tuple[str, float, bool]],
        max_length: int = 20,
        augment_prob: float = 0.0,
        include_language: bool = True,
        include_era: bool = True,
        include_hygiene: bool = False,
        include_temporal: bool = False,
        temporal_data: Optional[Dict[str, Dict[int, float]]] = None,
        temporal_decades: Optional[List[int]] = None,
        strip_language_prefix_in_bytes: bool = False,
        cache_byte_tensors: bool = True,
    ):
        """
        Args:
            word_icf_pairs: List of (word, icf_score) tuples
            max_length: Maximum character length
            augment_prob: Probability of applying augmentation
            include_language: If True, include language detection labels
            include_era: If True, include era classification labels
            include_hygiene: If True, include token hygiene labels
            include_temporal: If True, include temporal ICF data
            temporal_data: Optional dict mapping word -> {decade: icf_score}
            temporal_decades: Which decades to include (default: [1800, 1900, 2000])
            strip_language_prefix_in_bytes: If True, encode only the base token (drop `lang:` prefix).
            cache_byte_tensors: If True, cache byte tensor conversions
        """
        # Normalize input to (word, icf, icf_mask)
        pairs_norm: list[tuple[str, float, bool]] = []
        for item in word_icf_pairs:
            if len(item) == 2:  # type: ignore[arg-type]
                w, icf = item  # type: ignore[misc]
                pairs_norm.append((str(w), float(icf), True))
            else:
                w, icf, m = item  # type: ignore[misc]
                pairs_norm.append((str(w), float(icf), bool(m)))

        self.pairs = pairs_norm
        self.max_length = max_length
        self.augment_prob = augment_prob
        self.include_language = include_language
        self.include_era = include_era
        self.include_hygiene = include_hygiene
        self.include_temporal = include_temporal
        self.temporal_data = temporal_data or {}
        self.temporal_decades = temporal_decades or [1800, 1900, 2000]
        self.strip_language_prefix_in_bytes = strip_language_prefix_in_bytes
        self.cache_byte_tensors = cache_byte_tensors

        # Pre-compute labels for efficiency
        self._precompute_labels()

        # Cache byte tensors if no augmentation
        if cache_byte_tensors and augment_prob == 0.0:
            self._byte_cache = {}
            for word, _icf, _mask in self.pairs:
                self._byte_cache[word] = self._word_to_bytes(word)
        else:
            self._byte_cache = {}

    def _precompute_labels(self):
        """Pre-compute language and era labels for all words."""
        self.language_labels = {}
        self.era_labels = {}
        self.hygiene_labels = {}
        self.temporal_masks = {}

        for word, _icf, icf_mask in self.pairs:
            if self.include_language:
                # If token looks like `lang:word` and lang is known, use it as label.
                if ":" in word:
                    maybe_lang, _rest = word.split(":", 1)
                    if maybe_lang in LANGUAGE_TO_INDEX:
                        self.language_labels[word] = LANGUAGE_TO_INDEX[maybe_lang]
                        continue
                # Detect language and convert to index
                lang_results = detect_language_simple(word)
                if lang_results:
                    top_lang = lang_results[0][0]  # (lang_code, confidence)
                    self.language_labels[word] = LANGUAGE_TO_INDEX.get(
                        top_lang, 0
                    )  # Default to English
                else:
                    self.language_labels[word] = 0  # Default to English

            if self.include_era:
                # Detect era and convert to index
                era_results = detect_era_patterns(word)
                if era_results:
                    # Get era with highest confidence
                    top_era = max(era_results.items(), key=lambda x: x[1])[0]
                    self.era_labels[word] = ERA_TO_INDEX.get(top_era, 2)  # Default to 'modern'
                else:
                    self.era_labels[word] = 2  # Default to 'modern'

            if self.include_hygiene:
                self.hygiene_labels[word] = _label_hygiene(word)

            if self.include_temporal:
                # Mark whether we have any historical targets for this token.
                # For multilingual keys, try also the base token.
                key = word
                base = word.split(":", 1)[1] if ":" in word else word
                has_hist = key in self.temporal_data or base in self.temporal_data
                self.temporal_masks[word] = bool(has_hist and icf_mask)

    def _word_to_bytes(self, word: str) -> torch.Tensor:
        """Convert word to byte tensor."""
        if word in self._byte_cache:
            return self._byte_cache[word]

        # Use the shared, character-boundary-safe helper.
        return word_to_bytes(word, max_length=self.max_length).squeeze(0)

    def _augment(self, word: str) -> str:
        """Apply augmentation."""
        if self.augment_prob == 0.0:
            return word

        if HAS_AUGMENTATION:
            import random

            if random.random() < self.augment_prob:
                aug = AdvancedAugmentation()
                return aug(word)

        return word

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        """Return multi-task batch item."""
        word, icf, icf_mask = self.pairs[idx]

        # Apply augmentation
        word = self._augment(word)

        # Optionally strip a leading `lang:` prefix for byte encoding.
        word_for_bytes = word
        if self.strip_language_prefix_in_bytes and ":" in word:
            maybe_lang, rest = word.split(":", 1)
            if maybe_lang in LANGUAGE_TO_INDEX:
                word_for_bytes = rest

        # Convert to bytes
        byte_tensor = self._word_to_bytes(word_for_bytes)
        icf_tensor = torch.tensor(icf, dtype=torch.float32)

        # Build result dictionary
        result = {
            "byte_tensors": byte_tensor,
            "icf_targets": icf_tensor,
            "icf_mask": torch.tensor(bool(icf_mask), dtype=torch.bool),
            "words": word,
        }

        # Add language label if enabled
        if self.include_language:
            lang_idx = self.language_labels.get(word, 0)
            result["language_targets"] = torch.tensor(lang_idx, dtype=torch.long)

        # Add era label if enabled
        if self.include_era:
            era_idx = self.era_labels.get(word, 2)
            result["era_targets"] = torch.tensor(era_idx, dtype=torch.long)

        # Add hygiene label if enabled
        if self.include_hygiene:
            hyg_idx = self.hygiene_labels.get(word, HYGIENE_TO_INDEX["clean_word"])
            result["hygiene_targets"] = torch.tensor(hyg_idx, dtype=torch.long)

        # Add temporal ICF if enabled and available
        if self.include_temporal:
            base = word.split(":", 1)[1] if ":" in word else word
            hist = self.temporal_data.get(word) or self.temporal_data.get(base)
            has_hist = bool(hist) and bool(icf_mask)
            result["historical_mask"] = torch.tensor(has_hist, dtype=torch.bool)
            if hist:
                # Always populate all decades (fill missing with current icf).
                result["historical_targets"] = {
                    int(decade): torch.tensor(
                        float(hist.get(int(decade), icf)), dtype=torch.float32
                    )
                    for decade in self.temporal_decades
                }
            else:
                result["historical_targets"] = {
                    int(decade): icf_tensor for decade in self.temporal_decades
                }

        return result


def collate_multi_task_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for multi-task batches.

    Converts list of dicts into batched tensors.
    """
    # Stack byte tensors and ICF targets
    byte_tensors = torch.stack([item["byte_tensors"] for item in batch])
    icf_targets = torch.stack([item["icf_targets"] for item in batch]).unsqueeze(1)
    icf_mask = torch.stack([item.get("icf_mask", torch.tensor(True)) for item in batch]).to(
        torch.bool
    )

    result: Dict[str, torch.Tensor] = {
        "byte_tensors": byte_tensors,
        "icf_targets": icf_targets,
        "icf_mask": icf_mask,
    }

    # Pass through words for debugging / distillation interfaces.
    if "words" in batch[0]:
        result["words"] = [item.get("words", "") for item in batch]  # type: ignore[assignment]

    # Stack language targets if present
    if "language_targets" in batch[0]:
        result["language_targets"] = torch.stack([item["language_targets"] for item in batch])

    # Stack era targets if present
    if "era_targets" in batch[0]:
        result["era_targets"] = torch.stack([item["era_targets"] for item in batch])

    # Stack hygiene targets if present
    if "hygiene_targets" in batch[0]:
        result["hygiene_targets"] = torch.stack([item["hygiene_targets"] for item in batch])

    # Temporal targets: collate dict-of-scalars into dict-of-tensors + mask.
    if "historical_targets" in batch[0]:
        # Identify decades from the first element (they should be consistent).
        decades = list(batch[0]["historical_targets"].keys())  # type: ignore[index]
        hist_targets: dict[int, torch.Tensor] = {}
        for d in decades:
            hist_targets[int(d)] = torch.stack(
                [item["historical_targets"][d] for item in batch]  # type: ignore[index]
            ).unsqueeze(1)
        result["historical_targets"] = hist_targets  # type: ignore[assignment]
        if "historical_mask" in batch[0]:
            result["historical_mask"] = torch.stack(
                [item.get("historical_mask", torch.tensor(False)) for item in batch]
            ).to(torch.bool)

    return result
