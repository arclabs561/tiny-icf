"""Multi-task dataset supporting ICF, language detection, temporal ICF, and era classification."""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional

from tiny_icf.language_detection import detect_language_simple
from tiny_icf.temporal_detection import detect_era_patterns

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
        word_icf_pairs: List[Tuple[str, float]],
        max_length: int = 20,
        augment_prob: float = 0.0,
        include_language: bool = True,
        include_era: bool = True,
        include_temporal: bool = False,
        temporal_data: Optional[Dict[str, Dict[int, float]]] = None,
        cache_byte_tensors: bool = True,
    ):
        """
        Args:
            word_icf_pairs: List of (word, icf_score) tuples
            max_length: Maximum character length
            augment_prob: Probability of applying augmentation
            include_language: If True, include language detection labels
            include_era: If True, include era classification labels
            include_temporal: If True, include temporal ICF data
            temporal_data: Optional dict mapping word -> {decade: icf_score}
            cache_byte_tensors: If True, cache byte tensor conversions
        """
        self.pairs = word_icf_pairs
        self.max_length = max_length
        self.augment_prob = augment_prob
        self.include_language = include_language
        self.include_era = include_era
        self.include_temporal = include_temporal
        self.temporal_data = temporal_data or {}
        self.cache_byte_tensors = cache_byte_tensors

        # Pre-compute labels for efficiency
        self._precompute_labels()

        # Cache byte tensors if no augmentation
        if cache_byte_tensors and augment_prob == 0.0:
            self._byte_cache = {}
            for word, _ in self.pairs:
                self._byte_cache[word] = self._word_to_bytes(word)
        else:
            self._byte_cache = {}

    def _precompute_labels(self):
        """Pre-compute language and era labels for all words."""
        self.language_labels = {}
        self.era_labels = {}

        for word, _ in self.pairs:
            if self.include_language:
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

    def _word_to_bytes(self, word: str) -> torch.Tensor:
        """Convert word to byte tensor."""
        if word in self._byte_cache:
            return self._byte_cache[word]

        word_bytes = word.encode("utf-8")[: self.max_length]
        padded = list(word_bytes) + [0] * (self.max_length - len(word_bytes))
        return torch.tensor(padded, dtype=torch.long)

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
        word, icf = self.pairs[idx]

        # Apply augmentation
        word = self._augment(word)

        # Convert to bytes
        byte_tensor = self._word_to_bytes(word)
        icf_tensor = torch.tensor(icf, dtype=torch.float32)

        # Build result dictionary
        result = {
            "byte_tensors": byte_tensor,
            "icf_targets": icf_tensor,
        }

        # Add language label if enabled
        if self.include_language:
            lang_idx = self.language_labels.get(word, 0)
            result["language_targets"] = torch.tensor(lang_idx, dtype=torch.long)

        # Add era label if enabled
        if self.include_era:
            era_idx = self.era_labels.get(word, 2)
            result["era_targets"] = torch.tensor(era_idx, dtype=torch.long)

        # Add temporal ICF if enabled and available
        if self.include_temporal and word in self.temporal_data:
            # Convert to tensors (for now, just current ICF as placeholder)
            # In full implementation, would include historical predictions
            result["historical_targets"] = {
                "current": icf_tensor,
                # Can add: '1800': torch.tensor(self.temporal_data[word].get(1800, icf), ...),
                # etc.
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

    result: Dict[str, torch.Tensor] = {
        "byte_tensors": byte_tensors,
        "icf_targets": icf_targets,
    }

    # Stack language targets if present
    if "language_targets" in batch[0]:
        result["language_targets"] = torch.stack([item["language_targets"] for item in batch])

    # Stack era targets if present
    if "era_targets" in batch[0]:
        result["era_targets"] = torch.stack([item["era_targets"] for item in batch])

    # Handle temporal data if present (more complex, would need custom handling)
    if "historical_targets" in batch[0]:
        # For now, just pass through (would need more sophisticated batching)
        # Store as list, not tensor
        result["historical_targets"] = [item.get("historical_targets") for item in batch]  # type: ignore

    return result
