"""Real typo corpus integration for augmentation.

Based on MCP research: "real typo augmentation based on empirically derived
error models provides substantial benefits" - use real typo-correction pairs.
"""

import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

from tiny_icf.keyboard_augmentation import KeyboardAwareAugmentation


class RealTypoAugmentation:
    """
    Augmentation using real typo-correction pairs from corpus.
    
    Research shows this is more effective than synthetic augmentation.
    Uses correction's frequency (correct!) rather than mapping.
    """
    
    def __init__(
        self,
        typo_corpus_path: Path,
        use_prob: float = 0.2,
        fallback_augmentation: KeyboardAwareAugmentation | None = None,
    ):
        """
        Args:
            typo_corpus_path: Path to CSV with typo,correction pairs
            use_prob: Probability of using real typo vs fallback
            fallback_augmentation: Fallback if typo corpus unavailable
        """
        self.use_prob = use_prob
        self.fallback = fallback_augmentation or KeyboardAwareAugmentation()
        
        # Load real typo corpus
        self.typo_map: Dict[str, str] = {}
        self.correction_map: Dict[str, List[str]] = {}  # correction -> list of typos
        
        if typo_corpus_path.exists():
            self._load_typo_corpus(typo_corpus_path)
        else:
            print(f"Warning: Typo corpus not found at {typo_corpus_path}")
            print("  Falling back to keyboard-aware augmentation")
    
    def _load_typo_corpus(self, path: Path):
        """Load typo-correction pairs from CSV."""
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                typo = row['typo'].strip().lower()
                correction = row['correction'].strip().lower()
                
                if typo and correction and typo != correction:
                    self.typo_map[typo] = correction
                    self.correction_map.setdefault(correction, []).append(typo)
        
        print(f"Loaded {len(self.typo_map):,} real typo-correction pairs")
        print(f"  {len(self.correction_map):,} unique corrections")
    
    def augment_word(self, word: str) -> str:
        """
        Augment word using real typo corpus.
        
        Strategy:
        1. If word is a known correction, randomly pick one of its typos
        2. If word is a known typo, return it (already a typo)
        3. Otherwise, use fallback augmentation
        """
        word_lower = word.lower()
        
        # Use real typo with probability
        if random.random() < self.use_prob:
            # Check if word is a known correction
            if word_lower in self.correction_map:
                typos = self.correction_map[word_lower]
                if typos:
                    return random.choice(typos)
            
            # Check if word is already a typo (return as-is)
            if word_lower in self.typo_map:
                return word  # Already a typo, use it
        
        # Fallback to keyboard-aware augmentation
        return self.fallback(word)
    
    def get_correction_frequency(self, typo: str, word_icf: Dict[str, float]) -> float | None:
        """
        Get ICF score for correction of a typo.
        
        This is the key insight: typos should have similar frequency
        to their corrections (not mapped to nearest neighbor).
        """
        typo_lower = typo.lower()
        if typo_lower in self.typo_map:
            correction = self.typo_map[typo_lower]
            return word_icf.get(correction)
        return None


class TypoAwareDataset:
    """
    Dataset that uses real typo corpus for augmentation.
    
    When augmenting, uses correction's frequency (correct approach).
    """
    
    def __init__(
        self,
        word_icf_pairs: List[Tuple[str, float]],
        typo_corpus_path: Path | None = None,
        max_length: int = 20,
        augment_prob: float = 0.2,
    ):
        from tiny_icf.data import WordICFDataset
        
        self.base_dataset = WordICFDataset(
            word_icf_pairs,
            max_length=max_length,
            augment_prob=0.0,  # We'll handle augmentation ourselves
        )
        
        # Real typo augmentation
        if typo_corpus_path:
            self.typo_aug = RealTypoAugmentation(
                typo_corpus_path,
                use_prob=augment_prob,
            )
        else:
            self.typo_aug = None
        
        # Store original pairs for frequency lookup
        self.word_icf = dict(word_icf_pairs)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        word, icf = self.base_dataset.pairs[idx]
        
        # Augment with real typos
        if self.typo_aug and random.random() < 0.2:
            augmented_word = self.typo_aug.augment_word(word)
            
            # Use correction's frequency if available
            correction_icf = self.typo_aug.get_correction_frequency(
                augmented_word,
                self.word_icf,
            )
            
            if correction_icf is not None:
                # Use correction's ICF (correct approach!)
                icf = correction_icf
                word = augmented_word
            else:
                # Fallback: use original word's ICF
                word = augmented_word
        
        # Convert to bytes
        byte_tensor = self.base_dataset._word_to_bytes(word)
        import torch
        icf_tensor = torch.tensor(icf, dtype=torch.float32)
        
        return byte_tensor, icf_tensor

