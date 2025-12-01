"""Curriculum learning: progressively harder training examples."""

from typing import List, Tuple

import numpy as np


def compute_word_difficulty(word: str, icf_score: float) -> float:
    """
    Compute difficulty score for a word.
    
    Factors:
    - ICF score (rarer = harder)
    - Length (longer = harder)
    - Character complexity (more unique chars = harder)
    - Morphological complexity (affixes = harder)
    """
    length_factor = len(word) / 20.0  # Normalize by max length
    icf_factor = icf_score  # Already 0-1
    
    # Character diversity (unique chars / total chars)
    unique_chars = len(set(word.lower()))
    diversity_factor = unique_chars / max(len(word), 1)
    
    # Morphological complexity (count affixes)
    affixes = ["un", "re", "pre", "dis", "mis", "ing", "ed", "ly", "tion", "ness", "ment"]
    affix_count = sum(1 for affix in affixes if affix in word.lower())
    morphological_factor = min(affix_count / 3.0, 1.0)
    
    # Combine factors (weighted average)
    difficulty = (
        0.4 * icf_factor +  # Rarity is most important
        0.2 * length_factor +
        0.2 * diversity_factor +
        0.2 * morphological_factor
    )
    
    return min(difficulty, 1.0)


def create_curriculum_schedule(
    word_icf_pairs: List[Tuple[str, float]],
    num_stages: int = 5,
) -> List[List[Tuple[str, float]]]:
    """
    Create curriculum learning schedule with progressively harder words.
    
    Returns:
        List of stages, each containing (word, icf) pairs
        Stage 0 = easiest, Stage N-1 = hardest
    """
    # Compute difficulty for each word
    word_difficulties = [
        (word, icf, compute_word_difficulty(word, icf))
        for word, icf in word_icf_pairs
    ]
    
    # Sort by difficulty
    word_difficulties.sort(key=lambda x: x[2])
    
    # Split into stages
    total = len(word_difficulties)
    stage_size = total // num_stages
    
    stages = []
    for i in range(num_stages):
        start_idx = i * stage_size
        end_idx = (i + 1) * stage_size if i < num_stages - 1 else total
        
        stage_words = [(word, icf) for word, icf, _ in word_difficulties[start_idx:end_idx]]
        stages.append(stage_words)
    
    return stages


def get_stage_schedule(num_epochs: int, num_stages: int) -> List[int]:
    """
    Determine which stage to use at each epoch.
    
    Returns:
        List mapping epoch -> stage number
    """
    epochs_per_stage = num_epochs // num_stages
    
    schedule = []
    for epoch in range(num_epochs):
        # Linear progression: start with stage 0, end with stage N-1
        stage = min(epoch // epochs_per_stage, num_stages - 1)
        schedule.append(stage)
    
    return schedule


class CurriculumSampler:
    """Sample words based on curriculum learning schedule."""
    
    def __init__(
        self,
        stages: List[List[Tuple[str, float]]],
        schedule: List[int],
        warmup_epochs: int = 0,
    ):
        self.stages = stages
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def get_current_stage_words(self) -> List[Tuple[str, float]]:
        """Get words for current training stage."""
        if self.current_epoch < self.warmup_epochs:
            # Warmup: use easiest stage
            return self.stages[0]
        
        stage_idx = self.schedule[min(self.current_epoch, len(self.schedule) - 1)]
        return self.stages[stage_idx]
    
    def advance_epoch(self):
        """Move to next epoch."""
        self.current_epoch += 1
    
    def get_progress(self) -> Tuple[int, int, float]:
        """Get current progress: (current_stage, total_stages, progress_pct)."""
        if self.current_epoch < self.warmup_epochs:
            return (0, len(self.stages), 0.0)
        
        stage_idx = self.schedule[min(self.current_epoch, len(self.schedule) - 1)]
        progress = (stage_idx + 1) / len(self.stages)
        return (stage_idx, len(self.stages), progress)

