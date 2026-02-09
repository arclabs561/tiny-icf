"""Custom collate functions for distillation training."""

import torch
from typing import List, Tuple, Dict, Any


def collate_with_words(batch: List[Tuple[torch.Tensor, torch.Tensor, str]]) -> Dict[str, Any]:
    """
    Collate function for batches that include word strings (for distillation).

    Args:
        batch: List of (byte_tensor, icf_tensor, word_string) tuples

    Returns:
        Dictionary with:
        - 'byte_tensors': [batch_size, max_length] stacked byte tensors
        - 'icf_targets': [batch_size, 1] stacked ICF targets
        - 'words': List of word strings
    """
    byte_tensors = []
    icf_targets = []
    words = []

    for item in batch:
        if len(item) == 3:
            byte_tensor, icf_tensor, word = item
            words.append(word)
        else:
            # Fallback: no word string
            byte_tensor, icf_tensor = item
            words.append("")  # Empty string as placeholder

        byte_tensors.append(byte_tensor)
        icf_targets.append(icf_tensor)

    # Stack tensors
    byte_tensors = torch.stack(byte_tensors)
    icf_targets = torch.stack(icf_targets)

    # Ensure icf_targets is [batch_size, 1]
    if icf_targets.dim() == 1:
        icf_targets = icf_targets.unsqueeze(1)

    return {
        "byte_tensors": byte_tensors,
        "icf_targets": icf_targets,
        "words": words,
    }
