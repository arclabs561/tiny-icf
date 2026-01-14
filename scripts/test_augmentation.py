# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
# ]
# ///
"""Test different augmentation strategies to see their effects."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from tiny_icf.data import WordICFDataset
from tiny_icf.augmentation import AdvancedAugmentation

def test_augmentation(word: str, icf: float, augment_prob: float, num_samples: int = 10):
    """Test augmentation on a word."""
    dataset = WordICFDataset(
        [(word, icf)],
        max_length=20,
        augment_prob=augment_prob,
        augmentation_fn=AdvancedAugmentation(),
    )
    
    print(f"\nWord: '{word}' (ICF: {icf:.4f})")
    print(f"Augmentation probability: {augment_prob}")
    # Get original bytes from dataset
    original_item = dataset[0]
    original_bytes = original_item['bytes'] if isinstance(original_item, dict) else original_item[0]
    print(f"Original bytes: {original_bytes.tolist()[:10]}...")
    
    augmented_samples = []
    for i in range(num_samples):
        item = dataset[0]
        bytes_tensor = item['bytes'] if isinstance(item, dict) else item[0]
        # Convert back to string for display
        byte_list = bytes_tensor.tolist()
        # Remove padding (0s)
        byte_list = [b for b in byte_list if b != 0]
        try:
            augmented_word = ''.join(chr(b) for b in byte_list if 32 <= b < 127)
            if augmented_word != word:
                augmented_samples.append(augmented_word)
        except:
            pass
    
    if augmented_samples:
        print(f"Augmented samples (showing up to 5):")
        for aug in augmented_samples[:5]:
            print(f"  '{aug}'")
        print(f"Total unique augmentations: {len(set(augmented_samples))}")
    else:
        print("No augmentations generated (may be due to low probability)")

def main():
    test_words = [
        ("hello", 0.1),
        ("the", 0.05),
        ("xylophone", 0.7),
        ("qzxbjk", 0.95),
    ]
    
    print("=" * 80)
    print("AUGMENTATION TESTING")
    print("=" * 80)
    
    for word, icf in test_words:
        for prob in [0.0, 0.1, 0.2, 0.5]:
            test_augmentation(word, icf, prob, num_samples=20)
    
    print("\n" + "=" * 80)
    print("AUGMENTATION ANALYSIS")
    print("=" * 80)
    print("\nObservations:")
    print("- Low ICF words (common): Should see minimal changes")
    print("- High ICF words (rare): May see more variation")
    print("- Augmentation probability controls how often augmentation is applied")

if __name__ == "__main__":
    main()

