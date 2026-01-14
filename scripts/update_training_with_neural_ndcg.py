# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Update training scripts to use NeuralNDCG loss.

This script updates CombinedLoss initialization in training scripts
to enable NeuralNDCG by default.
"""

import re
from pathlib import Path

TRAINING_SCRIPTS = [
    "scripts/train_residual.py",
    "scripts/train_aggressive_regularization.py",
    "scripts/train_reduced_capacity.py",
    "scripts/train_batchnorm.py",
    "scripts/train_gated_residual.py",
    "scripts/train_nano.py",
    "scripts/train_ephemeral_robust.py",
]

def update_script(script_path: Path):
    """Update a training script to use NeuralNDCG."""
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check if already has use_neural_ndcg
    if 'use_neural_ndcg' in content:
        print(f"  ⏭️  {script_path.name} already has NeuralNDCG")
        return False
    
    # Find CombinedLoss initialization
    pattern = r'CombinedLoss\s*\([^)]*\)'
    matches = list(re.finditer(pattern, content))
    
    if not matches:
        print(f"  ⚠️  {script_path.name}: No CombinedLoss found")
        return False
    
    # Update each match
    updated = False
    for match in reversed(matches):  # Reverse to maintain positions
        old_call = match.group(0)
        
        # Check if it already has neural_ndcg parameters
        if 'use_neural_ndcg' in old_call:
            continue
        
        # Add NeuralNDCG parameters
        if old_call.endswith(')'):
            # Insert before closing paren
            new_call = old_call[:-1] + ', use_neural_ndcg=True, neural_ndcg_weight=0.5)'
        else:
            new_call = old_call + ', use_neural_ndcg=True, neural_ndcg_weight=0.5'
        
        content = content[:match.start()] + new_call + content[match.end():]
        updated = True
    
    if updated:
        with open(script_path, 'w') as f:
            f.write(content)
        print(f"  ✅ Updated {script_path.name}")
        return True
    
    return False

def main():
    print("Updating training scripts to use NeuralNDCG...")
    print("=" * 70)
    
    updated_count = 0
    for script_path_str in TRAINING_SCRIPTS:
        script_path = Path(script_path_str)
        if not script_path.exists():
            print(f"  ⚠️  {script_path.name}: File not found")
            continue
        
        if update_script(script_path):
            updated_count += 1
    
    print("=" * 70)
    print(f"✅ Updated {updated_count} scripts")
    print("\nNext steps:")
    print("  1. Test updated scripts")
    print("  2. Run experiments with NeuralNDCG enabled")
    print("  3. Compare performance vs previous runs")

if __name__ == "__main__":
    main()

