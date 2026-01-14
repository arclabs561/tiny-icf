#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "torch>=2.0.0",
# ]
# ///
"""
Validate checkpoint files for integrity and completeness.
"""

import argparse
import sys
from pathlib import Path
import torch


def validate_checkpoint(checkpoint_path: Path) -> bool:
    """Validate checkpoint file structure and integrity."""
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False
    
    required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict']
    missing_keys = [key for key in required_keys if key not in checkpoint]
    
    if missing_keys:
        print(f"❌ Missing required keys: {missing_keys}")
        return False
    
    # Validate state dicts are not empty
    if not checkpoint['model_state_dict']:
        print("❌ Model state dict is empty")
        return False
    
    if not checkpoint['optimizer_state_dict']:
        print("❌ Optimizer state dict is empty")
        return False
    
    # Check for best model state
    if 'best_model_state' in checkpoint and checkpoint['best_model_state'] is not None:
        if not checkpoint['best_model_state']:
            print("⚠️  Best model state dict is empty")
    
    print(f"✅ Checkpoint valid: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Best Spearman: {checkpoint.get('best_spearman', 'unknown')}")
    print(f"   Model keys: {len(checkpoint['model_state_dict'])}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate checkpoint files")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint file path")
    args = parser.parse_args()
    
    if not validate_checkpoint(args.checkpoint):
        sys.exit(1)


if __name__ == "__main__":
    main()

