# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
# ]
# ///
"""Restore residual experiment history from checkpoint."""

import json
import torch
from pathlib import Path

checkpoint_path = Path('models/checkpoint_residual.pt')
history_path = Path('models/history_residual.json')

if checkpoint_path.exists():
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        history = ckpt.get('history', {})
        
        if 'val' in history and len(history['val']) > 0:
            val_spearman = [e.get('spearman_corr', e.get('spearman', 0.0)) for e in history['val']]
            best_val = max(val_spearman) if val_spearman else 0.0
            print(f"✓ Found history in checkpoint:")
            print(f"  Train epochs: {len(history.get('train', []))}")
            print(f"  Val epochs: {len(history.get('val', []))}")
            print(f"  Best val Spearman: {best_val:.4f}")
            
            # Convert any torch tensors to Python types
            def convert_to_python(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_python(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_python(item) for item in obj]
                elif isinstance(obj, torch.Tensor):
                    return float(obj.item())
                elif isinstance(obj, (int, float)):
                    return float(obj)
                else:
                    return obj
            
            history_clean = convert_to_python(history)
            
            # Save to JSON
            with open(history_path, 'w') as f:
                json.dump(history_clean, f, indent=2)
            print(f"  ✓ Restored history to JSON")
        else:
            print("✗ No val history in checkpoint")
            print(f"  Available keys: {list(ckpt.keys())}")
            if 'history' in ckpt:
                print(f"  History keys: {list(ckpt['history'].keys())}")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
else:
    print("✗ No checkpoint found")

