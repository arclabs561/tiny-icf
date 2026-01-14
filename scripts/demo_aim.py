#!/usr/bin/env -S uv run
"""
Demo script showing Aim integration and what gets tracked.
This creates a mock training run to demonstrate Aim's capabilities.
"""

import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.aim_tracker import AimTracker


def demo_aim_tracking():
    """Demonstrate Aim tracking with mock training metrics."""
    
    print("=" * 70)
    print("Aim Integration Demo")
    print("=" * 70)
    print()
    
    # Initialize tracker
    print("Initializing Aim tracker...")
    with AimTracker(
        experiment_name="demo-experiment",
        run_name="demo-run-1",
    ) as tracker:
        
        if not tracker:
            print("⚠️  Aim not available or tracking disabled")
            print("   Install with: uv sync")
            return
        
        print("✓ Aim tracker initialized")
        print()
        
        # Track hyperparameters
        print("Tracking hyperparameters...")
        tracker.track_params({
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "max_length": 20,
            "augment_prob": 0.1,
            "model_params": 45000,
            "device": "cpu",
            "seed": 42,
        })
        print("✓ Hyperparameters tracked")
        print()
        
        # Simulate training loop with metrics
        print("Simulating training loop (5 epochs)...")
        for epoch in range(5):
            # Simulate training metrics
            train_loss = 0.5 - (epoch * 0.05) + (0.01 * (epoch % 2))
            val_loss = 0.52 - (epoch * 0.04) + (0.02 * (epoch % 2))
            spearman = 0.3 + (epoch * 0.1) + (0.05 * (epoch % 2))
            mae = 0.15 - (epoch * 0.02)
            
            # Track metrics
            tracker.track_metric("train_loss", train_loss, step=epoch)
            tracker.track_metric("val_loss", val_loss, step=epoch)
            tracker.track_metric("val_spearman_corr", spearman, step=epoch)
            tracker.track_metric("val_mae", mae, step=epoch)
            
            # Track prediction statistics
            tracker.track_metrics({
                "train_pred_mean": 0.4 + (epoch * 0.01),
                "train_pred_std": 0.25 - (epoch * 0.01),
                "train_pred_min": 0.0,
                "train_pred_max": 0.9 + (epoch * 0.01),
            }, step=epoch)
            
            # Track learning rate (decreasing)
            lr = 0.001 * (0.9 ** epoch)
            tracker.track_metric("learning_rate", lr, step=epoch)
            
            print(f"  Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"spearman={spearman:.4f}, mae={mae:.4f}")
            time.sleep(0.1)  # Small delay for demo
        
        print("✓ Training metrics tracked")
        print()
        
        # Track Jabberwocky results
        print("Tracking Jabberwocky Protocol results...")
        jabberwocky_words = ["the", "xylophone", "flimjam", "qzxbjk", "unfriendliness"]
        for word in jabberwocky_words:
            # Mock predictions
            if word == "the":
                pred = 0.05
            elif word == "xylophone":
                pred = 0.75
            elif word == "flimjam":
                pred = 0.65
            elif word == "qzxbjk":
                pred = 0.95
            else:  # unfriendliness
                pred = 0.55
            
            tracker.track_metric(f"jabberwocky_{word}", pred, step=4)
        
        tracker.track_metric("jabberwocky_pass_rate", 0.8, step=4)
        print("✓ Jabberwocky results tracked")
        print()
        
        # Track final best metrics
        print("Tracking best model metrics...")
        tracker.track_metrics({
            "best_spearman": 0.75,
            "best_epoch": 4,
            "best_val_loss": 0.32,
        })
        print("✓ Best metrics tracked")
        print()
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print()
    print("To view this run in Aim UI:")
    print("  1. Run: aim up")
    print("  2. Open: http://127.0.0.1:43800")
    print("  3. Look for experiment: 'demo-experiment'")
    print()
    print("You'll see:")
    print("  • Hyperparameters in the 'Params' tab")
    print("  • Training/validation metrics as line charts")
    print("  • Jabberwocky predictions")
    print("  • Learning rate schedule")
    print("  • Best model metrics")


if __name__ == "__main__":
    demo_aim_tracking()

