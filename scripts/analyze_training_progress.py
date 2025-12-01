"""Analyze training progress from log file."""

import re
from pathlib import Path
from collections import defaultdict


def parse_training_log(log_path: Path):
    """Parse training log and extract metrics."""
    epochs = []
    current_epoch = None
    
    with open(log_path, 'r') as f:
        for line in f:
            # Epoch line
            epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
            if epoch_match:
                epoch_num = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                current_epoch = {
                    'epoch': epoch_num,
                    'total': total_epochs,
                    'stage': None,
                    'progress': None,
                    'train_loss': None,
                    'val_loss': None,
                    'saved': False,
                }
            
            # Stage progress
            if current_epoch and 'Stage' in line:
                stage_match = re.search(r'Stage (\d+)/(\d+)', line)
                if stage_match:
                    current_epoch['stage'] = int(stage_match.group(1))
                    current_epoch['total_stages'] = int(stage_match.group(2))
                
                progress_match = re.search(r'(\d+\.\d+)% progress', line)
                if progress_match:
                    current_epoch['progress'] = float(progress_match.group(1))
            
            # Loss values
            if current_epoch and 'loss' in line.lower():
                train_match = re.search(r'Train loss: ([\d.]+)', line)
                if train_match:
                    current_epoch['train_loss'] = float(train_match.group(1))
                
                val_match = re.search(r'Val loss: ([\d.]+)', line)
                if val_match:
                    current_epoch['val_loss'] = float(val_match.group(1))
            
            # Saved model
            if current_epoch and 'Saved best model' in line:
                current_epoch['saved'] = True
                epochs.append(current_epoch.copy())
                current_epoch = None
            
            # Training complete
            if 'Training complete' in line:
                complete_match = re.search(r'Best validation loss: ([\d.]+)', line)
                if complete_match:
                    return {
                        'complete': True,
                        'best_val_loss': float(complete_match.group(1)),
                        'epochs': epochs,
                    }
    
    return {
        'complete': False,
        'current_epoch': current_epoch,
        'epochs': epochs,
    }


def print_analysis(result):
    """Print training analysis."""
    print("=" * 80)
    print("Training Progress Analysis")
    print("=" * 80)
    
    if result['complete']:
        print(f"\n✓ Training Complete!")
        print(f"  Best validation loss: {result['best_val_loss']:.6f}")
    else:
        if result.get('current_epoch'):
            ep = result['current_epoch']
            print(f"\n⏳ Training In Progress")
            print(f"  Current epoch: {ep['epoch']}/{ep['total']} ({ep['epoch']/ep['total']*100:.1f}%)")
            if ep.get('stage'):
                print(f"  Curriculum stage: {ep['stage']}/{ep.get('total_stages', '?')}")
            if ep.get('progress') is not None:
                print(f"  Stage progress: {ep['progress']:.1f}%")
        else:
            print("\n⏳ Training In Progress (checking log...)")
    
    epochs = result.get('epochs', [])
    if epochs:
        print(f"\nCompleted Epochs: {len(epochs)}")
        print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Saved':<8} {'Stage':<10}")
        print("-" * 60)
        
        for ep in epochs[-10:]:  # Last 10 epochs
            saved = "✓" if ep.get('saved') else ""
            stage = f"{ep.get('stage', '?')}/{ep.get('total_stages', '?')}" if ep.get('stage') else "-"
            train_loss = f"{ep['train_loss']:.6f}" if ep.get('train_loss') else "-"
            val_loss = f"{ep['val_loss']:.6f}" if ep.get('val_loss') else "-"
            
            print(f"{ep['epoch']:<8} {train_loss:<12} {val_loss:<12} {saved:<8} {stage:<10}")
        
        # Loss trends
        if len(epochs) >= 2:
            train_losses = [ep['train_loss'] for ep in epochs if ep.get('train_loss')]
            val_losses = [ep['val_loss'] for ep in epochs if ep.get('val_loss')]
            
            if train_losses:
                print(f"\nLoss Trends:")
                print(f"  Train loss: {train_losses[0]:.6f} → {train_losses[-1]:.6f} "
                      f"({(train_losses[-1] - train_losses[0]):.6f})")
                if val_losses:
                    print(f"  Val loss:   {val_losses[0]:.6f} → {val_losses[-1]:.6f} "
                          f"({(val_losses[-1] - val_losses[0]):.6f})")
                    
                    # Check if improving
                    if len(val_losses) >= 3:
                        recent_trend = val_losses[-3:]
                        if recent_trend[-1] < recent_trend[0]:
                            print(f"  ✓ Validation loss decreasing (improving)")
                        elif recent_trend[-1] > recent_trend[0]:
                            print(f"  ⚠ Validation loss increasing (may be overfitting)")
                        else:
                            print(f"  → Validation loss stable")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze training progress")
    parser.add_argument("--log", type=Path, default="training.log", help="Training log file")
    args = parser.parse_args()
    
    if not args.log.exists():
        print(f"Error: Log file not found: {args.log}")
        return
    
    result = parse_training_log(args.log)
    print_analysis(result)


if __name__ == "__main__":
    main()

