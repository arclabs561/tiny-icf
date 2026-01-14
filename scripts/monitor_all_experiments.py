#!/usr/bin/env python3
"""Monitor all running experiments in parallel."""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add trainctl to path
trainctl_path = Path("../trainctl")
if trainctl_path.exists():
    sys.path.insert(0, str(trainctl_path))
    sys.path.insert(0, str(trainctl_path / "utils"))

try:
    from metrics_loader import get_experiment_status, get_training_progress, get_best_metrics
    TRAINCTL_AVAILABLE = True
except ImportError:
    TRAINCTL_AVAILABLE = False


def get_running_experiments() -> List[str]:
    """Get list of running experiment names."""
    result = subprocess.run(['pgrep', '-f', 'train_flexible_opportunistic'], 
                           capture_output=True, text=True)
    if result.returncode != 0:
        return []
    
    # Find experiment directories with active training
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    running = []
    for exp_dir in models_dir.iterdir():
        if exp_dir.is_dir():
            log_file = exp_dir / "training.log"
            if log_file.exists():
                # Check if log was recently modified (within last 5 minutes)
                import time
                if time.time() - log_file.stat().st_mtime < 300:
                    running.append(exp_dir.name)
    
    return running


def get_experiment_info(exp_name: str) -> Dict:
    """Get status info for an experiment."""
    exp_dir = Path(f"models/{exp_name}")
    
    info = {
        'name': exp_name,
        'status': 'unknown',
        'epoch': 0,
        'val_spearman': 0.0,
        'best_spearman': 0.0,
        'best_epoch': 0,
        'eta': None,
    }
    
    if not TRAINCTL_AVAILABLE:
        return info
    
    try:
        status = get_experiment_status(exp_dir)
        if status:
            info['status'] = status.get('status', 'unknown')
            info['epoch'] = status.get('epoch', 0)
            info['val_spearman'] = status.get('val_spearman_corr', 0.0)
            info['best_spearman'] = status.get('best_spearman', 0.0)
            info['best_epoch'] = status.get('best_epoch', 0)
        
        progress = get_training_progress(exp_dir)
        if progress:
            eta = progress.get('eta_completion')
            if eta:
                hours = int(eta / 3600)
                mins = int((eta % 3600) / 60)
                info['eta'] = f"{hours}h {mins}m"
    except:
        pass
    
    return info


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m {int(seconds%60)}s"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"


def main():
    """Monitor all experiments."""
    print("=" * 70)
    print("MONITORING ALL EXPERIMENTS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Refreshing every 30 seconds")
    print("Press Ctrl+C to stop")
    print()
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            now = datetime.now().strftime('%H:%M:%S')
            
            if iteration > 1:
                print('\033[2J\033[H', end='')
            
            print("=" * 70)
            print(f"All Experiments Monitor | {now} | Check #{iteration}")
            print("=" * 70)
            print()
            
            # Get running experiments
            running_experiments = get_running_experiments()
            
            if not running_experiments:
                print("⚠️  No active experiments found")
                print("   (Check models/*/training.log for recent activity)")
            else:
                print(f"🔄 Active Experiments: {len(running_experiments)}")
                print()
                
                # Get info for each experiment
                for exp_name in sorted(running_experiments):
                    info = get_experiment_info(exp_name)
                    
                    print(f"📊 {info['name']}:")
                    print(f"   Status: {info['status']}")
                    print(f"   Epoch: {info['epoch']}/150")
                    print(f"   Val Spearman: {info['val_spearman']:.4f}")
                    print(f"   Best: {info['best_spearman']:.4f} (epoch {info['best_epoch']})")
                    if info['eta']:
                        print(f"   ETA: {info['eta']}")
                    print()
            
            # Check process count
            result = subprocess.run(['pgrep', '-f', 'train_flexible_opportunistic'], 
                                  capture_output=True, text=True)
            pids = [p for p in result.stdout.strip().split('\n') if p]
            print(f"🔄 Total Training Processes: {len(pids)}")
            print()
            
            print("⏱️  Next update in 30 seconds... (Ctrl+C to stop)")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
        sys.exit(0)


if __name__ == '__main__':
    main()

