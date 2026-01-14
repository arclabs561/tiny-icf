#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pandas>=2.0.0",
# ]
# ///
"""
Monitor experiments using trainctl utilities.

Provides unified monitoring interface using trainctl's metrics_loader.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import time
import argparse

# Add trainctl utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'trainctl' / 'utils'))

try:
    from trainctl.utils.metrics_loader import (
        list_experiments,
        get_experiment_status,
        get_latest_metrics,
        compare_experiments,
    )
    from trainctl.utils.storage_manager import get_storage_usage
    HAS_TRAINCTL_UTILS = True
except ImportError:
    HAS_TRAINCTL_UTILS = False
    print("❌ trainctl utilities not available.")
    sys.exit(1)


def monitor_experiments(
    experiment_names: Optional[List[str]] = None,
    follow: bool = False,
    interval: int = 10,
):
    """
    Monitor experiments using trainctl utilities.
    
    Args:
        experiment_names: Specific experiments to monitor (None = all)
        follow: If True, continuously monitor (like tail -f)
        interval: Update interval in seconds (when follow=True)
    """
    if experiment_names is None:
        experiment_names = list_experiments()
    
    if not experiment_names:
        print("No experiments found")
        return
    
    while True:
        # Clear screen (ANSI escape code)
        print("\033[2J\033[H", end="")
        
        print(f"{'='*80}")
        print(f"Experiment Monitor (trainctl) - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Get status for all experiments
        statuses = []
        for exp_name in experiment_names:
            exp_dir = Path(__file__).parent.parent / 'models' / exp_name
            if exp_dir.exists():
                status = get_experiment_status(exp_dir)
                if status:
                    statuses.append(status)
        
        # Sort by status (running first) then by best spearman
        statuses.sort(key=lambda x: (
            0 if x.get('status') == 'running' else 1,
            -x.get('best_spearman', 0.0)
        ))
        
        # Print table
        print(f"{'Experiment':<35} {'Status':<12} {'Epoch':<8} {'Best Spearman':>15} {'Current':>15}")
        print("-" * 80)
        
        for status in statuses:
            name = status.get('name', 'unknown')[:33]
            exp_status = status.get('status', 'unknown')[:10]
            epochs = status.get('epochs_trained', 0)
            best_spearman = status.get('best_spearman', 0.0)
            current_spearman = status.get('current_spearman', 0.0)
            
            print(f"{name:<35} {exp_status:<12} {epochs:<8} {best_spearman:>15.4f} {current_spearman:>15.4f}")
        
        # Storage summary
        stats = get_storage_usage()
        print(f"\n{'='*80}")
        print(f"Storage: {stats['num_experiments']} experiments, "
              f"{stats['num_checkpoints']} checkpoints, "
              f"{stats['total_size_mb']:.2f} MB")
        
        if not follow:
            break
        
        time.sleep(interval)


def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(description="Monitor experiments using trainctl")
    parser.add_argument('--experiments', nargs='+',
                       help='Specific experiments to monitor')
    parser.add_argument('--follow', '-f', action='store_true',
                       help='Continuously monitor (like tail -f)')
    parser.add_argument('--interval', type=int, default=10,
                       help='Update interval in seconds (default: 10)')
    
    args = parser.parse_args()
    
    try:
        monitor_experiments(
            experiment_names=args.experiments,
            follow=args.follow,
            interval=args.interval,
        )
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == '__main__':
    main()

