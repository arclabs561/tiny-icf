#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pandas>=2.0.0",
# ]
# ///
"""
Archive completed experiments using trainctl utilities.

Automatically archives experiments that are:
- Completed (status != 'running')
- Older than specified days (optional)
"""

import sys
from pathlib import Path
from typing import List, Optional
import argparse

# Add trainctl utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'trainctl' / 'utils'))

try:
    from trainctl.utils.storage_manager import (
        archive_experiment,
        cleanup_old_experiments,
        get_storage_usage,
    )
    from trainctl.utils.metrics_loader import (
        list_experiments,
        get_experiment_status,
    )
    HAS_TRAINCTL_UTILS = True
except ImportError:
    HAS_TRAINCTL_UTILS = False
    print("❌ trainctl utilities not available. Install trainctl utilities.")
    sys.exit(1)


def archive_experiments(
    experiment_names: Optional[List[str]] = None,
    days_old: Optional[int] = None,
    keep_best_only: bool = True,
    dry_run: bool = False,
) -> List[str]:
    """
    Archive experiments.
    
    Args:
        experiment_names: Specific experiments to archive (None = all completed)
        days_old: Only archive experiments older than this many days
        keep_best_only: If True, keep only best checkpoint + metrics in archive
        dry_run: If True, don't actually archive, just show what would be archived
    
    Returns:
        List of archived experiment names
    """
    if not HAS_TRAINCTL_UTILS:
        return []
    
    archived = []
    
    if days_old is not None:
        # Use trainctl's cleanup function
        if not dry_run:
            archived = cleanup_old_experiments(days_old=days_old, archive=True)
        else:
            # Dry run: just list what would be archived
            all_experiments = list_experiments()
            for exp_name in all_experiments:
                status = get_experiment_status(Path(__file__).parent.parent / 'models' / exp_name)
                if status and status.get('status') != 'running':
                    archived.append(exp_name)
    else:
        # Archive specific experiments
        if experiment_names is None:
            # Archive all completed experiments
            experiment_names = list_experiments()
        
        for exp_name in experiment_names:
            exp_dir = Path(__file__).parent.parent / 'models' / exp_name
            if not exp_dir.exists():
                print(f"⚠️  Experiment {exp_name} not found")
                continue
            
            status = get_experiment_status(exp_dir)
            if status and status.get('status') == 'running':
                print(f"⏸️  Skipping {exp_name} (still running)")
                continue
            
            if dry_run:
                print(f"📦 Would archive: {exp_name}")
                archived.append(exp_name)
            else:
                try:
                    archive_path = archive_experiment(exp_name, keep_best_only=keep_best_only)
                    print(f"✅ Archived {exp_name} to {archive_path}")
                    archived.append(exp_name)
                except Exception as e:
                    print(f"❌ Failed to archive {exp_name}: {e}")
    
    return archived


def main():
    """Main archival function."""
    parser = argparse.ArgumentParser(description="Archive completed experiments")
    parser.add_argument('--experiments', nargs='+',
                       help='Specific experiment names to archive')
    parser.add_argument('--days-old', type=int,
                       help='Archive experiments older than this many days')
    parser.add_argument('--keep-all', action='store_true',
                       help='Keep all files in archive (not just best checkpoint)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be archived without actually archiving')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No experiments will be archived\n")
    
    archived = archive_experiments(
        experiment_names=args.experiments,
        days_old=args.days_old,
        keep_best_only=not args.keep_all,
        dry_run=args.dry_run,
    )
    
    if archived:
        print(f"\n✅ {'Would archive' if args.dry_run else 'Archived'} {len(archived)} experiment(s)")
    else:
        print("\n📭 No experiments to archive")


if __name__ == '__main__':
    main()

