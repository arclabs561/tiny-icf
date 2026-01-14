# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "pandas>=2.0.0",
#   "tqdm>=4.65.0",
#   "scipy>=1.10.0",
# ]
# ///
"""
Unified experiment runner for tiny-icf experiments.

This script provides a consistent interface for running all experiments
with proper tracking, comparison, and organization.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Experiment configurations
EXPERIMENTS = {
    'residual': {
        'script': 'train_residual.py',
        'model': 'ResidualICF',
        'description': 'Residual connections for better gradient flow',
        'default_args': {
            'epochs': 100,
            'batch-size': 256,
            'lr': 1e-3,
            'rank-weight': 5.0,
            'dropout': 0.4,
            'weight-decay': 1e-4,
        }
    },
    'aggressive_reg': {
        'script': 'train_aggressive_regularization.py',
        'model': 'UniversalICF',
        'description': 'Aggressive regularization (dropout=0.5, weight_decay=1e-3)',
        'default_args': {
            'epochs': 100,
            'batch-size': 256,
            'lr': 1e-3,
            'rank-weight': 5.0,
            'dropout': 0.5,
            'weight-decay': 1e-3,
        }
    },
    'temporal_amoo': {
        'script': 'train_temporal_amoo.py',
        'model': 'UniversalICF',
        'description': 'Temporal data + Aligned Multi-Objective Optimization',
        'default_args': {
            'epochs': 100,
            'batch-size': 256,
            'lr': 1e-3,
        }
    },
    'batchnorm': {
        'script': 'train_batchnorm.py',
        'model': 'UniversalICF',
        'description': 'BatchNorm layers for normalization',
        'default_args': {
            'epochs': 100,
            'batch-size': 256,
            'lr': 1e-3,
            'rank-weight': 5.0,
        }
    },
    'reduced_capacity': {
        'script': 'train_reduced_capacity.py',
        'model': 'UniversalICF',
        'description': 'Reduced model capacity (37.9% fewer params)',
        'default_args': {
            'epochs': 100,
            'batch-size': 256,
            'lr': 1e-3,
            'rank-weight': 5.0,
        }
    },
    'gated_residual': {
        'script': 'train_gated_residual.py',
        'model': 'GatedResidualICF',
        'description': 'Gated residual connections',
        'default_args': {
            'epochs': 100,
            'batch-size': 256,
            'lr': 1e-3,
            'rank-weight': 5.0,
        }
    },
    'nano': {
        'script': 'train_nano.py',
        'model': 'NanoICF',
        'description': 'Ultra-small model (6,721 params)',
        'default_args': {
            'epochs': 100,
            'batch-size': 256,
            'lr': 1e-3,
            'rank-weight': 5.0,
        }
    },
    'ephemeral': {
        'script': 'train_ephemeral_robust.py',
        'model': 'ResidualICF',
        'description': 'Robust training for ephemeral environments',
        'default_args': {
            'epochs': 200,
            'batch-size': 256,
            'lr': 1e-3,
            'rank-weight': 5.0,
            'checkpoint-interval': 1,
        }
    },
}


def list_experiments():
    """List all available experiments."""
    print("Available Experiments:")
    print("=" * 80)
    for name, config in EXPERIMENTS.items():
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Model: {config['model']}")
        print(f"  Script: {config['script']}")
        print(f"  Default args: {config['default_args']}")


def run_experiment(experiment_name: str, override_args: Optional[Dict] = None):
    """Run a specific experiment."""
    if experiment_name not in EXPERIMENTS:
        print(f"❌ Unknown experiment: {experiment_name}")
        print("\nAvailable experiments:")
        for name in EXPERIMENTS.keys():
            print(f"  - {name}")
        return 1
    
    config = EXPERIMENTS[experiment_name]
    script_path = Path(__file__).parent / config['script']
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return 1
    
    # Build command
    import subprocess
    cmd = [sys.executable, str(script_path)]
    
    # Add default args
    for key, value in config['default_args'].items():
        arg_name = f"--{key.replace('_', '-')}"
        cmd.extend([arg_name, str(value)])
    
    # Override with user args
    if override_args:
        for key, value in override_args.items():
            arg_name = f"--{key.replace('_', '-')}"
            # Remove existing if present
            cmd = [c for c in cmd if not c.startswith(arg_name)]
            cmd.extend([arg_name, str(value)])
    
    print(f"🚀 Running experiment: {experiment_name}")
    print(f"   Description: {config['description']}")
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    # Run the script
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Unified experiment runner for tiny-icf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python scripts/run_experiment.py --list

  # Run residual experiment with defaults
  python scripts/run_experiment.py residual

  # Run with custom args
  python scripts/run_experiment.py residual --epochs 200 --batch-size 512

  # Run ephemeral training
  python scripts/run_experiment.py ephemeral --data data/word_frequency.csv
        """
    )
    
    parser.add_argument(
        'experiment',
        nargs='?',
        help='Experiment name to run (use --list to see all)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiments'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/word_frequency.csv',
        help='Path to training data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for models'
    )
    
    # Allow passing through other args
    args, unknown = parser.parse_known_args()
    
    if args.list:
        list_experiments()
        return 0
    
    if not args.experiment:
        parser.print_help()
        print("\n❌ Please specify an experiment name or use --list")
        return 1
    
    # Build override args
    override_args = {
        'data': args.data,
        'output-dir': args.output_dir,
    }
    
    # Parse unknown args (experiment-specific)
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:].replace('-', '_')
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                override_args[key] = unknown[i + 1]
                i += 2
            else:
                override_args[key] = True
                i += 1
        else:
            i += 1
    
    return run_experiment(args.experiment, override_args)


if __name__ == '__main__':
    sys.exit(main())

