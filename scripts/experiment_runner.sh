#!/bin/bash
# Quick experiment runner wrapper
# Makes it easy to run experiments with common patterns

set -e

EXPERIMENT=$1
shift  # Remove experiment name, pass rest to script

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: $0 <experiment> [args...]"
    echo ""
    echo "Available experiments:"
    python3 scripts/run_experiment.py --list
    exit 1
fi

# Run the experiment
python3 scripts/run_experiment.py "$EXPERIMENT" --data data/word_frequency.csv "$@"

