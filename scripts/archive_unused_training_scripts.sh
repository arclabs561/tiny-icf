#!/bin/bash
# Archive unused training scripts, keeping only the main ones

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE_DIR="$ROOT_DIR/archive/unused_training_scripts_$(date +%Y%m%d)"
mkdir -p "$ARCHIVE_DIR"

echo "📦 Archiving unused training scripts..."
echo "Archive directory: $ARCHIVE_DIR"
echo ""

# Main scripts to KEEP (do not archive)
KEEP_SCRIPTS=(
    "scripts/train_flexible_opportunistic.py"
    "scripts/scale_gpu_training.sh"
    "scripts/monitor_aws_training.sh"
    "scripts/show_training_results.sh"
    "scripts/run_flexible_training.sh"
    "scripts/monitor_residual_experiments.sh"
)

# Find all training scripts
TRAINING_SCRIPTS=$(find scripts -name "train*.py" -o -name "train*.sh" | sort)

ARCHIVED=0
KEPT=0

for script in $TRAINING_SCRIPTS; do
    # Check if script should be kept
    KEEP=false
    for keep_script in "${KEEP_SCRIPTS[@]}"; do
        if [ "$script" = "$keep_script" ]; then
            KEEP=true
            break
        fi
    done
    
    if [ "$KEEP" = true ]; then
        echo "✅ KEEP: $script"
        KEPT=$((KEPT + 1))
    else
        echo "📦 ARCHIVE: $script"
        mv "$script" "$ARCHIVE_DIR/"
        ARCHIVED=$((ARCHIVED + 1))
    fi
done

echo ""
echo "Summary:"
echo "  Kept: $KEPT scripts"
echo "  Archived: $ARCHIVED scripts"
echo "  Archive location: $ARCHIVE_DIR"
