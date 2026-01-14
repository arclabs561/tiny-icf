#!/bin/bash
# Organize root directory markdown files into appropriate locations

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE_DIR="$ROOT_DIR/archive/root_markdown_$(date +%Y%m%d)"
mkdir -p "$ARCHIVE_DIR"

echo "📁 Organizing root directory markdown files..."
echo "Archive directory: $ARCHIVE_DIR"
echo ""

# Categories
STATUS_FILES=(
    "ALL_FIXES_COMPLETE.md"
    "COMPLETE_IMPLEMENTATION.md"
    "COMPLETE_IMPROVEMENTS.md"
    "COMPLETE_STATUS.md"
    "COMPLETE_SUMMARY.md"
    "CURRENT_STATUS.md"
    "FINAL_STATUS.md"
    "SESSION_IMPROVEMENTS.md"
    "SUMMARY.md"
    "TRAINING_STATUS.md"
)

DESIGN_FILES=(
    "ARGUMENT_AGAINST_MAPPING.md"
    "BETTER_THAN_MAPPING.md"
    "CRITIQUE_AND_RECOMMENDATIONS.md"
    "DESIGN_CRITIQUE.md"
    "DESIGN_CRITIQUE_ENHANCED.md"
    "DESIGN_CRITIQUE_FINAL.md"
    "DESIGN_REVIEW.md"
    "FUN_PROJECT_PHILOSOPHY.md"
    "GOALS_AND_STRATEGY.md"
    "GOALS_CRITIQUE_SUMMARY.md"
    "GOALS_REFINED.md"
    "IMPLEMENTATION_BEST_PRACTICE.md"
    "IMPLEMENTATION_PLAN.md"
    "IMPLEMENTATION_REVIEW.md"
    "MODEL_OPTIMIZATION_PLAN.md"
    "PRODUCT_DECISION.md"
    "PRODUCT_THINKING.md"
)

GUIDE_FILES=(
    "BATCH_TRAINING_GUIDE.md"
    "CURRENT_TRAINING_SCRIPTS.md"
    "DATA_AND_MODELS.md"
    "DATA_PREP.md"
    "DOCS_SUMMARY.md"
    "ENHANCED_TRAINING.md"
    "EPHEMERAL_TRAINING.md"
    "GPU_SCALING_QUICK_START.md"
    "GPU_SCALING_SOLUTION.md"
    "QUICK_REFERENCE.md"
    "QUICK_START.md"
    "QUICKSTART_BATCH.md"
    "QUICKSTART.md"
    "README_BATCH.md"
    "README_ROOT.md"
    "README_RUNPOD.md"
    "README_TRAINING.md"
    "TRAINING_GUIDE.md"
)

RESULTS_FILES=(
    "EXPERIMENT_RESULTS.md"
    "EXPERIMENTS_COMPLETE.md"
    "EXPERIMENTS_README.md"
    "EXPERIMENTS.md"
    "FINAL_ANALYSIS.md"
    "FINAL_RESULTS.md"
    "REAL_WORLD_TEST_RESULTS.md"
)

# Move files
move_files() {
    local dest_dir="$1"
    shift
    local files=("$@")
    
    mkdir -p "$dest_dir"
    for file in "${files[@]}"; do
        if [ -f "$ROOT_DIR/$file" ]; then
            echo "  Moving $file -> $dest_dir/"
            mv "$ROOT_DIR/$file" "$dest_dir/"
        fi
    done
}

echo "📊 Status files -> docs/results/status/"
move_files "$ROOT_DIR/docs/results/status" "${STATUS_FILES[@]}"

echo ""
echo "🎨 Design files -> docs/design/"
move_files "$ROOT_DIR/docs/design" "${DESIGN_FILES[@]}"

echo ""
echo "📖 Guide files -> docs/guides/"
move_files "$ROOT_DIR/docs/guides" "${GUIDE_FILES[@]}"

echo ""
echo "📈 Results files -> docs/results/"
move_files "$ROOT_DIR/docs/results" "${RESULTS_FILES[@]}"

echo ""
echo "✅ Organization complete!"
echo "Files moved to appropriate directories"
