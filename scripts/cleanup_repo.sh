#!/bin/bash
# Cleanup script to organize the repository
# Removes redundant files and organizes structure

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  REPOSITORY CLEANUP"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Create archive directory for old files
ARCHIVE_DIR="archive/$(date +%Y%m%d)"
mkdir -p "$ARCHIVE_DIR"

echo "📦 Archiving old files to $ARCHIVE_DIR"
echo "───────────────────────────────────────────────────────────────"

# Archive old log files (keep recent ones)
if [ -d "training_history" ]; then
    echo "  Moving old training logs..."
    find training_history -name "*.log" -mtime +30 -exec mv {} "$ARCHIVE_DIR/" \; 2>/dev/null || true
fi

# Archive old model checkpoints (keep best models)
echo "  Archiving old checkpoints..."
find models -name "checkpoint_*.pt" -mtime +7 -exec mv {} "$ARCHIVE_DIR/" \; 2>/dev/null || true

# Archive old zip files
echo "  Archiving old zip files..."
find . -maxdepth 1 -name "*.zip" -exec mv {} "$ARCHIVE_DIR/" \; 2>/dev/null || true

# Archive redundant markdown files (keep essential ones)
echo "  Archiving redundant documentation..."
ESSENTIAL_DOCS=(
    "README.md"
    "PROJECT_PURPOSE.md"
    "EXPERIMENTS.md"
    "EPHEMERAL_TRAINING.md"
    "JABBERWOCKY_PROTOCOL.md"
    "QUICK_START.md"
    "TRAINING_GUIDE.md"
)

REDUNDANT_PATTERNS=(
    "*SUMMARY*.md"
    "*COMPLETE*.md"
    "*FINAL*.md"
    "*STATUS*.md"
    "*NEXT_STEPS*.md"
    "*ITERATION*.md"
    "*SESSION*.md"
)

for pattern in "${REDUNDANT_PATTERNS[@]}"; do
    for file in $pattern; do
        if [ -f "$file" ]; then
            # Check if it's essential
            is_essential=false
            for essential in "${ESSENTIAL_DOCS[@]}"; do
                if [ "$file" == "$essential" ]; then
                    is_essential=true
                    break
                fi
            done
            
            if [ "$is_essential" == "false" ]; then
                echo "    Archiving: $file"
                mv "$file" "$ARCHIVE_DIR/" 2>/dev/null || true
            fi
        fi
    done
done

# Clean up Python cache
echo ""
echo "🧹 Cleaning Python cache..."
echo "───────────────────────────────────────────────────────────────"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Clean up test cache
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Organize log files
echo ""
echo "📁 Organizing log files..."
echo "───────────────────────────────────────────────────────────────"
mkdir -p logs/training
mkdir -p logs/experiments

# Move training logs
find . -maxdepth 1 -name "training*.log" -exec mv {} logs/training/ \; 2>/dev/null || true
find . -maxdepth 1 -name "*.log" -exec mv {} logs/experiments/ \; 2>/dev/null || true

# Create consolidated documentation
echo ""
echo "📝 Creating consolidated documentation..."
echo "───────────────────────────────────────────────────────────────"

# Summary
cat > DOCS_SUMMARY.md << 'EOF'
# Documentation Summary

## Essential Documentation
- `README.md` - Main project overview
- `PROJECT_PURPOSE.md` - What this repo does and why
- `EXPERIMENTS.md` - Experiment history and results
- `EPHEMERAL_TRAINING.md` - Training on ephemeral environments
- `JABBERWOCKY_PROTOCOL.md` - Evaluation protocol
- `QUICK_START.md` - Quick start guide
- `TRAINING_GUIDE.md` - Detailed training guide

## Organized Documentation
All other documentation is organized in `docs/` subdirectories:
- `docs/guides/` - Training guides and quick starts
- `docs/results/` - Experiment results and analysis
- `docs/design/` - Design decisions and plans
- `docs/integrations/` - Integration guides
- `docs/concepts/` - Concepts and ideas
- `docs/technical/` - Technical improvements

## Archive
Old documentation and logs are archived in `archive/` directory.
EOF

echo "  ✓ Created DOCS_SUMMARY.md"

# Final organization - move remaining non-essential files
echo ""
echo "📁 Final organization pass..."
echo "───────────────────────────────────────────────────────────────"

mkdir -p docs/concepts docs/technical

# Move remaining files
for file in CODE_REVIEW.md CRITICAL_ISSUES.md MULTI_OBJECTIVE_AND_TEMPORAL.md; do
    if [ -f "$file" ]; then
        echo "  → $file -> docs/technical/"
        mv "$file" docs/technical/ 2>/dev/null || true
    fi
done

for file in DATA_AND_MODELS.md DATA_PREP.md PROJECT_OVERVIEW.md; do
    if [ -f "$file" ]; then
        echo "  → $file -> docs/"
        mv "$file" docs/ 2>/dev/null || true
    fi
done

for file in GOALS_AND_STRATEGY.md GOALS_REFINED.md; do
    if [ -f "$file" ]; then
        echo "  → $file -> docs/design/"
        mv "$file" docs/design/ 2>/dev/null || true
    fi
done

for file in LATEST_BREAKTHROUGH.md; do
    if [ -f "$file" ]; then
        echo "  → $file -> docs/results/"
        mv "$file" docs/results/ 2>/dev/null || true
    fi
done

for file in QUICK_REFERENCE.md QUICK_START_TEMPORAL_AMOO.md; do
    if [ -f "$file" ]; then
        echo "  → $file -> docs/guides/"
        mv "$file" docs/guides/ 2>/dev/null || true
    fi
done

echo "  ✓ Final organization complete"

# Summary
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ CLEANUP COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Archived files: $ARCHIVE_DIR"
echo "Logs organized: logs/"
echo ""
echo "Essential docs remain in root. See DOCS_SUMMARY.md for details."

