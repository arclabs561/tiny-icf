#!/bin/bash
# Further organize root directory - move non-essential docs to subdirectories

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  FURTHER ROOT ORGANIZATION"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Create organized directory structure
mkdir -p docs/guides
mkdir -p docs/results
mkdir -p docs/design
mkdir -p docs/integrations
mkdir -p config

# Essential docs to keep in root
ESSENTIAL=(
    "README.md"
    "PROJECT_PURPOSE.md"
    "EXPERIMENTS.md"
    "EPHEMERAL_TRAINING.md"
    "JABBERWOCKY_PROTOCOL.md"
    "QUICK_START.md"
    "TRAINING_GUIDE.md"
    "DOCS_SUMMARY.md"
)

# Move guide/quickstart docs
echo "📚 Moving guides to docs/guides/..."
for file in *_GUIDE.md *_QUICK_START.md *_TRAINING.md QUICKSTART*.md README_*.md; do
    if [ -f "$file" ]; then
        is_essential=false
        for essential in "${ESSENTIAL[@]}"; do
            if [ "$file" == "$essential" ]; then
                is_essential=true
                break
            fi
        done
        if [ "$is_essential" == "false" ]; then
            echo "  → $file"
            mv "$file" docs/guides/ 2>/dev/null || true
        fi
    fi
done

# Move result/analysis docs
echo ""
echo "📊 Moving results/analysis to docs/results/..."
for file in *RESULTS*.md *ANALYSIS*.md *CRITIQUE*.md *PROGRESS*.md; do
    if [ -f "$file" ]; then
        echo "  → $file"
        mv "$file" docs/results/ 2>/dev/null || true
    fi
done

# Move design docs
echo ""
echo "🎨 Moving design docs to docs/design/..."
for file in *DESIGN*.md *PRODUCT*.md *IMPLEMENTATION*.md *PLAN*.md; do
    if [ -f "$file" ]; then
        echo "  → $file"
        mv "$file" docs/design/ 2>/dev/null || true
    fi
done

# Move integration docs
echo ""
echo "🔌 Moving integration docs to docs/integrations/..."
for file in *API*.md *MCP*.md *RUNPOD*.md *LYCEUM*.md *AIM*.md; do
    if [ -f "$file" ]; then
        echo "  → $file"
        mv "$file" docs/integrations/ 2>/dev/null || true
    fi
done

# Move other docs
echo ""
echo "📄 Moving other docs to docs/..."
for file in *DEBUG*.md *WORKAROUND*.md *SOLUTION*.md *DECISION*.md *OPTIONS*.md *METADATA*.md *USAGE*.md; do
    if [ -f "$file" ]; then
        echo "  → $file"
        mv "$file" docs/ 2>/dev/null || true
    fi
done

# Move model files to models/ if in root
echo ""
echo "🤖 Moving model files to models/..."
for file in model*.pt; do
    if [ -f "$file" ]; then
        echo "  → $file"
        mv "$file" models/ 2>/dev/null || true
    fi
done

# Move CSV files to data/ if in root
echo ""
echo "📈 Moving data files to data/..."
for file in *.csv; do
    if [ -f "$file" ] && [ ! -d "$file" ]; then
        echo "  → $file"
        mv "$file" data/ 2>/dev/null || true
    fi
done

# Move JSON configs
echo ""
echo "⚙️  Moving config files to config/..."
for file in *config.json runpod_config.json weights.json eval_existing.json; do
    if [ -f "$file" ]; then
        echo "  → $file"
        mv "$file" config/ 2>/dev/null || true
    fi
done

# Create index files
echo ""
echo "📝 Creating index files..."

cat > docs/README.md << 'EOF'
# Documentation

This directory contains organized documentation for the tiny-icf project.

## Structure

- `guides/` - Training guides, quick starts, and how-to documentation
- `results/` - Experiment results, analysis, and progress reports
- `design/` - Design decisions, implementation plans, and product thinking
- `integrations/` - Integration guides (RunPod, Lyceum, Aim, MCP, APIs)

## Essential Documentation

The most important documentation remains in the project root:
- `README.md` - Main project overview
- `PROJECT_PURPOSE.md` - What this repo does and why
- `EXPERIMENTS.md` - Experiment history and results
- `EPHEMERAL_TRAINING.md` - Training on ephemeral environments
- `JABBERWOCKY_PROTOCOL.md` - Evaluation protocol
- `QUICK_START.md` - Quick start guide
- `TRAINING_GUIDE.md` - Detailed training guide
EOF

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ ROOT ORGANIZATION COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Essential docs remain in root."
echo "Organized docs moved to docs/ subdirectories."
echo ""
echo "Remaining root files:"
ls -1 *.md 2>/dev/null | wc -l | xargs echo "  Markdown files:"
ls -1 *.pt *.json *.csv *.zip 2>/dev/null | wc -l | xargs echo "  Other files:"

