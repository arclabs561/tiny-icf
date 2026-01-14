#!/bin/bash
# Self-contained, idempotent setup script for ephemeral training
# Handles all dependencies, fixes, and verification automatically

set -e

SSH_HOST="${SSH_HOST:-213.173.111.79}"
SSH_PORT="${SSH_PORT:-34185}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

echo "═══════════════════════════════════════════════════════════════"
echo "  EPHEMERAL TRAINING SETUP"
echo "═══════════════════════════════════════════════════════════════"
echo "  Host: $SSH_HOST:$SSH_PORT"
echo ""

ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p "$SSH_PORT" root@"$SSH_HOST" << ENDSSH
cd /root/idf-est
export PATH="$HOME/.cargo/bin:$PATH"

echo "🔍 CHECKING ENVIRONMENT"
echo "───────────────────────────────────────────────────────────────"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE "[0-9]+\.[0-9]+")
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "  Python: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "  ⚠️  Python 3.8+ required"
    exit 1
fi

# Check/install uv
if ! command -v uv &> /dev/null; then
    echo "  📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "  ✓ uv installed"
else
    echo "  ✓ uv already installed ($(uv --version))"
    export PATH="$HOME/.cargo/bin:$PATH"
    # Update uv
    uv self update 2>&1 | grep -E "(Upgraded|already)" || true
fi

echo ""

echo "📦 CHECKING DEPENDENCIES"
echo "───────────────────────────────────────────────────────────────"

# Check if dependencies are installed
MISSING_DEPS=()
for dep in torch numpy pandas tqdm scipy; do
    if ! python3 -c "import $dep" 2>/dev/null; then
        MISSING_DEPS+=($dep)
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "  Installing missing dependencies: ${MISSING_DEPS[*]}"
    uv pip install --system ${MISSING_DEPS[*]} 2>&1 | grep -E "(Installing|Successfully|Requirement)" | tail -3
    echo "  ✓ Dependencies installed"
else
    echo "  ✓ All dependencies installed"
    python3 -c "import torch; print(f'    PyTorch: {torch.__version__}')" 2>&1
fi

echo ""

echo "📁 CHECKING PROJECT STRUCTURE"
echo "───────────────────────────────────────────────────────────────"

# Ensure directory structure
mkdir -p src/tiny_icf scripts data models
echo "  ✓ Directory structure ready"

# Check if source files exist
if [ ! -f "src/tiny_icf/model.py" ]; then
    echo "  ⚠️  Source files missing - need to copy from local"
    echo "  Run: rsync -avz -e 'ssh -i $SSH_KEY -p $SSH_PORT' src/ root@$SSH_HOST:/root/idf-est/src/"
    exit 1
else
    echo "  ✓ Source files present"
fi

# Check if training script exists
if [ ! -f "scripts/train_ephemeral_robust.py" ]; then
    echo "  ⚠️  Training script missing - need to copy from local"
    echo "  Run: scp -i $SSH_KEY -P $SSH_PORT scripts/train_ephemeral_robust.py root@$SSH_HOST:/root/idf-est/scripts/"
    exit 1
else
    echo "  ✓ Training script present"
fi

echo ""

echo "🔧 FIXING PYTHON 3.8 COMPATIBILITY"
echo "───────────────────────────────────────────────────────────────"

# Fix type hints for Python 3.8 compatibility
FIXED=0
for file in src/tiny_icf/*.py; do
    if [ -f "$file" ] && grep -q " | None" "$file" 2>/dev/null; then
        echo "  Fixing $file..."
        # Replace | None with Optional[...]
        sed -i 's/: Dict\[\([^\]]*\)\] | None =/: Optional[Dict[\1]] =/g' "$file"
        sed -i 's/: List\[\([^\]]*\)\] | None =/: Optional[List[\1]] =/g' "$file"
        sed -i 's/: Callable\[\([^\]]*\)\] | None =/: Optional[Callable[\1]] =/g' "$file"
        sed -i 's/: Path | None =/: Optional[Path] =/g' "$file"
        sed -i 's/: str | None =/: Optional[str] =/g' "$file"
        sed -i 's/: int | None =/: Optional[int] =/g' "$file"
        sed -i 's/: float | None =/: Optional[float] =/g' "$file"
        
        # Ensure Optional is imported
        if ! grep -q "from typing import.*Optional" "$file" 2>/dev/null; then
            if grep -q "from typing import" "$file"; then
                sed -i 's/from typing import \([^)]*\)/from typing import \1, Optional/' "$file"
            else
                sed -i '1a from typing import Optional' "$file"
            fi
        fi
        FIXED=$((FIXED + 1))
    fi
done

if [ $FIXED -gt 0 ]; then
    echo "  ✓ Fixed $FIXED file(s) for Python 3.8 compatibility"
else
    echo "  ✓ No compatibility fixes needed"
fi

echo ""

echo "📊 CHECKING DATA"
echo "───────────────────────────────────────────────────────────────"

if [ ! -f "data/word_frequency.csv" ]; then
    echo "  ⚠️  Data file missing: data/word_frequency.csv"
    echo "  Need to copy from local machine"
    exit 1
else
    DATA_SIZE=$(du -h data/word_frequency.csv | cut -f1)
    echo "  ✓ Data file exists ($DATA_SIZE)"
fi

echo ""

echo "✅ STATIC CHECKS"
echo "───────────────────────────────────────────────────────────────"

# Run type checking script if available
if [ -f "scripts/check_types.sh" ]; then
    echo "  Running static type checks..."
    if bash scripts/check_types.sh 2>&1 | tail -20; then
        echo "  ✓ Static checks passed"
    else
        echo "  ⚠️  Some static checks failed (continuing anyway)"
    fi
else
    echo "  ⚠️  check_types.sh not found - skipping static checks"
fi

echo ""

echo "✅ VERIFICATION"
echo "───────────────────────────────────────────────────────────────"

# Test imports
if python3 -c "
import sys
sys.path.insert(0, 'src')
from tiny_icf.model_residual import ResidualICF
from tiny_icf.data import WordICFDataset
from tiny_icf.loss import CombinedLoss
print('  ✓ All imports working')
" 2>&1; then
    echo "  ✓ Code verification passed"
else
    echo "  ✗ Import test failed - check errors above"
    exit 1
fi

echo ""

echo "🎯 SETUP COMPLETE"
echo "───────────────────────────────────────────────────────────────"
echo "  Ready to start training!"
echo ""
echo "  Start training:"
echo "    python3 scripts/train_ephemeral_robust.py \\"
echo "      --data data/word_frequency.csv \\"
echo "      --output-dir models \\"
echo "      --epochs 200 \\"
echo "      --batch-size 256 \\"
echo "      --lr 1e-3 \\"
echo "      --rank-weight 5.0 \\"
echo "      --early-stop-patience 20 \\"
echo "      --checkpoint-interval 1"
echo ""
echo "  Monitor: ./scripts/monitor_ephemeral.sh"
echo ""

ENDSSH

