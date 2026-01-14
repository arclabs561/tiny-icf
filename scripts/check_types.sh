#!/bin/bash
# Static type checking and linting before deployment
# Catches errors early in the deployment process

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  STATIC TYPE CHECKING & LINTING"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if we're in the right directory
if [ ! -d "src/tiny_icf" ]; then
    echo "⚠️  Must run from project root"
    exit 1
fi

ERRORS=0

# 1. Check for Python 3.8 incompatible syntax (| None)
echo "🔍 Checking for Python 3.8 incompatible syntax..."
echo "───────────────────────────────────────────────────────────────"
INCOMPATIBLE=$(grep -rn " | None" src/tiny_icf/*.py 2>/dev/null | grep -v "__pycache__" || true)
if [ -n "$INCOMPATIBLE" ]; then
    echo "✗ Found Python 3.8 incompatible syntax (| None):"
    echo "$INCOMPATIBLE" | while IFS= read -r line; do
        echo "  $line"
    done
    ERRORS=$((ERRORS + 1))
else
    echo "✓ No incompatible syntax found"
fi

echo ""

# 2. Check for missing Optional imports
echo "🔍 Checking for missing Optional imports..."
echo "───────────────────────────────────────────────────────────────"
FILES_WITH_OPTIONAL=$(grep -l "Optional\[" src/tiny_icf/*.py 2>/dev/null || true)
MISSING_IMPORT=0
for file in $FILES_WITH_OPTIONAL; do
    if ! grep -q "from typing import.*Optional\|from typing import Optional" "$file" 2>/dev/null; then
        echo "✗ Missing Optional import in: $file"
        MISSING_IMPORT=$((MISSING_IMPORT + 1))
    fi
done
if [ $MISSING_IMPORT -eq 0 ]; then
    echo "✓ All files with Optional have proper imports"
else
    ERRORS=$((ERRORS + MISSING_IMPORT))
fi

echo ""

# 3. Check for Python 3.8 incompatible builtin subscripting (dict[str] vs Dict[str])
echo "🔍 Checking for Python 3.8 incompatible builtin subscripting..."
echo "───────────────────────────────────────────────────────────────"
BUILTIN_SUBSCRIPT=$(grep -rnE "(dict\[|list\[|tuple\[|set\[)" src/tiny_icf/*.py 2>/dev/null | grep -v "__pycache__" | grep -v "Dict\[" | grep -v "List\[" | grep -v "Tuple\[" | grep -v "Set\[" || true)
if [ -n "$BUILTIN_SUBSCRIPT" ]; then
    echo "✗ Found builtin subscripting (use Dict/List/Tuple from typing):"
    echo "$BUILTIN_SUBSCRIPT" | head -10
    ERRORS=$((ERRORS + 1))
else
    echo "✓ No builtin subscripting found"
fi

echo ""

# 4. Python syntax check
echo "🔍 Checking Python syntax..."
echo "───────────────────────────────────────────────────────────────"
SYNTAX_ERRORS=0
for file in src/tiny_icf/*.py; do
    if [ -f "$file" ]; then
        if ! python3 -m py_compile "$file" 2>/dev/null; then
            echo "✗ Syntax error in: $file"
            python3 -m py_compile "$file" 2>&1 || true
            SYNTAX_ERRORS=$((SYNTAX_ERRORS + 1))
        fi
    fi
done
if [ $SYNTAX_ERRORS -eq 0 ]; then
    echo "✓ All Python files have valid syntax"
else
    ERRORS=$((ERRORS + SYNTAX_ERRORS))
fi

echo ""

# 5. Import check (only if torch is available)
echo "🔍 Checking imports..."
echo "───────────────────────────────────────────────────────────────"
if python3 -c "import torch" 2>/dev/null; then
    if python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from tiny_icf.model_residual import ResidualICF
    from tiny_icf.data import WordICFDataset, load_frequency_list
    from tiny_icf.loss import CombinedLoss
    from tiny_icf.training_utils import train_epoch_unified, validate_unified
    from tiny_icf.initialization import init_model_weights
    print('✓ All imports working')
except Exception as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
" 2>&1; then
        echo "✓ Import check passed"
    else
        echo "✗ Import check failed"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "⚠️  torch not installed - skipping import check (expected on local machine)"
fi

echo ""

# 6. Check for common issues
echo "🔍 Checking for common issues..."
echo "───────────────────────────────────────────────────────────────"
COMMON_ISSUES=0

# Check for undefined variables in type hints
UNDEFINED=$(grep -rn "Optional\[" src/tiny_icf/*.py 2>/dev/null | grep -v "from typing import" | while IFS=: read -r file line rest; do
    if ! grep -q "Optional" "$file" | head -20 | grep -q "from typing"; then
        echo "$file:$line"
    fi
done || true)

if [ -n "$UNDEFINED" ]; then
    echo "⚠️  Files using Optional but may be missing import:"
    echo "$UNDEFINED"
    COMMON_ISSUES=$((COMMON_ISSUES + 1))
fi

# Check for torch.Tensor | None (should be Optional[torch.Tensor])
TORCH_UNION=$(grep -rn "torch\.Tensor | None\|torch\.Tensor| None" src/tiny_icf/*.py 2>/dev/null || true)
if [ -n "$TORCH_UNION" ]; then
    echo "✗ Found torch.Tensor | None (should be Optional[torch.Tensor]):"
    echo "$TORCH_UNION"
    COMMON_ISSUES=$((COMMON_ISSUES + 1))
fi

if [ $COMMON_ISSUES -eq 0 ]; then
    echo "✓ No common issues found"
else
    ERRORS=$((ERRORS + COMMON_ISSUES))
fi

echo ""

# Check critical files for training
echo "🔍 Checking critical training files..."
echo "───────────────────────────────────────────────────────────────"
CRITICAL_FILES=(
    "src/tiny_icf/loss.py"
    "src/tiny_icf/data.py"
    "src/tiny_icf/augmentation.py"
    "src/tiny_icf/model_residual.py"
    "src/tiny_icf/training_utils.py"
)

CRITICAL_ERRORS=0
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        if grep -q " | None" "$file" 2>/dev/null; then
            echo "✗ Critical file has incompatible syntax: $file"
            CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
        fi
    fi
done

if [ $CRITICAL_ERRORS -eq 0 ]; then
    echo "✓ All critical training files are compatible"
else
    echo "✗ $CRITICAL_ERRORS critical file(s) have errors"
    ERRORS=$((ERRORS + CRITICAL_ERRORS))
fi

echo ""

# Summary
echo "═══════════════════════════════════════════════════════════════"
if [ $ERRORS -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED"
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
else
    echo "❌ FOUND $ERRORS ERROR(S)"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    if [ $CRITICAL_ERRORS -gt 0 ]; then
        echo "⚠️  CRITICAL: Fix errors in training files before deploying!"
    else
        echo "⚠️  Non-critical errors found (other files). Training should work."
    fi
    echo ""
    echo "Fix the errors above before deploying."
    exit 1
fi

