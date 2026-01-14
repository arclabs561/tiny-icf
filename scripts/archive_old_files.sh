#!/bin/bash
# Archive old files after consolidation

ARCHIVE_DIR="archive/$(date +%Y%m%d)/consolidated"
mkdir -p "$ARCHIVE_DIR"

echo "Archiving old files after consolidation..."
echo ""

# Archive old loss files (after consolidation into loss.py)
if [ -f "src/tiny_icf/loss_research.py" ]; then
    mv "src/tiny_icf/loss_research.py" "$ARCHIVE_DIR/" 2>/dev/null && echo "✅ Archived loss_research.py"
fi

if [ -f "src/tiny_icf/loss_listwise.py" ]; then
    mv "src/tiny_icf/loss_listwise.py" "$ARCHIVE_DIR/" 2>/dev/null && echo "✅ Archived loss_listwise.py"
fi

if [ -f "src/tiny_icf/loss_multi.py" ]; then
    mv "src/tiny_icf/loss_multi.py" "$ARCHIVE_DIR/" 2>/dev/null && echo "✅ Archived loss_multi.py"
fi

if [ -f "src/tiny_icf/loss_diffsort.py" ]; then
    mv "src/tiny_icf/loss_diffsort.py" "$ARCHIVE_DIR/" 2>/dev/null && echo "✅ Archived loss_diffsort.py"
fi

# Archive old train files from src/ (use scripts/ versions)
for train_file in train.py train_curriculum.py train_cv.py train_lightning.py train_multi_loss.py train_optimized.py train_with_eval.py; do
    if [ -f "src/tiny_icf/$train_file" ]; then
        mv "src/tiny_icf/$train_file" "$ARCHIVE_DIR/" 2>/dev/null && echo "✅ Archived $train_file"
    fi
done

# Archive old predict files (after consolidation)
if [ -f "src/tiny_icf/predict_enhanced.py" ]; then
    mv "src/tiny_icf/predict_enhanced.py" "$ARCHIVE_DIR/" 2>/dev/null && echo "✅ Archived predict_enhanced.py"
fi

if [ -f "src/tiny_icf/predict_advanced.py" ]; then
    mv "src/tiny_icf/predict_advanced.py" "$ARCHIVE_DIR/" 2>/dev/null && echo "✅ Archived predict_advanced.py"
fi

echo ""
echo "✅ Archive complete: $ARCHIVE_DIR"
echo ""
echo "Note: Functions are now consolidated into:"
echo "  - loss.py (all losses)"
echo "  - predict.py or predict_consolidated.py (all prediction features)"
echo "  - scripts/train_*.py (all training scripts)"
