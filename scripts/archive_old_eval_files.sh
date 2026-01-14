#!/bin/bash
# Archive old eval files after consolidation

ARCHIVE_DIR="archive/$(date +%Y%m%d)/eval_files"
mkdir -p "$ARCHIVE_DIR"

echo "Archiving old eval files..."
echo ""

# Move eval_calibration.py (consolidated into eval.py)
if [ -f "src/tiny_icf/eval_calibration.py" ]; then
    mv "src/tiny_icf/eval_calibration.py" "$ARCHIVE_DIR/"
    echo "✅ Archived eval_calibration.py"
fi

# Move eval_stratified.py (consolidated into eval.py)
if [ -f "src/tiny_icf/eval_stratified.py" ]; then
    mv "src/tiny_icf/eval_stratified.py" "$ARCHIVE_DIR/"
    echo "✅ Archived eval_stratified.py"
fi

echo ""
echo "✅ Archive complete: $ARCHIVE_DIR"
echo ""
echo "Note: Functions are now in src/tiny_icf/eval.py"
echo "      Update any imports if needed"
