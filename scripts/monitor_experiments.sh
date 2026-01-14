#!/bin/bash
# Monitor active training experiments

echo "=================================================================================="
echo "EXPERIMENT MONITORING"
echo "=================================================================================="
echo ""

# Check if processes are running
echo "1. Checking active processes..."
PROCESSES=$(ps aux | grep -E "train_(residual|aggressive)" | grep -v grep | wc -l | tr -d ' ')
echo "   Active training processes: $PROCESSES"
echo ""

# Residual model status
if [ -f "training_residual.log" ]; then
    echo "2. ResidualICF Model:"
    echo "   Latest progress:"
    tail -5 training_residual.log | grep -E "Epoch|Train|Val|Spearman|MAE|Best|Patience" | tail -3
    if [ -f "models/model_residual.pt" ]; then
        SIZE=$(ls -lh models/model_residual.pt | awk '{print $5}')
        echo "   Model saved: models/model_residual.pt ($SIZE)"
    fi
else
    echo "2. ResidualICF Model: No log file found"
fi
echo ""

# Aggressive regularization status
if [ -f "training_aggressive_reg.log" ]; then
    echo "3. Aggressive Regularization Model:"
    echo "   Latest progress:"
    tail -5 training_aggressive_reg.log | grep -E "Epoch|Train|Val|Spearman|MAE|Best|Patience" | tail -3
    if [ -f "models/model_aggressive_reg.pt" ]; then
        SIZE=$(ls -lh models/model_aggressive_reg.pt | awk '{print $5}')
        echo "   Model saved: models/model_aggressive_reg.pt ($SIZE)"
    fi
else
    echo "3. Aggressive Regularization Model: No log file found"
fi
echo ""

# Check for completion
echo "4. Completion status:"
if grep -q "Training Complete\|Early stopping" training_residual.log 2>/dev/null; then
    echo "   ✓ ResidualICF: COMPLETE"
else
    echo "   ⏳ ResidualICF: Running"
fi

if grep -q "Training Complete\|Early stopping" training_aggressive_reg.log 2>/dev/null; then
    echo "   ✓ Aggressive Reg: COMPLETE"
else
    echo "   ⏳ Aggressive Reg: Running"
fi
echo ""

echo "=================================================================================="

