#!/bin/bash
# Monitor training progress repeatedly

EXPERIMENT_NAME="${1:-multitask_icf_only}"
MODEL_DIR="models/${EXPERIMENT_NAME}"
LOG_FILE="${MODEL_DIR}/training_progress.jsonl"
CSV_METRICS="${MODEL_DIR}/lightning_logs/version_0/metrics.csv"

echo "Monitoring: ${EXPERIMENT_NAME}"
echo "Model dir: ${MODEL_DIR}"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "Training Monitor: ${EXPERIMENT_NAME}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Check if model dir exists
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "⚠️  Model directory not found: ${MODEL_DIR}"
        echo "   Waiting for training to start..."
        sleep 10
        continue
    fi
    
    # Check Lightning CSV metrics
    if [ -f "${CSV_METRICS}" ]; then
        echo "📊 Latest Metrics (from Lightning CSV):"
        tail -5 "${CSV_METRICS}" | column -t -s, | head -6
        echo ""
    fi
    
    # Check JSONL progress log
    if [ -f "${LOG_FILE}" ]; then
        echo "📈 Latest Progress (from JSONL):"
        tail -3 "${LOG_FILE}" | python3 -m json.tool 2>/dev/null || tail -3 "${LOG_FILE}"
        echo ""
    fi
    
    # Check for best model
    if [ -f "${MODEL_DIR}/model_best.pt" ]; then
        echo "✅ Best model found: $(ls -lh ${MODEL_DIR}/model_best.pt | awk '{print $5, $6, $7, $8}')"
    fi
    
    # Check for checkpoints
    CHECKPOINTS=$(find "${MODEL_DIR}" -name "*.pt" 2>/dev/null | wc -l)
    if [ "${CHECKPOINTS}" -gt 0 ]; then
        echo "💾 Checkpoints: ${CHECKPOINTS} files"
        echo "   Latest: $(ls -t ${MODEL_DIR}/*.pt 2>/dev/null | head -1 | xargs basename)"
    fi
    
    # Check if training is still running
    if pgrep -f "train_flexible_opportunistic" > /dev/null; then
        echo ""
        echo "🟢 Training process is running"
    else
        echo ""
        echo "🔴 Training process not found (may have completed or failed)"
    fi
    
    echo ""
    echo "Refreshing in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done
