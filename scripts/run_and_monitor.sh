#!/bin/bash
# Run experiment and monitor it

EXPERIMENT="${1:-multitask_icf_only}"
MAX_EXPERIMENTS="${2:-1}"

echo "🚀 Starting experiment: ${EXPERIMENT}"
echo ""

# Start training in background
cd ../trainctl/training/scripts
uv run python train_flexible_opportunistic.py \
    --data ../../idf-est/data/word_frequency.csv \
    --experiments "${EXPERIMENT}" \
    --max_experiments "${MAX_EXPERIMENTS}" \
    --train_split 0.8 \
    2>&1 | tee "../../idf-est/models/${EXPERIMENT}/training.log" &

TRAIN_PID=$!
echo "Training PID: ${TRAIN_PID}"
echo ""

# Start monitoring in another terminal/process
cd ../../idf-est
./scripts/monitor_training.sh "${EXPERIMENT}" &
MONITOR_PID=$!

echo "Monitor PID: ${MONITOR_PID}"
echo ""
echo "✅ Training and monitoring started"
echo "   Training log: models/${EXPERIMENT}/training.log"
echo "   To stop: kill ${TRAIN_PID} ${MONITOR_PID}"
echo ""

# Wait for training to complete
wait ${TRAIN_PID}
TRAIN_EXIT=$?

# Stop monitor
kill ${MONITOR_PID} 2>/dev/null

echo ""
if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo "✅ Training completed successfully"
else
    echo "⚠️  Training exited with code: ${TRAIN_EXIT}"
fi
