#!/bin/bash
# RunPod ablation study script - synced and executed remotely
# This version runs in background (oneshot)

set -e

cd /root/idf-est
export PATH="$HOME/.cargo/bin:$PATH"

# Parse arguments
ARGS="${@}"

# Use the oneshot version that runs in background
bash scripts/runpod_ablation_oneshot.sh ${ARGS}

