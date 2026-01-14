#!/bin/bash
# Download training results from Lyceum VM
# Usage: ./scripts/lyceum_download_results.sh <vm-ip> [ssh-key-path]

set -euo pipefail

VM_IP="${1:-}"
SSH_KEY="${2:-$HOME/.ssh/id_ed25519}"

if [ -z "$VM_IP" ]; then
    echo "Usage: $0 <vm-ip> [ssh-key-path]"
    echo "Example: $0 10.0.1.5"
    exit 1
fi

if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found: $SSH_KEY"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/models/lyceum"

mkdir -p "$OUTPUT_DIR"

echo "📥 Downloading results from $VM_IP..."

# Download models
echo "Downloading model files..."
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    ubuntu@"$VM_IP":~/idf-est/models/lyceum/*.pt \
    "$OUTPUT_DIR/" 2>/dev/null || echo "No .pt files found"

scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    ubuntu@"$VM_IP":~/idf-est/models/lyceum/*.ckpt \
    "$OUTPUT_DIR/" 2>/dev/null || echo "No .ckpt files found"

# Download logs
echo "Downloading log files..."
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    ubuntu@"$VM_IP":~/idf-est/training.log \
    "$PROJECT_ROOT/training_lyceum.log" 2>/dev/null || echo "No log file found"

scp -i "$SSH_KEY" -r -o StrictHostKeyChecking=no \
    ubuntu@"$VM_IP":~/idf-est/models/lyceum/logs/ \
    "$OUTPUT_DIR/" 2>/dev/null || echo "No log directory found"

echo "✅ Download complete!"
echo "Results saved to: $OUTPUT_DIR"

