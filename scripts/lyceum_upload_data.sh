#!/bin/bash
# Upload data files to Lyceum VM
# Usage: ./scripts/lyceum_upload_data.sh <vm-ip> [ssh-key-path]

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

echo "📤 Uploading data files to $VM_IP..."

# Upload data directory
if [ -d "$PROJECT_ROOT/data" ]; then
    echo "Uploading data/ directory..."
    scp -i "$SSH_KEY" -r -o StrictHostKeyChecking=no \
        "$PROJECT_ROOT/data/" \
        ubuntu@"$VM_IP":~/idf-est/data/
    echo "✓ Data uploaded"
else
    echo "⚠️  No data/ directory found"
fi

echo "✅ Upload complete!"

