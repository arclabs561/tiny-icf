#!/bin/bash
# Quick launch and monitor - launches experiments and sets up monitoring

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

DATA_FILE="${1:-data/word_frequency.csv}"

echo "🚀 Quick Launch and Monitor"
echo "=========================="
echo ""

# Launch all 5 initial experiments
echo "1️⃣  Launching experiments..."
./scripts/launch_research_aligned_experiments.sh "$DATA_FILE" --yes

echo ""
echo "2️⃣  Setting up monitoring..."
echo "   Monitoring every 60 seconds..."
echo "   Press Ctrl+C to stop"
echo ""

# Monitor loop
while true; do
    clear
    echo "📊 Research-Aligned Experiments Status"
    echo "======================================"
    echo "$(date)"
    echo ""
    
    ./scripts/monitor_research_aligned_experiments.sh
    
    echo ""
    echo "⏳ Next update in 60 seconds..."
    sleep 60
done

