#!/bin/bash
# Pre-deployment checks - run before deploying to remote
# Catches errors locally before they reach the remote server

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  PRE-DEPLOYMENT CHECKS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Run type checks
if [ -f "scripts/check_types.sh" ]; then
    bash scripts/check_types.sh
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Type checks failed - fix errors before deploying"
        exit 1
    fi
else
    echo "⚠️  check_types.sh not found"
fi

echo ""
echo "✅ All pre-deployment checks passed!"
echo "   Ready to deploy to remote server"

