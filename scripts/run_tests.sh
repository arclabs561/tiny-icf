#!/bin/bash
# Run all tests with uv

set -e

echo "=========================================="
echo "Running tiny-icf Test Suite"
echo "=========================================="
echo ""

# Run pytest with uv
uv run pytest tests/ -v --cov=src/tiny_icf --cov-report=term-missing

echo ""
echo "✅ All tests complete!"

