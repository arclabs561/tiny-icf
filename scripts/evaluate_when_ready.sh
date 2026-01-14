#!/bin/bash
# Script to evaluate all completed experiments

echo "=================================================================================="
echo "EVALUATING ALL COMPLETED EXPERIMENTS"
echo "=================================================================================="
echo ""

# Run comprehensive evaluation
echo "1. Running comprehensive evaluation..."
uv run scripts/comprehensive_evaluation.py

echo ""
echo "2. Comparing experiment histories..."
uv run scripts/compare_all_experiments.py

echo ""
echo "=================================================================================="
echo "EVALUATION COMPLETE"
echo "=================================================================================="
echo ""
echo "Results saved to:"
echo "  - models/comprehensive_evaluation.json"
echo "  - models/experiment_comparison.json"
echo ""

