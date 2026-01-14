#!/bin/bash
# Plan Iteration 5 experiments based on Iter4 findings
# 
# Key Findings from Iter4:
# - Best: iter4_residual_distillation (0.187) - ModernBERT distillation works!
# - Second: iter4_residual_adaptive_reg (0.178) - Adaptive regularization works!
# - Most failed (0.0316) - embedding LR 0.1× was too low (fixed to 0.3×)
#
# Iter5 Strategy:
# 1. Combine distillation + adaptive_reg (best of both)
# 2. Test with corrected embedding LR (0.3× instead of 0.1×)
# 3. Fine-tune distillation temperature and alpha
# 4. Test longer training with best configs
# 5. Explore other successful combinations

echo "📋 Iteration 5 Experiment Plan"
echo "=============================="
echo ""
echo "Based on Iter4 Results:"
echo "  ✅ Best: iter4_residual_distillation (0.187)"
echo "  ✅ Second: iter4_residual_adaptive_reg (0.178)"
echo "  ⚠️  Most failed due to embedding LR 0.1× (fixed to 0.3×)"
echo ""
echo "Iter5 Experiments:"
echo "  1. iter5_distillation_adaptive_reg (combine best two)"
echo "  2. iter5_distillation_fixed_lr (with 0.3× embedding LR)"
echo "  3. iter5_distillation_temp_tune (temperature 2.0, 3.0, 4.0)"
echo "  4. iter5_distillation_alpha_tune (alpha 0.3, 0.5, 0.7)"
echo "  5. iter5_adaptive_reg_longer (300 epochs)"
echo "  6. iter5_distillation_longer (300 epochs)"
echo "  7. iter5_distillation_adaptive_longer (300 epochs, best combo)"
echo "  8. iter5_distillation_feature_align (with feature alignment)"
echo "  9. iter5_distillation_hierarchical (hierarchical feature alignment)"
echo "  10. iter5_baseline_fixed_lr (baseline with 0.3× embedding LR)"
echo ""
echo "Key Improvements:"
echo "  - Embedding LR: 0.1× → 0.3× (fixed)"
echo "  - Adaptive regularization: More robust (MAD-based)"
echo "  - Distillation: Temperature and alpha tuning"
echo "  - Longer training: 300 epochs for best configs"
echo ""
echo "Ready to launch when approved!"

