#!/bin/bash
# Continuous monitoring for all running experiments

INTERVAL="${1:-60}"  # Check every N seconds (default: 60)
MAX_ITERATIONS="${2:-0}"  # 0 = infinite

echo "📊 Continuous Experiment Monitor"
echo "================================="
echo "Interval: ${INTERVAL}s"
echo "Max iterations: ${MAX_ITERATIONS:-infinite}"
echo ""

iteration=0
while [ $MAX_ITERATIONS -eq 0 ] || [ $iteration -lt $MAX_ITERATIONS ]; do
    iteration=$((iteration + 1))
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iteration $iteration"
    echo "----------------------------------------"
    
    # Check loss ablation experiments
    if [ -d "models/loss_ablation_balanced_hybrid" ]; then
        echo "📊 Loss Ablation Status:"
        for exp_dir in models/loss_ablation_*/; do
            if [ -d "$exp_dir" ]; then
                exp_name=$(basename "$exp_dir")
                metrics_file="$exp_dir/lightning_logs/version_0/metrics.csv"
                if [ -f "$metrics_file" ]; then
                    epoch=$(tail -1 "$metrics_file" 2>/dev/null | cut -d',' -f1 | grep -E '^[0-9]+$' || echo "0")
                    spearman=$(tail -1 "$metrics_file" 2>/dev/null | awk -F',' '{for(i=1;i<=NF;i++) if($i ~ /val_spearman_corr/) {print $(i+1); exit}}' || echo "N/A")
                    if [ "$spearman" != "N/A" ] && [ -n "$spearman" ]; then
                        printf "   %-35s Epoch: %3s | Spearman: %7s\n" "$exp_name" "$epoch" "$spearman"
                    fi
                fi
            fi
        done
    fi
    
    # Check iteration 3 experiments
    if [ -d "models/iter3_finetune_8x_06x" ]; then
        echo ""
        echo "📊 Iteration 3 Status:"
        for exp_dir in models/iter3_*/; do
            if [ -d "$exp_dir" ]; then
                exp_name=$(basename "$exp_dir")
                metrics_file="$exp_dir/lightning_logs/version_0/metrics.csv"
                if [ -f "$metrics_file" ]; then
                    epoch=$(tail -1 "$metrics_file" 2>/dev/null | cut -d',' -f1 | grep -E '^[0-9]+$' || echo "0")
                    spearman=$(tail -1 "$metrics_file" 2>/dev/null | awk -F',' '{for(i=1;i<=NF;i++) if($i ~ /val_spearman_corr/) {print $(i+1); exit}}' || echo "N/A")
                    if [ "$spearman" != "N/A" ] && [ -n "$spearman" ]; then
                        printf "   %-35s Epoch: %3s | Spearman: %7s\n" "$exp_name" "$epoch" "$spearman"
                    fi
                fi
            fi
        done
    fi
    
    # Check ModernBERT distillation
    if [ -d "models/distillation_modernbert" ]; then
        echo ""
        echo "📊 ModernBERT Distillation:"
        metrics_file="models/distillation_modernbert/lightning_logs/version_0/metrics.csv"
        if [ -f "$metrics_file" ]; then
            epoch=$(tail -1 "$metrics_file" 2>/dev/null | cut -d',' -f1 | grep -E '^[0-9]+$' || echo "0")
            spearman=$(tail -1 "$metrics_file" 2>/dev/null | awk -F',' '{for(i=1;i<=NF;i++) if($i ~ /val_spearman_corr/) {print $(i+1); exit}}' || echo "N/A")
            if [ "$spearman" != "N/A" ] && [ -n "$spearman" ]; then
                printf "   %-35s Epoch: %3s | Spearman: %7s\n" "distillation_modernbert" "$epoch" "$spearman"
            fi
        fi
    fi
    
    # Check active processes
    active=$(ps aux | grep -E "train_flexible|iter3|loss_ablation|distillation" | grep -v grep | wc -l | tr -d ' ')
    echo ""
    echo "🔄 Active training processes: $active"
    
    sleep "$INTERVAL"
done

echo ""
echo "✅ Monitoring complete"
