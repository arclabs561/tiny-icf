#!/usr/bin/env -S uv run
"""
Training optimization utilities:
- Find best GPU for cost/performance
- Optimize batch size based on GPU memory
- Generate optimized training commands
"""

import subprocess
import sys
from pathlib import Path


def get_available_gpus():
    """Get available GPU types and prices."""
    try:
        result = subprocess.run(
            ["runpodctl", "get", "cloud"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def recommend_gpu(budget_per_hour=0.25, min_memory_gb=24):
    """Recommend best GPU based on budget and requirements."""
    gpus = get_available_gpus()
    if not gpus:
        return None
    
    recommendations = []
    for line in gpus.split("\n")[1:]:  # Skip header
        if not line.strip() or "GPU TYPE" in line:
            continue
        
        parts = line.split()
        if len(parts) < 5:
            continue
        
        try:
            gpu_name = " ".join(parts[:-4])
            mem_gb = int(parts[-4])
            spot_price = parts[-2]
            ondemand_price = parts[-1]
            
            if mem_gb < min_memory_gb:
                continue
            
            # Parse prices
            spot = float(spot_price) if spot_price != "Reserved" else float('inf')
            ondemand = float(ondemand_price) if ondemand_price != "Reserved" else float('inf')
            
            if spot <= budget_per_hour or ondemand <= budget_per_hour:
                recommendations.append({
                    "name": gpu_name,
                    "memory_gb": mem_gb,
                    "spot_price": spot,
                    "ondemand_price": ondemand,
                    "value": mem_gb / spot if spot != float('inf') else 0,
                })
        except (ValueError, IndexError):
            continue
    
    # Sort by value (memory per dollar)
    recommendations.sort(key=lambda x: x["value"], reverse=True)
    return recommendations[:5]


def optimize_batch_size(gpu_memory_gb=27, model_params=33000):
    """Recommend optimal batch size based on GPU memory."""
    # Rough estimate: each sample ~100 bytes, model ~130KB
    # With mixed precision: ~50 bytes per sample
    # Account for gradients, optimizer states, etc.
    
    # Conservative estimate: 200 bytes per sample
    bytes_per_sample = 200
    model_overhead_mb = 1  # Model + gradients + optimizer states
    
    available_mb = (gpu_memory_gb * 1024) - (model_overhead_mb * 1024)
    max_batch = int(available_mb * 1024 / bytes_per_sample)
    
    # Use 70% of max for safety
    recommended = int(max_batch * 0.7)
    
    # Round to nice numbers
    if recommended > 512:
        recommended = 512
    elif recommended > 256:
        recommended = 256
    elif recommended > 128:
        recommended = 128
    else:
        recommended = 64
    
    return recommended


def main():
    print("🔍 GPU Recommendations for IDF Training")
    print("=" * 60)
    
    # Current: RTX 3090, 27GB, $0.110/hr spot
    print("\n📊 Recommended GPUs (budget: $0.25/hr):")
    recs = recommend_gpu(budget_per_hour=0.25, min_memory_gb=24)
    if recs:
        for i, gpu in enumerate(recs, 1):
            print(f"\n{i}. {gpu['name']}")
            print(f"   Memory: {gpu['memory_gb']} GB")
            print(f"   Spot: ${gpu['spot_price']:.3f}/hr")
            print(f"   On-demand: ${gpu['ondemand_price']:.3f}/hr")
            print(f"   Value: {gpu['value']:.1f} GB/$")
            batch = optimize_batch_size(gpu['memory_gb'])
            print(f"   Recommended batch size: {batch}")
    else:
        print("   No GPUs found matching criteria")
    
    print("\n💡 Optimization Tips:")
    print("   - Use mixed precision (AMP) for 2x speed")
    print("   - Increase batch size for better GPU utilization")
    print("   - Use DataLoader with num_workers=4 for parallel loading")
    print("   - Enable pin_memory for faster CPU->GPU transfer")
    print("   - Use learning rate scheduling (CosineAnnealing)")
    print("   - Enable early stopping to save time")


if __name__ == "__main__":
    main()

