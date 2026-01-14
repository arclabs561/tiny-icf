#!/usr/bin/env -S uv run
"""Run multiple training experiments in batch with different configurations."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

def run_experiment(config: Dict, output_dir: Path, data_path: Path):
    """Run a single training experiment."""
    name = config.get("name", "experiment")
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"{'='*70}\n")
    
    # Create experiment directory
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "-m", "tiny_icf.train_adaptive",
        "--data", str(data_path),
        "--epochs", str(config.get("epochs", 50)),
        "--batch-size", str(config.get("batch_size", 64)),
        "--lr", str(config.get("lr", 1e-3)),
        "--output", str(exp_dir / "model.pt"),
        "--history", str(exp_dir / "history.json"),
        "--eval-interval", str(config.get("eval_interval", 5)),
    ]
    
    if config.get("scheduler"):
        cmd.extend(["--scheduler", config["scheduler"]])
    
    if config.get("early_stop"):
        cmd.extend(["--early-stop", "--early-stop-patience", str(config.get("early_stop_patience", 15))])
    
    # Run experiment
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ {name} completed successfully")
        
        # Save config and output
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        with open(exp_dir / "output.log", "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n" + result.stderr)
        
        return {"name": name, "status": "success", "output_dir": str(exp_dir)}
    except subprocess.CalledProcessError as e:
        print(f"✗ {name} failed: {e}")
        return {"name": name, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run batch training experiments")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"), help="Output directory")
    parser.add_argument("--config", type=Path, help="JSON config file with experiments")
    parser.add_argument("--quick", action="store_true", help="Run quick experiments (fewer epochs)")
    
    args = parser.parse_args()
    
    # Default experiments if no config provided
    if args.config and args.config.exists():
        with open(args.config, "r") as f:
            experiments = json.load(f)
    else:
        # Default experiment configurations
        experiments = [
            {
                "name": "baseline",
                "epochs": 30 if args.quick else 100,
                "batch_size": 64,
                "lr": 1e-3,
                "scheduler": "cosine",
                "early_stop": True,
            },
            {
                "name": "adaptive_scheduler",
                "epochs": 30 if args.quick else 100,
                "batch_size": 64,
                "lr": 1e-3,
                "scheduler": "adaptive",
                "early_stop": True,
            },
            {
                "name": "higher_lr",
                "epochs": 30 if args.quick else 100,
                "batch_size": 64,
                "lr": 2e-3,
                "scheduler": "adaptive",
                "early_stop": True,
            },
            {
                "name": "larger_batch",
                "epochs": 30 if args.quick else 100,
                "batch_size": 128,
                "lr": 1e-3,
                "scheduler": "adaptive",
                "early_stop": True,
            },
        ]
    
    print(f"Running {len(experiments)} experiments")
    print(f"Output directory: {args.output_dir}")
    print(f"Data: {args.data}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    with open(output_dir / "experiments.json", "w") as f:
        json.dump(experiments, f, indent=2)
    
    # Run experiments
    results = []
    for exp_config in experiments:
        result = run_experiment(exp_config, output_dir, args.data)
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("Batch Experiment Summary")
    print(f"{'='*70}\n")
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    
    if successful:
        print(f"\nSuccessful experiments:")
        for r in successful:
            print(f"  ✓ {r['name']}: {r.get('output_dir', 'N/A')}")
    
    if failed:
        print(f"\nFailed experiments:")
        for r in failed:
            print(f"  ✗ {r['name']}: {r.get('error', 'Unknown error')}")
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "results": results,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

