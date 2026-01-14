#!/usr/bin/env -S uv run
"""Monitor training progress and provide real-time updates."""

import argparse
import re
import time
from pathlib import Path
from datetime import datetime


def parse_training_log(log_path: Path) -> dict:
    """Parse training log and extract current status."""
    if not log_path.exists():
        return {"error": "Log file not found"}
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Find latest epoch
    latest_epoch = None
    latest_train_loss = None
    latest_val_loss = None
    best_val_loss = None
    saved_models = []
    
    for line in lines:
        # Epoch pattern
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
        if epoch_match:
            latest_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
        
        # Loss pattern
        loss_match = re.search(r'Train loss: ([\d.]+), Val loss: ([\d.]+)', line)
        if loss_match:
            latest_train_loss = float(loss_match.group(1))
            latest_val_loss = float(loss_match.group(2))
            if best_val_loss is None or latest_val_loss < best_val_loss:
                best_val_loss = latest_val_loss
        
        # Saved model pattern
        saved_match = re.search(r'Saved.*model.*to (.+)', line)
        if saved_match:
            saved_models.append(saved_match.group(1))
    
    return {
        "epoch": latest_epoch,
        "train_loss": latest_train_loss,
        "val_loss": latest_val_loss,
        "best_val_loss": best_val_loss,
        "saved_models": saved_models[-5:] if saved_models else [],  # Last 5
        "log_lines": len(lines),
    }


def format_status(status: dict) -> str:
    """Format status for display."""
    if "error" in status:
        return f"Error: {status['error']}"
    
    lines = []
    lines.append("=" * 70)
    lines.append("Training Progress Monitor")
    lines.append("=" * 70)
    lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    if status["epoch"]:
        lines.append(f"Current Epoch: {status['epoch']}")
        lines.append(f"Train Loss:    {status['train_loss']:.6f}" if status['train_loss'] else "Train Loss:    N/A")
        lines.append(f"Val Loss:      {status['val_loss']:.6f}" if status['val_loss'] else "Val Loss:      N/A")
        lines.append(f"Best Val Loss: {status['best_val_loss']:.6f}" if status['best_val_loss'] else "Best Val Loss: N/A")
        lines.append("")
        
        # Progress bar (if we know total epochs)
        if status.get("total_epochs"):
            progress = status["epoch"] / status["total_epochs"]
            bar_length = 50
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)
            lines.append(f"Progress: [{bar}] {progress:.1%}")
            lines.append("")
    
    if status["saved_models"]:
        lines.append("Recent Model Saves:")
        for model in status["saved_models"]:
            lines.append(f"  - {model}")
        lines.append("")
    
    lines.append(f"Log Lines: {status['log_lines']}")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--log", type=Path, default=Path("training_v3.log"), help="Training log file")
    parser.add_argument("--watch", action="store_true", help="Watch mode (update every N seconds)")
    parser.add_argument("--interval", type=int, default=10, help="Update interval in seconds (watch mode)")
    
    args = parser.parse_args()
    
    if args.watch:
        print("Watching training log (Ctrl+C to stop)...")
        print()
        try:
            while True:
                status = parse_training_log(args.log)
                # Clear screen (ANSI escape)
                print("\033[2J\033[H", end="")
                print(format_status(status))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped monitoring.")
    else:
        status = parse_training_log(args.log)
        print(format_status(status))


if __name__ == "__main__":
    main()

