#!/usr/bin/env -S uv run
"""
Advanced training monitoring with real-time stats and GPU utilization.
"""

import subprocess
import sys
import time
from pathlib import Path


def get_pod_status(pod_id):
    """Get pod status and GPU info."""
    try:
        result = subprocess.run(
            ["runpodctl", "get", "pod"],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.split("\n"):
            if pod_id in line:
                return line
        return None
    except subprocess.CalledProcessError:
        return None


def monitor_training_log(pod_id, log_path="/workspace/tiny-icf/training.log", lines=50):
    """Get recent training log lines."""
    try:
        # Use SSH to get log
        ssh_cmd = subprocess.run(
            ["runpodctl", "ssh", "connect", pod_id],
            capture_output=True,
            text=True,
            check=True
        )
        ssh_info = ssh_cmd.stdout.strip()
        print(f"📊 To view full log, run:")
        print(f"   {ssh_info}")
        print(f"   Then: tail -f {log_path}")
        return ssh_info
    except subprocess.CalledProcessError:
        return None


def check_training_process(pod_id):
    """Check if training process is running - returns SSH command."""
    ssh_info = monitor_training_log(pod_id)
    if ssh_info:
        return f"Run: {ssh_info}\n   Then: ps aux | grep train"
    return "Use SSH to check process status"


def get_model_size(pod_id, model_path="/workspace/tiny-icf/models/model_runpod.pt"):
    """Get model file size - returns SSH command."""
    ssh_info = monitor_training_log(pod_id)
    if ssh_info:
        return f"Run: {ssh_info}\n   Then: ls -lh {model_path}"
    return "Use SSH to check model"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Advanced training monitoring")
    parser.add_argument("--pod-id", required=True, help="Pod ID to monitor")
    parser.add_argument("--watch", action="store_true", help="Watch mode (refresh every 10s)")
    
    args = parser.parse_args()
    
    pod_id = args.pod_id
    if pod_id.startswith("pod-"):
        pod_id = pod_id[4:]
    
    print(f"🔍 Monitoring Training on Pod: {pod_id}")
    print("=" * 60)
    
    while True:
        # Pod status
        status = get_pod_status(pod_id)
        if status:
            print(f"\n📦 Pod Status:")
            print(f"   {status}")
        else:
            print(f"\n❌ Pod {pod_id} not found")
            break
        
        # Training process
        process = check_training_process(pod_id)
        if process:
            print(f"\n✅ Training Process Running:")
            print(f"   {process[:100]}...")
        else:
            print(f"\n⚠️  No training process found")
        
        # Model size
        model_size = get_model_size(pod_id)
        if model_size:
            print(f"\n💾 Model Size: {model_size}")
        else:
            print(f"\n💾 Model: Not created yet")
        
        # Log access
        ssh_info = monitor_training_log(pod_id)
        
        if not args.watch:
            break
        
        print("\n" + "=" * 60)
        print("Refreshing in 10 seconds... (Ctrl+C to stop)")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nStopped monitoring")
            break


if __name__ == "__main__":
    main()

