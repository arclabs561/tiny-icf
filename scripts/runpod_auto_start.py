#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""
Auto-start training on RunPod pod using entrypoint script.
This is the most reliable non-interactive method - training starts automatically.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def get_api_key():
    """Get RunPod API key from MCP config."""
    mcp_config = Path.home() / ".cursor" / "mcp.json"
    if not mcp_config.exists():
        return None
    
    try:
        content = mcp_config.read_text()
        try:
            config = json.loads(content)
            def find_key(obj, key):
                if isinstance(obj, dict):
                    if key in obj:
                        return obj[key]
                    for v in obj.values():
                        result = find_key(v, key)
                        if result:
                            return result
                elif isinstance(obj, list):
                    for item in obj:
                        result = find_key(item, key)
                        if result:
                            return result
                return None
            return find_key(config, "RUNPOD_API_KEY")
        except json.JSONDecodeError:
            pass
        
        match = re.search(r'"RUNPOD_API_KEY"\s*:\s*"([^"]+)"', content)
        return match.group(1) if match else None
    except Exception:
        return None


def configure_runpodctl():
    """Configure runpodctl with API key."""
    api_key = get_api_key()
    if not api_key:
        print("❌ Could not find RUNPOD_API_KEY")
        return False
    
    try:
        subprocess.run(
            ["runpodctl", "config", "--apiKey", api_key],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def create_auto_start_script(pod_id: str) -> bool:
    """
    Create an auto-start script on the pod that runs training.
    This script can be triggered manually or via systemd/cron.
    """
    print(f"📝 Creating auto-start script on pod {pod_id}...")
    
    # Create a comprehensive startup script
    startup_script = """#!/bin/bash
# Auto-start training script for RunPod
# This script runs training non-interactively and exits cleanly

set -e

WORK_DIR="/workspace/tiny-icf"
LOG_FILE="$WORK_DIR/training.log"
PID_FILE="$WORK_DIR/training.pid"

cd "$WORK_DIR"

# Check if training is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Training already running (PID: $PID)"
        exit 0
    fi
fi

# Check if training already completed
if [ -f "$WORK_DIR/models/model_final.pt" ]; then
    echo "Training already completed (model_final.pt exists)"
    exit 0
fi

# Start training
echo "Starting training at $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

nohup uv run scripts/train_batch.py \\
    --data data/multilingual/multilingual_combined.csv \\
    --typo-corpus data/typos/github_typos.csv \\
    --output-dir "$WORK_DIR/models" \\
    --epochs 50 \\
    --batch-size 256 \\
    --devices 1 \\
    --precision 16-mixed \\
    >> "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "Training started with PID: $TRAIN_PID" | tee -a "$LOG_FILE"
echo "PID: $TRAIN_PID"
"""
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(startup_script)
        script_path = f.name
    
    try:
        import os
        os.chmod(script_path, 0o755)
        
        # Upload script
        subprocess.run(
            ["runpodctl", "send", script_path, pod_id, "/workspace/tiny-icf/auto_start_training.sh"],
            check=True,
            capture_output=True,
        )
        
        # Make it executable on pod
        subprocess.run(
            ["runpodctl", "exec", pod_id, "--", "chmod", "+x", "/workspace/tiny-icf/auto_start_training.sh"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        
        print("✅ Auto-start script created")
        print("   Location: /workspace/tiny-icf/auto_start_training.sh")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create script: {e}")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  Timeout, but script may have been uploaded")
        return False
    finally:
        import os
        try:
            os.unlink(script_path)
        except:
            pass


def trigger_auto_start(pod_id: str) -> bool:
    """Trigger the auto-start script using runpodctl exec."""
    print(f"🚀 Triggering auto-start on pod {pod_id}...")
    
    # Use runpodctl exec with a simple command that runs the script
    # This is more reliable than complex commands
    command = "bash /workspace/tiny-icf/auto_start_training.sh"
    
    try:
        result = subprocess.run(
            ["runpodctl", "exec", pod_id, "--", "bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            print("✅ Auto-start triggered successfully")
            if result.stdout:
                print(f"   {result.stdout.strip()}")
            return True
        else:
            print(f"⚠️  Auto-start had issues (exit code {result.returncode})")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            # Still return True - script may have started in background
            return True
            
    except subprocess.TimeoutExpired:
        print("⚠️  Timeout (script may be running in background)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to trigger: {e}")
        return False


def check_training_status(pod_id: str) -> dict:
    """Check training status without SSH."""
    status = {"running": False, "complete": False, "pid": None, "log_tail": ""}
    
    # Check if process is running
    try:
        result = subprocess.run(
            ["runpodctl", "exec", pod_id, "--", "bash", "-c", 
             "if [ -f /workspace/tiny-icf/training.pid ]; then "
             "PID=$(cat /workspace/tiny-icf/training.pid); "
             "if ps -p $PID > /dev/null 2>&1; then echo RUNNING:$PID; else echo STOPPED; fi; "
             "else echo NOT_STARTED; fi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if output.startswith("RUNNING:"):
                status["running"] = True
                status["pid"] = output.split(":")[1]
            elif output == "STOPPED":
                status["running"] = False
    except:
        pass
    
    # Check for completion
    try:
        result = subprocess.run(
            ["runpodctl", "exec", pod_id, "--", "test", "-f", "/workspace/tiny-icf/models/model_final.pt"],
            capture_output=True,
        )
        status["complete"] = result.returncode == 0
    except:
        pass
    
    # Get log tail
    try:
        result = subprocess.run(
            ["runpodctl", "exec", pod_id, "--", "tail", "-n", "5", "/workspace/tiny-icf/training.log"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            status["log_tail"] = result.stdout.strip()
    except:
        pass
    
    return status


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-start training on RunPod pod")
    parser.add_argument("pod_id", help="Pod ID")
    parser.add_argument("--create-only", action="store_true", help="Only create script, don't trigger")
    parser.add_argument("--status", action="store_true", help="Check training status")
    
    args = parser.parse_args()
    
    if not configure_runpodctl():
        sys.exit(1)
    
    if args.status:
        status = check_training_status(args.pod_id)
        print(f"📊 Training Status:")
        print(f"   Running: {status['running']}")
        print(f"   Complete: {status['complete']}")
        if status['pid']:
            print(f"   PID: {status['pid']}")
        if status['log_tail']:
            print(f"   Last log: {status['log_tail']}")
        return
    
    # Create auto-start script
    if not create_auto_start_script(args.pod_id):
        sys.exit(1)
    
    if args.create_only:
        print(f"\n✅ Script created. To start training:")
        print(f"   runpodctl exec {args.pod_id} -- bash /workspace/tiny-icf/auto_start_training.sh")
        return
    
    # Trigger auto-start
    if trigger_auto_start(args.pod_id):
        print(f"\n📊 Check status:")
        print(f"   uv run scripts/runpod_auto_start.py {args.pod_id} --status")
        print(f"\n📋 Monitor logs:")
        print(f"   runpodctl exec {args.pod_id} -- tail -f /workspace/tiny-icf/training.log")


if __name__ == "__main__":
    main()

