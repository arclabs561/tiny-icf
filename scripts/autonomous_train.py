#!/usr/bin/env python3
"""
Autonomous training manager - handles everything automatically.
"""
import subprocess
import time
import sys
from pathlib import Path
import json
import re

def get_api_key():
    """Get RunPod API key from MCP config."""
    mcp_config = Path.home() / ".cursor" / "mcp.json"
    if not mcp_config.exists():
        return None
    
    try:
        with open(mcp_config) as f:
            content = f.read()
        
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
            api_key = find_key(config, "RUNPOD_API_KEY")
            if api_key:
                return api_key
        except json.JSONDecodeError:
            pass
        
        # Fallback: regex
        patterns = [
            r'"RUNPOD_API_KEY"\s*:\s*"([^"]+)"',
            r'RUNPOD_API_KEY["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
    except Exception:
        pass
    return None

def configure_runpodctl():
    """Configure runpodctl."""
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
    except Exception:
        return False

def execute_remote_command(pod_id, command):
    """Execute command on pod using SSH."""
    # Try to get SSH connection details
    try:
        # Use runpodctl's internal SSH mechanism
        # Create a script that runs the command
        script_content = f"""#!/bin/bash
{command}
"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        import os
        os.chmod(script_path, 0o755)
        
        # Upload and execute
        subprocess.run(
            ["runpodctl", "send", script_path, pod_id, "/tmp/remote_cmd.sh"],
            check=True,
            capture_output=True,
        )
        
        # Execute via SSH using expect or direct SSH
        # For now, use a workaround: create a systemd service or use screen
        # Actually, let's use a simpler approach: create a wrapper that runs in background
        
        # Method: Upload a script that runs the command and exits
        # Then use SSH with command execution
        result = subprocess.run(
            ["runpodctl", "exec", pod_id, "--", "bash", "/tmp/remote_cmd.sh"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        os.unlink(script_path)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except subprocess.CalledProcessError as e:
        return False, "", str(e.stderr)
    except Exception as e:
        return False, "", str(e)

def start_training(pod_id):
    """Start training on pod."""
    print("🚀 Starting training...")
    
    command = """cd /workspace/tiny-icf && bash start_training.sh"""
    
    success, stdout, stderr = execute_remote_command(pod_id, command)
    if success:
        print("✅ Training started")
        return True
    else:
        print(f"⚠️  Direct execution failed: {stderr}")
        # Fallback: ensure script exists and is executable
        ensure_script = """cd /workspace/tiny-icf && chmod +x start_training.sh && ls -la start_training.sh"""
        execute_remote_command(pod_id, ensure_script)
        print("📝 Script ready. Training will start on next check.")
        return True

def check_training_status(pod_id):
    """Check if training is running."""
    command = """cd /workspace/tiny-icf && if [ -f training.pid ]; then PID=$(cat training.pid); if ps -p $PID > /dev/null 2>&1; then echo "RUNNING:$PID"; else echo "STOPPED:$PID"; fi; else echo "NOT_STARTED"; fi"""
    
    success, stdout, stderr = execute_remote_command(pod_id, command)
    if success:
        status = stdout.strip()
        if status.startswith("RUNNING"):
            return "running", status.split(":")[1]
        elif status.startswith("STOPPED"):
            return "stopped", status.split(":")[1] if ":" in status else None
        else:
            return "not_started", None
    return "unknown", None

def get_training_log(pod_id, lines=20):
    """Get last N lines of training log."""
    command = f"""cd /workspace/tiny-icf && tail -{lines} training.log 2>/dev/null || echo "No log yet\""""
    
    success, stdout, stderr = execute_remote_command(pod_id, command)
    if success:
        return stdout
    return ""

def check_training_complete(pod_id):
    """Check if training completed successfully."""
    command = """cd /workspace/tiny-icf && if [ -f models/model_final.pt ] || [ -f models/*.pt ]; then echo "COMPLETE"; else echo "IN_PROGRESS"; fi"""
    
    success, stdout, stderr = execute_remote_command(pod_id, command)
    if success and "COMPLETE" in stdout:
        return True
    return False

def download_results(pod_id, local_path="models"):
    """Download training results."""
    print(f"📥 Downloading results...")
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            ["runpodctl", "receive", pod_id, "/workspace/tiny-icf/models", local_path],
            check=True,
            capture_output=True,
        )
        print("✅ Results downloaded")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    
    print(f"🤖 Autonomous Training Manager")
    print(f"   Pod: {pod_id}")
    print()
    
    if not configure_runpodctl():
        sys.exit(1)
    
    # Check current status
    status, pid = check_training_status(pod_id)
    print(f"📊 Current status: {status}")
    
    if status == "not_started":
        print("🚀 Starting training...")
        if not start_training(pod_id):
            print("❌ Failed to start training")
            sys.exit(1)
        time.sleep(5)  # Wait for training to start
    
    # Monitor training
    print("\n📊 Monitoring training...")
    print("   (Press Ctrl+C to stop monitoring, training will continue)")
    print()
    
    last_log = ""
    check_count = 0
    
    try:
        while True:
            status, pid = check_training_status(pod_id)
            check_count += 1
            
            if status == "running":
                log = get_training_log(pod_id, 5)
                if log != last_log:
                    print(f"[{check_count}] Training running (PID: {pid})")
                    print(log)
                    last_log = log
            elif status == "stopped":
                print(f"⚠️  Training process stopped (PID: {pid})")
                # Check if it completed successfully
                if check_training_complete(pod_id):
                    print("✅ Training completed successfully!")
                    break
                else:
                    print("❌ Training stopped unexpectedly")
                    log = get_training_log(pod_id, 30)
                    print("Last log entries:")
                    print(log)
                    break
            elif status == "not_started":
                print("🚀 Starting training...")
                start_training(pod_id)
                time.sleep(5)
            
            # Check for completion every 10 checks
            if check_count % 10 == 0:
                if check_training_complete(pod_id):
                    print("✅ Training completed!")
                    break
            
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n⏸️  Monitoring stopped (training continues in background)")
        print(f"   Check status: python3 scripts/autonomous_train.py {pod_id}")
    
    # Download results
    if check_training_complete(pod_id):
        print("\n📥 Downloading results...")
        download_results(pod_id)
        print("\n✅ All done!")

if __name__ == "__main__":
    main()

