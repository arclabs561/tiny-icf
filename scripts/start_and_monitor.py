#!/usr/bin/env -S uv run
"""
Simple, direct training starter and monitor.
Uses SSH directly for reliable command execution.
"""
import subprocess
import time
import sys
import json
import re
from pathlib import Path

def get_api_key():
    """Get RunPod API key."""
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
            return find_key(config, "RUNPOD_API_KEY")
        except:
            match = re.search(r'"RUNPOD_API_KEY"\s*:\s*"([^"]+)"', content)
            if match:
                return match.group(1)
    except:
        pass
    return None

def ssh_execute(pod_id, command, timeout=30):
    """Execute command via SSH using runpodctl's SSH connection."""
    # Get SSH connection string
    result = subprocess.run(
        ["runpodctl", "ssh", "connect", pod_id],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    # Try to extract SSH command from output
    # runpodctl ssh connect usually prints the SSH command
    ssh_line = result.stdout.strip() or result.stderr.strip()
    
    if not ssh_line or not ssh_line.startswith("ssh"):
        # Try alternative: use expect or pexpect to handle interactive SSH
        print(f"⚠️  Could not extract SSH command automatically")
        print(f"   Manual: runpodctl ssh connect {pod_id}")
        print(f"   Then run: {command}")
        return False, "", "Could not get SSH connection"
    
    # Parse SSH command
    import shlex
    try:
        ssh_parts = shlex.split(ssh_line)
        # Add the command to execute
        ssh_parts.append(command)
        
        # Execute via SSH
        result = subprocess.run(
            ssh_parts,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def start_training_direct(pod_id):
    """Start training using direct method."""
    print("🚀 Starting training...")
    
    # Create a simple script that starts training
    script = """cd /workspace/tiny-icf && nohup bash start_training.sh > /tmp/train_start.log 2>&1 & echo "Started: $!" """
    
    # Try multiple methods
    methods = [
        # Method 1: Use runpodctl exec python (if it supports subprocess)
        lambda: subprocess.run(
            ["runpodctl", "exec", "python", pod_id, "--", "-c", 
             f"import subprocess; subprocess.Popen(['bash', '/workspace/tiny-icf/start_training.sh'])"],
            capture_output=True,
            timeout=10,
        ),
        # Method 2: Upload and execute via SSH
        lambda: ssh_execute(pod_id, "cd /workspace/tiny-icf && bash start_training.sh &"),
    ]
    
    for i, method in enumerate(methods, 1):
        try:
            print(f"   Trying method {i}...")
            result = method()
            if isinstance(result, tuple):
                success, stdout, stderr = result
                if success:
                    print(f"✅ Training started via method {i}")
                    print(f"   Output: {stdout[:200]}")
                    return True
            elif hasattr(result, 'returncode') and result.returncode == 0:
                print(f"✅ Training started via method {i}")
                return True
        except Exception as e:
            print(f"   Method {i} failed: {e}")
            continue
    
    print("⚠️  Could not start automatically. Using fallback...")
    return False

def check_status_simple(pod_id):
    """Simple status check using file operations."""
    # Check if PID file exists and process is running
    check_cmd = """cd /workspace/tiny-icf && if [ -f training.pid ]; then PID=$(cat training.pid); if ps -p $PID >/dev/null 2>&1; then echo "RUNNING:$PID"; else echo "STOPPED"; fi; elif [ -f training.log ]; then echo "STARTING"; else echo "NOT_STARTED"; fi"""
    
    success, stdout, stderr = ssh_execute(pod_id, check_cmd, timeout=10)
    if success:
        return stdout.strip()
    return "UNKNOWN"

def get_log_tail(pod_id, lines=10):
    """Get tail of training log."""
    cmd = f"cd /workspace/tiny-icf && tail -{lines} training.log 2>/dev/null || echo 'No log yet'"
    success, stdout, stderr = ssh_execute(pod_id, cmd, timeout=10)
    if success:
        return stdout
    return ""

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    
    print("🤖 Direct Training Manager")
    print(f"   Pod: {pod_id}\n")
    
    # Configure
    api_key = get_api_key()
    if api_key:
        subprocess.run(["runpodctl", "config", "--apiKey", api_key], 
                      capture_output=True, check=False)
    
    # Check current status
    print("📊 Checking current status...")
    status = check_status_simple(pod_id)
    print(f"   Status: {status}\n")
    
    if "NOT_STARTED" in status or "UNKNOWN" in status:
        print("🚀 Starting training...")
        if start_training_direct(pod_id):
            time.sleep(3)
        else:
            # Fallback: create script and provide instructions
            print("\n📝 Fallback: Training script is ready on pod")
            print(f"   To start manually:")
            print(f"   runpodctl ssh connect {pod_id}")
            print(f"   Then: cd /workspace/tiny-icf && bash start_training.sh")
            print("\n   Or wait 10 seconds and I'll check again...")
            time.sleep(10)
    
    # Monitor with clear progress
    print("\n📊 Monitoring training (showing progress every 30s)...")
    print("   Press Ctrl+C to stop monitoring (training continues)\n")
    
    last_log_lines = []
    iteration = 0
    
    try:
        while True:
            iteration += 1
            status = check_status_simple(pod_id)
            
            # Get fresh log
            log = get_log_tail(pod_id, 15)
            log_lines = [l for l in log.split('\n') if l.strip()]
            
            # Show new lines
            new_lines = [l for l in log_lines if l not in last_log_lines]
            
            if new_lines or iteration % 2 == 0:  # Show status every other iteration
                print(f"\n[{iteration}] Status: {status}")
                if new_lines:
                    print("   New log entries:")
                    for line in new_lines[-5:]:  # Last 5 new lines
                        print(f"   {line}")
                elif log_lines:
                    print("   Latest:")
                    for line in log_lines[-3:]:  # Last 3 lines
                        print(f"   {line}")
            
            last_log_lines = log_lines[-10:]  # Keep last 10 for comparison
            
            # Check for completion
            if "STOPPED" in status:
                print("\n⚠️  Training process stopped")
                final_log = get_log_tail(pod_id, 20)
                print("Final log:")
                print(final_log)
                break
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n⏸️  Monitoring stopped (check {iteration} iterations)")
        print(f"   Training continues. Check status:")
        print(f"   python3 scripts/start_and_monitor.py {pod_id}")

if __name__ == "__main__":
    main()

