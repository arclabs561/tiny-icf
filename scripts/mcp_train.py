#!/usr/bin/env python3
"""
Autonomous training using MCP RunPod tools - fast and direct.
"""
import sys
import time
import subprocess
from pathlib import Path

# Note: This script demonstrates the approach, but MCP tools are called via the MCP protocol
# In practice, we'll use runpodctl for file operations and check status via MCP

def check_pod_status_mcp(pod_id):
    """Check pod status - would use MCP get-pod tool."""
    # In actual implementation, this would call MCP tool
    # For now, use runpodctl to get status
    try:
        result = subprocess.run(
            ["runpodctl", "get", "pod", pod_id],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "RUNNING" in result.stdout:
            return "RUNNING"
        return "UNKNOWN"
    except:
        return "UNKNOWN"

def execute_via_script(pod_id):
    """Execute training by uploading and running a script."""
    print(f"🚀 Starting training on {pod_id}...")
    
    # Create a simple runner script
    runner = """#!/bin/bash
cd /workspace/tiny-icf
if [ ! -f start_training.sh ]; then
    echo "Creating start_training.sh..."
    cat > start_training.sh << 'EOF'
#!/bin/bash
cd /workspace/tiny-icf
nohup uv run scripts/train_batch.py \\
    --data data/multilingual/multilingual_combined.csv \\
    --typo-corpus data/typos/github_typos.csv \\
    --output-dir /workspace/tiny-icf/models \\
    --epochs 50 \\
    --batch-size 256 \\
    --devices 1 \\
    >> training.log 2>&1 &
echo $! > training.pid
echo "Training PID: $(cat training.pid)"
EOF
    chmod +x start_training.sh
fi

# Start training
bash start_training.sh
"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(runner)
        script_path = f.name
    
    import os
    os.chmod(script_path, 0o755)
    
    try:
        # Upload
        print("   Uploading runner script...")
        subprocess.run(
            ["runpodctl", "send", script_path, pod_id, "/tmp/run_train.sh"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        
        # Execute via SSH (we'll need to get SSH connection)
        # For now, create a systemd service or use screen/tmux
        # Actually, let's use a simpler approach: create a wrapper that auto-runs
        
        print("   ✅ Script uploaded")
        print(f"   To execute: runpodctl ssh connect {pod_id}")
        print(f"   Then: bash /tmp/run_train.sh")
        
        # Try to execute via Python on pod
        python_code = """import subprocess, os
os.chdir('/workspace/tiny-icf')
if not os.path.exists('start_training.sh'):
    # Create it
    with open('start_training.sh', 'w') as f:
        f.write('''#!/bin/bash
cd /workspace/tiny-icf
nohup uv run scripts/train_batch.py --data data/multilingual/multilingual_combined.csv --typo-corpus data/typos/github_typos.csv --output-dir /workspace/tiny-icf/models --epochs 50 --batch-size 256 --devices 1 >> training.log 2>&1 &
echo $! > training.pid
''')
    os.chmod('start_training.sh', 0o755)

proc = subprocess.Popen(['bash', 'start_training.sh'], 
                       stdout=open('training_start.log', 'a'),
                       stderr=subprocess.STDOUT)
print(f'Started: {proc.pid}')
"""
        
        py_path = script_path.replace('.sh', '.py')
        with open(py_path, 'w') as f:
            f.write(python_code)
        
        subprocess.run(
            ["runpodctl", "send", py_path, pod_id, "/tmp/start_train.py"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        
        # Try executing
        print("   Executing via Python...")
        result = subprocess.run(
            ["runpodctl", "exec", "python", "--pod_id", pod_id, "/tmp/start_train.py"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        
        if result.returncode == 0:
            print(f"   ✅ {result.stdout.strip()}")
            return True
        else:
            print(f"   ⚠️  Python exec had issues, but script is ready")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⚠️  Timeout, but script uploaded")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        import os
        for p in [script_path, py_path]:
            try:
                os.unlink(p)
            except:
                pass

def quick_check(pod_id):
    """Quick status check."""
    python_code = """import os
pid_file = '/workspace/tiny-icf/training.pid'
if os.path.exists(pid_file):
    pid = open(pid_file).read().strip()
    if os.path.exists(f'/proc/{pid}'):
        print(f'RUNNING:{pid}')
    else:
        print('STOPPED')
elif os.path.exists('/workspace/tiny-icf/training.log'):
    print('STARTING')
else:
    print('NOT_STARTED')
"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(python_code)
        py_path = f.name
    
    try:
        subprocess.run(
            ["runpodctl", "send", py_path, pod_id, "/tmp/check.py"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        result = subprocess.run(
            ["runpodctl", "exec", "python", "--pod_id", pod_id, "/tmp/check.py"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"
    except:
        return "UNKNOWN"
    finally:
        import os
        os.unlink(py_path)

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    
    print(f"🤖 MCP-Based Training Manager")
    print(f"   Pod: {pod_id}\n")
    
    # Check pod status via MCP (simulated)
    status = check_pod_status_mcp(pod_id)
    print(f"📊 Pod status: {status}\n")
    
    if status != "RUNNING":
        print("❌ Pod is not running")
        return
    
    # Check training status
    train_status = quick_check(pod_id)
    print(f"📊 Training status: {train_status}\n")
    
    if "NOT_STARTED" in train_status or "UNKNOWN" in train_status:
        if execute_via_script(pod_id):
            time.sleep(3)
        else:
            print("\n⚠️  Auto-start had issues, but scripts are ready")
            print(f"   Manual: runpodctl ssh connect {pod_id}")
            print(f"   Then: bash /tmp/run_train.sh")
    
    # Quick monitor
    print("\n📊 Quick monitor (10s intervals, Ctrl+C to stop):\n")
    try:
        for i in range(1, 100):
            status = quick_check(pod_id)
            print(f"[{i}] {status}")
            
            if "STOPPED" in status:
                print("\n⚠️  Training stopped")
                break
            
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"\n⏸️  Stopped after {i} checks")

if __name__ == "__main__":
    main()

