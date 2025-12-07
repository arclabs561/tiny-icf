#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""
Autonomous training manager using MCP RunPod tools and PEP 723.
Run with: uv run scripts/mcp_autonomous_train.py
"""
import sys
import time
import subprocess
import json
import re
from pathlib import Path

# MCP tools would be called via protocol, but we can use runpodctl as proxy
# For direct execution, we'll use uv run on the pod itself

def get_api_key():
    """Get API key from MCP config."""
    mcp = Path.home() / ".cursor" / "mcp.json"
    if not mcp.exists():
        return None
    try:
        content = mcp.read_text()
        match = re.search(r'"RUNPOD_API_KEY"\s*:\s*"([^"]+)"', content)
        return match.group(1) if match else None
    except:
        return None

def start_training_uv(pod_id):
    """Start training using uv run (PEP 723) on pod."""
    print(f"🚀 Starting training with uv on {pod_id}...")
    
    # Create a PEP 723 script that runs training
    # This will be uploaded and executed with `uv run`
    train_script = '''#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "pandas>=2.0.0",
#   "tqdm>=4.65.0",
#   "scipy>=1.10.0",
#   "lightning>=2.0.0",
# ]
# ///
"""Auto-start training script."""
import subprocess
import os
import sys
from pathlib import Path

os.chdir('/workspace/tiny-icf')

# Add src to path
sys.path.insert(0, str(Path('/workspace/tiny-icf/src')))

# Import and run training
from tiny_icf.train_lightning import main as train_main

# Set up arguments
import sys as sys_module
original_argv = sys_module.argv[:]
try:
    sys_module.argv = [
        'train_lightning.py',
        '--data', 'data/multilingual/multilingual_combined.csv',
        '--typo-corpus', 'data/typos/github_typos.csv',
        '--output-dir', '/workspace/tiny-icf/models',
        '--epochs', '50',
        '--batch-size', '256',
        '--devices', '1',
        '--precision', '16-mixed',
        '--num-workers', '4',
        '--early-stopping-patience', '10',
    ]
    exit_code = train_main()
    sys.exit(exit_code)
finally:
    sys_module.argv = original_argv
'''
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(train_script)
        script_path = f.name
    
    import os
    os.chmod(script_path, 0o755)
    
    try:
        # Upload script
        print("   Uploading PEP 723 training script...")
        subprocess.run(
            ["runpodctl", "send", script_path, pod_id, "/workspace/tiny-icf/auto_train.py"],
            check=True,
            capture_output=True,
            timeout=15,
        )
        
        # Create a simple bash wrapper that uses uv run
        wrapper = """#!/bin/bash
cd /workspace/tiny-icf
nohup uv run auto_train.py > training.log 2>&1 &
echo $! > training.pid
echo "Training started with PID: $(cat training.pid)"
"""
        
        wrapper_path = script_path.replace('.py', '.sh')
        with open(wrapper_path, 'w') as f:
            f.write(wrapper)
        os.chmod(wrapper_path, 0o755)
        
        subprocess.run(
            ["runpodctl", "send", wrapper_path, pod_id, "/workspace/tiny-icf/start_uv_train.sh"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        
        # Try to execute via Python (which can run bash)
        python_starter = """import subprocess, os
os.chdir('/workspace/tiny-icf')
proc = subprocess.Popen(['bash', 'start_uv_train.sh'],
                       stdout=open('train_start.log', 'a'),
                       stderr=subprocess.STDOUT)
print(f'Started PID: {proc.pid}')
"""
        
        py_starter_path = script_path.replace('.py', '_starter.py')
        with open(py_starter_path, 'w') as f:
            f.write(python_starter)
        
        subprocess.run(
            ["runpodctl", "send", py_starter_path, pod_id, "/tmp/start_train.py"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        
        # Execute starter
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
            print(f"   ⚠️  Direct execution had issues, but scripts ready")
            print(f"   Manual: runpodctl ssh connect {pod_id}")
            print(f"   Then: cd /workspace/tiny-icf && bash start_uv_train.sh")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⚠️  Timeout, but scripts uploaded")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        import os
        for p in [script_path, wrapper_path, py_starter_path]:
            try:
                if 'p' in locals():
                    os.unlink(p)
            except:
                pass

def check_status(pod_id):
    """Quick status check."""
    python_code = """import os
pid_file = '/workspace/tiny-icf/training.pid'
log_file = '/workspace/tiny-icf/training.log'

if os.path.exists(pid_file):
    pid = open(pid_file).read().strip()
    if os.path.exists(f'/proc/{pid}'):
        # Get last log line
        if os.path.exists(log_file):
            with open(log_file) as f:
                lines = f.readlines()
                last = lines[-1].strip() if lines else ''
            print(f'RUNNING:{pid}:{last[:60]}')
        else:
            print(f'RUNNING:{pid}')
    else:
        print('STOPPED')
elif os.path.exists(log_file):
    with open(log_file) as f:
        lines = f.readlines()
        if lines:
            print(f'STARTING:{lines[-1].strip()[:60]}')
        else:
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
            ["runpodctl", "send", py_path, pod_id, "/tmp/status.py"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        result = subprocess.run(
            ["runpodctl", "exec", "python", "--pod_id", pod_id, "/tmp/status.py"],
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
    
    print("🤖 Autonomous Training (MCP + uv + PEP 723)")
    print(f"   Pod: {pod_id}\n")
    
    # Configure
    api_key = get_api_key()
    if api_key:
        subprocess.run(
            ["runpodctl", "config", "--apiKey", api_key],
            capture_output=True,
            check=False,
        )
    
    # Check status
    status = check_status(pod_id)
    print(f"📊 Current: {status}\n")
    
    # Start if needed
    if "NOT_STARTED" in status or "UNKNOWN" in status:
        if start_training_uv(pod_id):
            time.sleep(3)
        else:
            print("\n⚠️  Auto-start had issues")
            print(f"   Manual: runpodctl ssh connect {pod_id}")
            print(f"   Then: cd /workspace/tiny-icf && bash start_uv_train.sh")
            return
    
    # Monitor
    print("📊 Monitoring (10s intervals, Ctrl+C to stop):\n")
    try:
        for i in range(1, 200):
            status = check_status(pod_id)
            parts = status.split(':')
            state = parts[0]
            
            if i % 3 == 0 or len(parts) > 1:  # Show every 3rd or if there's detail
                print(f"[{i}] {state}")
                if len(parts) > 2:
                    print(f"     {parts[2]}")
            
            if "STOPPED" in state:
                print("\n⚠️  Training stopped")
                break
            
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"\n⏸️  Stopped after {i} checks")
        print(f"   Training continues. Check: uv run scripts/mcp_autonomous_train.py {pod_id}")

if __name__ == "__main__":
    main()

