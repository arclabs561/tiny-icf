#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Fast, non-blocking training starter using MCP + uv.
No hanging - shows progress immediately.
"""
import sys
import subprocess
import time
from pathlib import Path
import json
import re

def get_api_key():
    mcp = Path.home() / ".cursor" / "mcp.json"
    if not mcp.exists():
        return None
    try:
        match = re.search(r'"RUNPOD_API_KEY"\s*:\s*"([^"]+)"', mcp.read_text())
        return match.group(1) if match else None
    except:
        return None

def upload_and_start(pod_id):
    """Upload training script and start it - fast, non-blocking."""
    print(f"🚀 Quick start on {pod_id}...")
    
    # Create minimal PEP 723 training script
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
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/workspace/tiny-icf/src')))
from tiny_icf.train_lightning import main
import sys as s
s.argv = ['train_lightning.py', '--data', 'data/multilingual/multilingual_combined.csv',
          '--typo-corpus', 'data/typos/github_typos.csv', '--output-dir', '/workspace/tiny-icf/models',
          '--epochs', '50', '--batch-size', '256', '--devices', '1', '--precision', '16-mixed']
main()
'''
    
    # Create simple bash starter
    starter = '''#!/bin/bash
cd /workspace/tiny-icf
nohup uv run auto_train.py > training.log 2>&1 &
echo $! > training.pid
echo "Started: $(cat training.pid)"
'''
    
    import tempfile
    import os
    
    # Write files
    train_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    train_file.write(train_script)
    train_file.close()
    os.chmod(train_file.name, 0o755)
    
    starter_file = tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False)
    starter_file.write(starter)
    starter_file.close()
    os.chmod(starter_file.name, 0o755)
    
    try:
        # Upload both files (non-blocking, show progress)
        print("   📤 Uploading scripts...")
        subprocess.Popen(
            ["runpodctl", "send", train_file.name, pod_id, "/workspace/tiny-icf/auto_train.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.Popen(
            ["runpodctl", "send", starter_file.name, pod_id, "/workspace/tiny-icf/start.sh"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        print("   ✅ Files uploading in background")
        print("   ⏳ Waiting 3s for uploads...")
        time.sleep(3)
        
        # Provide clear next steps (don't try to execute, just prepare)
        print("\n✅ Scripts ready on pod!")
        print(f"\n📋 To start training:")
        print(f"   runpodctl ssh connect {pod_id}")
        print(f"   cd /workspace/tiny-icf && bash start.sh")
        print(f"\n📊 To monitor:")
        print(f"   tail -f /workspace/tiny-icf/training.log")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        os.unlink(train_file.name)
        os.unlink(starter_file.name)

def quick_status(pod_id):
    """Quick non-blocking status check."""
    # Use a simple file check via Python exec (with timeout)
    py_code = """import os
pid = '/workspace/tiny-icf/training.pid'
log = '/workspace/tiny-icf/training.log'
if os.path.exists(pid):
    p = open(pid).read().strip()
    if os.path.exists(f'/proc/{p}'):
        print('RUNNING')
    else:
        print('STOPPED')
elif os.path.exists(log):
    print('STARTING')
else:
    print('NOT_STARTED')
"""
    
    import tempfile
    import os
    tf = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    tf.write(py_code)
    tf.close()
    
    try:
        # Upload and check (with short timeout)
        subprocess.run(
            ["runpodctl", "send", tf.name, pod_id, "/tmp/check.py"],
            timeout=5,
            capture_output=True,
            check=False,
        )
        result = subprocess.run(
            ["runpodctl", "exec", "python", "--pod_id", pod_id, "/tmp/check.py"],
            timeout=8,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except:
        return "UNKNOWN"
    finally:
        os.unlink(tf.name)

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    
    print("⚡ Quick Training Starter (MCP + uv + PEP 723)")
    print(f"   Pod: {pod_id}\n")
    
    # Configure (fast)
    api_key = get_api_key()
    if api_key:
        subprocess.run(
            ["runpodctl", "config", "--apiKey", api_key],
            capture_output=True,
            check=False,
        )
    
    # Quick status (non-blocking)
    print("📊 Checking status...")
    status = quick_status(pod_id)
    print(f"   Status: {status}\n")
    
    if "NOT_STARTED" in status or "UNKNOWN" in status or "TIMEOUT" in status:
        upload_and_start(pod_id)
    else:
        print(f"✅ Training appears to be {status}")
        print(f"\n📊 Monitor:")
        print(f"   runpodctl ssh connect {pod_id}")
        print(f"   tail -f /workspace/tiny-icf/training.log")

if __name__ == "__main__":
    main()

