#!/usr/bin/env python3
"""Quick training starter - fast and direct."""
import subprocess
import sys
import time
import json
import re
from pathlib import Path

def get_api_key():
    mcp = Path.home() / ".cursor" / "mcp.json"
    if not mcp.exists():
        return None
    try:
        content = mcp.read_text()
        match = re.search(r'"RUNPOD_API_KEY"\s*:\s*"([^"]+)"', content)
        return match.group(1) if match else None
    except:
        return None

def quick_start(pod_id):
    """Start training quickly."""
    print(f"🚀 Starting training on {pod_id}...")
    
    # Method: Use Python exec to start bash script
    python_code = """import subprocess, os
os.chdir('/workspace/tiny-icf')
proc = subprocess.Popen(['bash', 'start_training.sh'], 
                       stdout=open('training.log', 'a'), 
                       stderr=subprocess.STDOUT)
print(f'Started PID: {proc.pid}')
with open('training.pid', 'w') as f:
    f.write(str(proc.pid))
"""
    
    # Write temp Python file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(python_code)
        py_path = f.name
    
    try:
        # Upload
        subprocess.run(['runpodctl', 'send', py_path, pod_id, '/tmp/start.py'], 
                      check=True, capture_output=True, timeout=10)
        
        # Execute
        result = subprocess.run(['runpodctl', 'exec', 'python', '--pod_id', pod_id, '/tmp/start.py'],
                               capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
            return True
        else:
            print(f"⚠️  {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        import os
        os.unlink(py_path)

def quick_status(pod_id):
    """Quick status check."""
    python_code = """import os
pid_file = '/workspace/tiny-icf/training.pid'
log_file = '/workspace/tiny-icf/training.log'

if os.path.exists(pid_file):
    pid = open(pid_file).read().strip()
    if os.path.exists(f'/proc/{pid}'):
        print(f'RUNNING:{pid}')
    else:
        print('STOPPED')
elif os.path.exists(log_file):
    print('STARTING')
else:
    print('NOT_STARTED')
"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(python_code)
        py_path = f.name
    
    try:
        subprocess.run(['runpodctl', 'send', py_path, pod_id, '/tmp/status.py'],
                      check=True, capture_output=True, timeout=10)
        result = subprocess.run(['runpodctl', 'exec', 'python', '--pod_id', pod_id, '/tmp/status.py'],
                               capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"
    except:
        return "UNKNOWN"
    finally:
        import os
        os.unlink(py_path)

def quick_log(pod_id, lines=5):
    """Get last N lines of log."""
    python_code = f"""try:
    with open('/workspace/tiny-icf/training.log') as f:
        lines = f.readlines()
        print(''.join(lines[-{lines}:]))
except:
    print('No log yet')
"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(python_code)
        py_path = f.name
    
    try:
        subprocess.run(['runpodctl', 'send', py_path, pod_id, '/tmp/log.py'],
                      check=True, capture_output=True, timeout=10)
        result = subprocess.run(['runpodctl', 'exec', 'python', '--pod_id', pod_id, '/tmp/log.py'],
                               capture_output=True, text=True, timeout=10)
        return result.stdout if result.returncode == 0 else ""
    except:
        return ""
    finally:
        import os
        os.unlink(py_path)

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    
    # Configure
    api_key = get_api_key()
    if api_key:
        subprocess.run(['runpodctl', 'config', '--apiKey', api_key], 
                      capture_output=True, check=False)
    
    print(f"📊 Pod: {pod_id}\n")
    
    # Check status
    status = quick_status(pod_id)
    print(f"Current: {status}\n")
    
    # Start if needed
    if "NOT_STARTED" in status or "UNKNOWN" in status:
        if quick_start(pod_id):
            time.sleep(2)
        else:
            print("⚠️  Auto-start failed. Manual:")
            print(f"   runpodctl ssh connect {pod_id}")
            print(f"   cd /workspace/tiny-icf && bash start_training.sh")
            return
    
    # Quick monitor
    print("📊 Quick monitor (Ctrl+C to stop):\n")
    try:
        for i in range(1, 100):
            status = quick_status(pod_id)
            log = quick_log(pod_id, 3)
            
            print(f"[{i}] {status}")
            if log.strip():
                for line in log.strip().split('\n')[-2:]:
                    if line.strip():
                        print(f"    {line[:80]}")
            
            if "STOPPED" in status:
                print("\n⚠️  Training stopped")
                break
            
            time.sleep(10)  # Check every 10 seconds
    except KeyboardInterrupt:
        print(f"\n⏸️  Stopped after {i} checks")

if __name__ == "__main__":
    main()

