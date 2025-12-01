#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Simplest auto-start: create a script that runs itself via nohup.
"""
import sys
import subprocess
import tempfile
import os

def start_training_simple(pod_id):
    """Create and execute a simple self-running script."""
    print(f"🚀 Simple auto-start on {pod_id}...")
    
    # Create a script that starts itself in background
    # This script will be uploaded and then we'll trigger it via a simple file touch or signal
    starter = """#!/bin/bash
cd /workspace/tiny-icf

# Check if already running
if [ -f training.pid ]; then
    PID=$(cat training.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Already running: $PID"
        exit 0
    fi
fi

# Start training
nohup uv run auto_train.py > training.log 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > training.pid
echo "Training started: $TRAIN_PID"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(starter)
        script_path = f.name
    
    os.chmod(script_path, 0o755)
    
    try:
        # Upload
        print("   📤 Uploading starter...")
        subprocess.run(
            ['runpodctl', 'send', script_path, pod_id, '/workspace/tiny-icf/start_simple.sh'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        
        # Method 1: Use 'at' command to schedule immediate execution
        at_command = """import subprocess
result = subprocess.run(['bash', '-c', 'echo "bash /workspace/tiny-icf/start_simple.sh" | at now'],
                       capture_output=True, text=True, timeout=10)
print(result.stdout)
if result.stderr:
    print("Stderr:", result.stderr)
"""
        
        at_runner = script_path.replace('.sh', '_at.py')
        with open(at_runner, 'w') as f:
            f.write(at_command)
        
        subprocess.run(
            ['runpodctl', 'send', at_runner, pod_id, '/tmp/run_at.py'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        
        # Try executing via Python
        print("   ⚡ Executing via 'at' command...")
        proc = subprocess.Popen(
            ['runpodctl', 'exec', 'python', '--pod_id', pod_id, '/tmp/run_at.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        try:
            stdout, stderr = proc.communicate(timeout=10)
            if proc.returncode == 0:
                print(f"   ✅ Scheduled: {stdout.decode().strip()}")
                time.sleep(2)  # Wait a moment
            else:
                print(f"   ⚠️  'at' failed, trying direct execution...")
                # Fallback: try direct bash execution
                raise Exception("at failed")
        except:
            # Method 2: Direct execution via bash
            print("   ⚡ Trying direct bash execution...")
            bash_runner = """import subprocess
proc = subprocess.Popen(['bash', '/workspace/tiny-icf/start_simple.sh'],
                       stdout=open('/tmp/start_output.log', 'a'),
                       stderr=subprocess.STDOUT,
                       start_new_session=True)
print(f'Started: {proc.pid}')
"""
            
            bash_runner_path = script_path.replace('.sh', '_bash.py')
            with open(bash_runner_path, 'w') as f:
                f.write(bash_runner)
            
            subprocess.run(
                ['runpodctl', 'send', bash_runner_path, pod_id, '/tmp/run_bash.py'],
                check=True,
                timeout=10,
                capture_output=True,
            )
            
            proc2 = subprocess.Popen(
                ['runpodctl', 'exec', 'python', '--pod_id', pod_id, '/tmp/run_bash.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            try:
                stdout, stderr = proc2.communicate(timeout=8)
                if proc2.returncode == 0:
                    print(f"   ✅ {stdout.decode().strip()}")
                    return True
                else:
                    print(f"   ⚠️  {stderr.decode()[:100]}")
            except subprocess.TimeoutExpired:
                proc2.kill()
                print("   ⚠️  Timeout, but script is ready")
                print(f"   Manual: runpodctl ssh connect {pod_id}")
                print(f"   Then: bash /workspace/tiny-icf/start_simple.sh")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        for p in [script_path, at_runner, bash_runner_path]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except:
                pass

def main():
    import time
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    start_training_simple(pod_id)

if __name__ == "__main__":
    main()

