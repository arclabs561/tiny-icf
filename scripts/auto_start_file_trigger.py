#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
File-based trigger: create a trigger file that starts training.
Uses a simple file existence check - most reliable.
"""
import sys
import subprocess
import tempfile
import os
import time

def start_via_file_trigger(pod_id):
    """Start training by creating a trigger file that a watcher monitors."""
    print(f"🚀 File-trigger auto-start on {pod_id}...")
    
    # Create a watcher script that runs in background
    watcher = """#!/bin/bash
# Watcher script - runs training when trigger file appears
cd /workspace/tiny-icf

TRIGGER_FILE="/workspace/tiny-icf/.start_training"
PID_FILE="/workspace/tiny-icf/training.pid"
LOG_FILE="/workspace/tiny-icf/training.log"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Training already running (PID: $PID)"
        exit 0
    fi
fi

# Wait for trigger file
while [ ! -f "$TRIGGER_FILE" ]; do
    sleep 1
done

# Remove trigger
rm -f "$TRIGGER_FILE"

# Start training
echo "Starting training..." >> "$LOG_FILE"
nohup uv run auto_train.py >> "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"
echo "Training started: $TRAIN_PID" >> "$LOG_FILE"
"""
    
    # Create a simple starter that just creates the trigger file
    trigger_creator = """#!/bin/bash
touch /workspace/tiny-icf/.start_training
echo "Trigger file created"
"""
    
    # Create a systemd service or cron job to run the watcher
    # Actually, simpler: run watcher in background via nohup
    watcher_runner = """#!/bin/bash
# Run watcher in background
nohup bash /workspace/tiny-icf/watcher.sh > /tmp/watcher.log 2>&1 &
echo $! > /workspace/tiny-icf/watcher.pid
echo "Watcher started: $(cat /workspace/tiny-icf/watcher.pid)"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(watcher)
        watcher_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(trigger_creator)
        trigger_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(watcher_runner)
        runner_path = f.name
    
    for p in [watcher_path, trigger_path, runner_path]:
        os.chmod(p, 0o755)
    
    try:
        # Upload all scripts
        print("   📤 Uploading scripts...")
        subprocess.run(
            ['runpodctl', 'send', watcher_path, pod_id, '/workspace/tiny-icf/watcher.sh'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        subprocess.run(
            ['runpodctl', 'send', trigger_path, pod_id, '/workspace/tiny-icf/create_trigger.sh'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        subprocess.run(
            ['runpodctl', 'send', runner_path, pod_id, '/workspace/tiny-icf/start_watcher.sh'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        
        # Create Python script to start watcher and create trigger
        python_starter = """import subprocess, os, time
os.chdir('/workspace/tiny-icf')

# Start watcher if not running
if not os.path.exists('watcher.pid') or not os.path.exists(f'/proc/{open("watcher.pid").read().strip()}'):
    proc = subprocess.Popen(['bash', 'start_watcher.sh'],
                           stdout=open('/tmp/watcher_start.log', 'a'),
                           stderr=subprocess.STDOUT)
    time.sleep(1)

# Create trigger file
with open('.start_training', 'w') as f:
    f.write('start')
print('Trigger created, training should start...')
"""
        
        py_starter_path = watcher_path.replace('.sh', '_starter.py')
        with open(py_starter_path, 'w') as f:
            f.write(python_starter)
        
        subprocess.run(
            ['runpodctl', 'send', py_starter_path, pod_id, '/tmp/file_trigger_starter.py'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        
        # Execute - but use a very short timeout since we're just creating a file
        print("   ⚡ Starting watcher and creating trigger...")
        proc = subprocess.Popen(
            ['runpodctl', 'exec', 'python', '--pod_id', pod_id, '/tmp/file_trigger_starter.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        try:
            stdout, stderr = proc.communicate(timeout=5)  # Very short timeout
            if proc.returncode == 0:
                print(f"   ✅ {stdout.decode().strip()}")
                print("   ⏳ Waiting 3s for training to start...")
                time.sleep(3)
                return True
            else:
                print(f"   ⚠️  {stderr.decode()[:100]}")
        except subprocess.TimeoutExpired:
            proc.kill()
            # Even if timeout, the file creation might have worked
            print("   ⚠️  Timeout, but trigger may have been created")
            print("   ⏳ Waiting 3s...")
            time.sleep(3)
            return True  # Assume it worked
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        for p in [watcher_path, trigger_path, runner_path, py_starter_path]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except:
                pass

def check_status(pod_id):
    """Quick status check."""
    python_code = """import os
pid_file = '/workspace/tiny-icf/training.pid'
if os.path.exists(pid_file):
    pid = open(pid_file).read().strip()
    if os.path.exists(f'/proc/{pid}'):
        print(f'RUNNING:{pid}')
    else:
        print('STOPPED')
else:
    print('NOT_STARTED')
"""
    
    tf = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    tf.write(python_code)
    tf.close()
    
    try:
        subprocess.run(
            ['runpodctl', 'send', tf.name, pod_id, '/tmp/check_status.py'],
            timeout=5,
            capture_output=True,
            check=False,
        )
        result = subprocess.run(
            ['runpodctl', 'exec', 'python', '--pod_id', pod_id, '/tmp/check_status.py'],
            timeout=6,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"
    except:
        return "UNKNOWN"
    finally:
        os.unlink(tf.name)

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    
    if start_via_file_trigger(pod_id):
        print("\n📊 Checking status...")
        status = check_status(pod_id)
        print(f"   Status: {status}")
        
        if "RUNNING" in status:
            print("\n✅ Training is running!")
            print(f"   Monitor: runpodctl ssh connect {pod_id}")
            print(f"   Then: tail -f /workspace/tiny-icf/training.log")
        else:
            print("\n⚠️  Training may still be starting...")
            print(f"   Check: runpodctl ssh connect {pod_id}")
            print(f"   Then: cat /workspace/tiny-icf/training.pid")

if __name__ == "__main__":
    main()

