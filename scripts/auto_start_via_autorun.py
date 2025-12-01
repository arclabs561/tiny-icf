#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Auto-start training using /etc/rc.local or startup script approach.
"""
import sys
import subprocess
import tempfile
import os

def create_autorun_script(pod_id):
    """Create an autorun script that executes on pod startup."""
    print(f"🚀 Creating autorun script on {pod_id}...")
    
    # Create a script that runs training and stores PID
    autorun = """#!/bin/bash
# Autorun script for training
cd /workspace/tiny-icf

# Check if already running
if [ -f training.pid ]; then
    PID=$(cat training.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Training already running (PID: $PID)"
        exit 0
    fi
fi

# Start training
nohup bash start.sh > /tmp/autorun.log 2>&1 &
echo "Autorun started training"
"""
    
    # Create installer that adds to startup
    installer = """#!/bin/bash
set -e

# Copy autorun script
cp /tmp/autorun.sh /workspace/tiny-icf/autorun.sh
chmod +x /workspace/tiny-icf/autorun.sh

# Add to rc.local if it exists
if [ -f /etc/rc.local ]; then
    if ! grep -q "autorun.sh" /etc/rc.local; then
        sudo sed -i '$i /workspace/tiny-icf/autorun.sh' /etc/rc.local
    fi
fi

# Also create a cron job that runs on boot
(crontab -l 2>/dev/null | grep -v autorun.sh; echo "@reboot /workspace/tiny-icf/autorun.sh") | crontab -

# Run it now
/workspace/tiny-icf/autorun.sh

echo "Autorun configured and started"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(autorun)
        autorun_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(installer)
        installer_path = f.name
    
    os.chmod(autorun_path, 0o755)
    os.chmod(installer_path, 0o755)
    
    try:
        # Upload both
        print("   📤 Uploading scripts...")
        subprocess.run(
            ['runpodctl', 'send', autorun_path, pod_id, '/tmp/autorun.sh'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        subprocess.run(
            ['runpodctl', 'send', installer_path, pod_id, '/tmp/install_autorun.sh'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        
        # Execute installer via Python
        runner = '''import subprocess
result = subprocess.run(['bash', '/tmp/install_autorun.sh'],
                       capture_output=True, text=True, timeout=20)
print(result.stdout)
if result.stderr:
    print("Stderr:", result.stderr)
exit(result.returncode)
'''
        
        runner_path = installer_path.replace('.sh', '_runner.py')
        with open(runner_path, 'w') as f:
            f.write(runner)
        
        subprocess.run(
            ['runpodctl', 'send', runner_path, pod_id, '/tmp/run_autorun_installer.py'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        
        # Execute
        print("   ⚡ Installing autorun...")
        proc = subprocess.Popen(
            ['runpodctl', 'exec', 'python', '--pod_id', pod_id, '/tmp/run_autorun_installer.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        try:
            stdout, stderr = proc.communicate(timeout=20)
            output = stdout.decode()
            if proc.returncode == 0:
                print(f"   ✅ {output.strip()}")
                return True
            else:
                print(f"   ⚠️  Return code {proc.returncode}")
                print(f"   Output: {output[:200]}")
                if stderr:
                    print(f"   Error: {stderr.decode()[:200]}")
                return False
        except subprocess.TimeoutExpired:
            proc.kill()
            print("   ⚠️  Timeout, but autorun may be configured")
            return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        for p in [autorun_path, installer_path, runner_path]:
            try:
                os.unlink(p)
            except:
                pass

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    create_autorun_script(pod_id)

if __name__ == "__main__":
    main()

