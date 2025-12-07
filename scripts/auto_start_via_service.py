#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Auto-start training using systemd user service - reliable and persistent.
"""
import sys
import subprocess
import tempfile
import os
from pathlib import Path

def create_systemd_service(pod_id):
    """Create a systemd user service that starts training."""
    print(f"🚀 Creating systemd service on {pod_id}...")
    
    # Create systemd service file
    service_content = """[Unit]
Description=IDF Training Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace/tiny-icf
ExecStart=/bin/bash /workspace/tiny-icf/start.sh
Restart=on-failure
StandardOutput=append:/workspace/tiny-icf/training.log
StandardError=append:/workspace/tiny-icf/training.log

[Install]
WantedBy=multi-user.target
"""
    
    # Create installer script
    installer = f"""#!/bin/bash
set -e
cd /workspace/tiny-icf

# Create service file
cat > /tmp/idf-training.service << 'EOFSERVICE'
{service_content}
EOFSERVICE

# Copy to systemd
sudo cp /tmp/idf-training.service /etc/systemd/system/idf-training.service

# Reload and start
sudo systemctl daemon-reload
sudo systemctl enable idf-training.service
sudo systemctl start idf-training.service

echo "Service started: $(sudo systemctl is-active idf-training.service)"
echo "Status: $(sudo systemctl status idf-training.service --no-pager | head -3)"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(installer)
        installer_path = f.name
    
    os.chmod(installer_path, 0o755)
    
    try:
        # Upload installer
        print("   📤 Uploading service installer...")
        subprocess.run(
            ['runpodctl', 'send', installer_path, pod_id, '/tmp/install_service.sh'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        
        # Execute installer
        print("   🔧 Installing service...")
        # Use a Python script to run the installer
        runner = '''import subprocess
result = subprocess.run(['bash', '/tmp/install_service.sh'],
                       capture_output=True, text=True, timeout=30)
print(result.stdout)
if result.returncode != 0:
    print("Error:", result.stderr)
    exit(result.returncode)
'''
        
        runner_path = installer_path.replace('.sh', '_runner.py')
        with open(runner_path, 'w') as f:
            f.write(runner)
        
        subprocess.run(
            ['runpodctl', 'send', runner_path, pod_id, '/tmp/run_installer.py'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        
        # Try to execute
        print("   ⚡ Starting service...")
        proc = subprocess.Popen(
            ['runpodctl', 'exec', 'python', '--pod_id', pod_id, '/tmp/run_installer.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        try:
            stdout, stderr = proc.communicate(timeout=15)
            if proc.returncode == 0:
                print(f"   ✅ {stdout.decode().strip()}")
                return True
            else:
                print(f"   ⚠️  {stderr.decode()[:200]}")
                return False
        except subprocess.TimeoutExpired:
            proc.kill()
            print("   ⚠️  Timeout, but service may be installing")
            return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        for p in [installer_path, runner_path]:
            try:
                os.unlink(p)
            except:
                pass

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    create_systemd_service(pod_id)

if __name__ == "__main__":
    main()

