#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pexpect>=4.8.0",
# ]
# ///
"""
Working auto-start: Uses pexpect to automate SSH execution.
"""
import sys
import subprocess
import pexpect
import re
from pathlib import Path

def get_ssh_command(pod_id):
    """Get SSH command from runpodctl."""
    result = subprocess.run(
        ['runpodctl', 'ssh', 'connect', pod_id],
        capture_output=True,
        text=True,
        timeout=5,
    )
    
    output = result.stdout + result.stderr
    # Look for SSH command
    ssh_match = re.search(r'ssh\s+[\w@\.-]+', output)
    if ssh_match:
        return ssh_match.group(0)
    return None

def start_via_pexpect(pod_id):
    """Start training using pexpect to automate SSH."""
    print(f"🚀 Auto-starting training on {pod_id}...")
    
    ssh_cmd = get_ssh_command(pod_id)
    if not ssh_cmd:
        print("   ❌ Could not get SSH command")
        return False
    
    print(f"   Using SSH: {ssh_cmd}")
    
    # Command to execute
    command = "cd /workspace/tiny-icf && bash start_mcp.sh && echo 'Training started successfully'"
    
    try:
        # Use pexpect to handle SSH
        child = pexpect.spawn(f"{ssh_cmd} '{command}'", timeout=30)
        
        # Handle various prompts
        index = child.expect([
            pexpect.EOF,
            pexpect.TIMEOUT,
            'password:',
            'Password:',
            'Are you sure',
            'yes/no',
        ], timeout=15)
        
        if index == 2 or index == 3:
            print("   ⚠️  SSH requires password (not supported)")
            return False
        elif index == 4 or index == 5:
            child.sendline('yes')
            child.expect(pexpect.EOF, timeout=15)
        
        output = child.before.decode() if child.before else ""
        if "Training started" in output or child.exitstatus == 0:
            print(f"   ✅ {output.strip()}")
            return True
        else:
            print(f"   ⚠️  Output: {output[:200]}")
            return False
            
    except pexpect.TIMEOUT:
        print("   ⚠️  SSH timeout")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def create_persistent_service(pod_id):
    """Create a systemd service that starts training."""
    print(f"🔧 Creating persistent service on {pod_id}...")
    
    # Create service file
    service_content = """[Unit]
Description=IDF Training Auto-Start
After=network.target

[Service]
Type=oneshot
ExecStart=/bin/bash /workspace/tiny-icf/start_mcp.sh
StandardOutput=append:/workspace/tiny-icf/training.log
StandardError=append:/workspace/tiny-icf/training.log
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""
    
    # Create installer
    installer = f"""#!/bin/bash
set -e
cat > /tmp/idf-training.service << 'EOFSERVICE'
{service_content}
EOFSERVICE
sudo cp /tmp/idf-training.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable idf-training.service
sudo systemctl start idf-training.service
echo "Service status: $(sudo systemctl is-active idf-training.service)"
"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(installer)
        installer_path = f.name
    
    import os
    os.chmod(installer_path, 0o755)
    
    try:
        # Upload
        subprocess.run(
            ['runpodctl', 'send', installer_path, pod_id, '/tmp/install_service.sh'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        
        # Execute via Python
        python_exec = """import subprocess
result = subprocess.run(['bash', '/tmp/install_service.sh'],
                       capture_output=True, text=True, timeout=30)
print(result.stdout)
if result.stderr:
    print("Stderr:", result.stderr)
exit(result.returncode)
"""
        
        py_path = installer_path.replace('.sh', '.py')
        with open(py_path, 'w') as f:
            f.write(python_exec)
        
        subprocess.run(
            ['runpodctl', 'send', py_path, pod_id, '/tmp/run_service_installer.py'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        
        # Try to execute
        print("   ⚡ Installing service...")
        proc = subprocess.Popen(
            ['runpodctl', 'exec', 'python', '--pod_id', pod_id, '/tmp/run_service_installer.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        try:
            stdout, stderr = proc.communicate(timeout=20)
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
        for p in [installer_path, py_path]:
            try:
                os.unlink(p)
            except:
                pass

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    
    print("🎯 Working Auto-Start Solution")
    print(f"   Pod: {pod_id}\n")
    
    # Try pexpect first
    print("📡 Method 1: Automated SSH (pexpect)")
    if start_via_pexpect(pod_id):
        print("\n✅ Training started via SSH!")
        return
    
    # Fallback to persistent service
    print("\n📡 Method 2: Persistent Systemd Service")
    if create_persistent_service(pod_id):
        print("\n✅ Service created and started!")
        return
    
    # Final fallback
    print("\n⚠️  Automated methods had issues")
    print("📋 Manual start (guaranteed to work):")
    print(f"   runpodctl ssh connect {pod_id}")
    print(f"   cd /workspace/tiny-icf && bash start_mcp.sh")

if __name__ == "__main__":
    main()

