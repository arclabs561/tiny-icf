#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pexpect>=4.8.0",
# ]
# ///
"""
Auto-start using direct SSH with pexpect - most reliable.
"""
import sys
import subprocess
import re
import pexpect

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
    ssh_match = re.search(r'ssh\s+[^\s]+@[^\s]+', output)
    if ssh_match:
        return ssh_match.group(0)
    
    # Try alternative pattern
    ssh_match = re.search(r'ssh\s+[\w@\.-]+', output)
    if ssh_match:
        return ssh_match.group(0)
    
    return None

def start_via_ssh(pod_id):
    """Start training via direct SSH using pexpect."""
    print(f"🚀 Starting training via SSH on {pod_id}...")
    
    ssh_cmd = get_ssh_command(pod_id)
    if not ssh_cmd:
        print("   ❌ Could not get SSH command")
        return False
    
    print(f"   Using SSH: {ssh_cmd}")
    
    # Command to execute
    command = "cd /workspace/tiny-icf && bash start.sh && echo 'Training started'"
    
    try:
        # Use pexpect to handle SSH
        child = pexpect.spawn(f"{ssh_cmd} '{command}'", timeout=30)
        
        # Handle potential prompts
        index = child.expect([
            pexpect.EOF,
            pexpect.TIMEOUT,
            'password:',
            'Password:',
            'Are you sure',
        ], timeout=10)
        
        if index == 2 or index == 3:
            print("   ⚠️  SSH requires password (not supported)")
            return False
        elif index == 4:
            child.sendline('yes')
            child.expect(pexpect.EOF, timeout=10)
        
        output = child.before.decode() if child.before else ""
        print(f"   ✅ {output.strip()}")
        return True
        
    except pexpect.TIMEOUT:
        print("   ⚠️  SSH timeout")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    
    # Try direct SSH first
    if start_via_ssh(pod_id):
        print("\n✅ Training started via SSH!")
    else:
        print("\n⚠️  SSH method failed, trying autorun...")
        # Fallback to autorun
        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/auto_start_via_autorun.py', pod_id],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

if __name__ == "__main__":
    main()

