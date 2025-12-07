#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""
Final auto-start solution: Use RunPod API directly + simple file-based approach.
"""
import sys
import subprocess
import json
import re
from pathlib import Path

def get_api_key():
    """Get API key from MCP config."""
    mcp = Path.home() / ".cursor" / "mcp.json"
    if not mcp.exists():
        return None
    try:
        match = re.search(r'"RUNPOD_API_KEY"\s*:\s*"([^"]+)"', mcp.read_text())
        return match.group(1) if match else None
    except:
        return None

def start_via_api(pod_id):
    """Start training using RunPod API to execute command."""
    api_key = get_api_key()
    if not api_key:
        print("❌ No API key found")
        return False
    
    print(f"🚀 Starting via RunPod API on {pod_id}...")
    
    # Use RunPod API to execute command
    # The API endpoint for executing commands is typically:
    # POST https://api.runpod.io/graphql
    # But we can also use runpodctl with API key
    
    # Actually, simplest: create a script that runs on pod startup
    # and use runpodctl to trigger it via file creation
    
    # Create a simple one-liner script
    oneliner = """cd /workspace/tiny-icf && nohup bash start.sh > /tmp/auto_start.log 2>&1 & echo $! > training.pid && echo "Started: $(cat training.pid)" """
    
    # Upload a Python script that executes this
    python_exec = f"""import subprocess, os
os.chdir('/workspace/tiny-icf')
proc = subprocess.Popen(['bash', '-c', '{oneliner}'],
                       stdout=open('/tmp/api_start.log', 'a'),
                       stderr=subprocess.STDOUT,
                       start_new_session=True)
print('Started via API:', proc.pid)
"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(python_exec)
        py_path = f.name
    
    try:
        # Upload
        subprocess.run(
            ['runpodctl', 'send', py_path, pod_id, '/tmp/api_start.py'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        
        # Use runpodctl with API (if it supports it)
        # Or use requests to call RunPod API directly
        print("   ⚡ Executing via API...")
        
        # Try using runpodctl with shorter timeout
        proc = subprocess.Popen(
            ['timeout', '5', 'runpodctl', 'exec', 'python', '--pod_id', pod_id, '/tmp/api_start.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        try:
            stdout, stderr = proc.communicate(timeout=6)
            if proc.returncode == 0:
                print(f"   ✅ {stdout.decode().strip()}")
                return True
            else:
                # Even if it fails, the script is there
                print("   ⚠️  Execution had issues, but script ready")
                print(f"   Script location: /tmp/api_start.py on pod")
                return False
        except subprocess.TimeoutExpired:
            proc.kill()
            print("   ⚠️  Timeout (expected with runpodctl exec)")
            print("   ✅ Script uploaded and ready")
            print(f"\n💡 Best approach: Use SSH directly")
            print(f"   runpodctl ssh connect {pod_id}")
            print(f"   python3 /tmp/api_start.py")
            return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        import os
        os.unlink(py_path)

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    
    print("🎯 Final Auto-Start Solution")
    print(f"   Pod: {pod_id}\n")
    
    if start_via_api(pod_id):
        print("\n✅ Training started!")
    else:
        print("\n📋 Alternative: One-command start via SSH")
        print(f"   runpodctl ssh connect {pod_id} -c 'cd /workspace/tiny-icf && bash start.sh'")
        print("\n   Or interactively:")
        print(f"   runpodctl ssh connect {pod_id}")
        print(f"   cd /workspace/tiny-icf && bash start.sh")

if __name__ == "__main__":
    main()

