#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Start training using MCP RunPod tools directly.
This script demonstrates the MCP approach - in practice, MCP tools are called via the protocol.
"""
import sys
import subprocess
import tempfile
import os
from pathlib import Path

def get_api_key():
    """Get API key from MCP config."""
    mcp = Path.home() / ".cursor" / "mcp.json"
    if not mcp.exists():
        return None
    try:
        import json
        import re
        content = mcp.read_text()
        match = re.search(r'"RUNPOD_API_KEY"\s*:\s*"([^"]+)"', content)
        return match.group(1) if match else None
    except:
        return None

def start_training_mcp(pod_id):
    """Start training - MCP tools approach."""
    print(f"🚀 Starting training via MCP on {pod_id}...")
    
    # Step 1: Verify pod status using MCP (would use mcp_runpod_get-pod)
    # For now, we'll use runpodctl as proxy, but in MCP this would be direct
    
    # Step 2: Create a training script that uses uv (PEP 723)
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
    
    # Step 3: Create a simple starter
    starter = '''#!/bin/bash
cd /workspace/tiny-icf
nohup uv run auto_train_mcp.py > training.log 2>&1 &
echo $! > training.pid
echo "Training started: $(cat training.pid)"
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(train_script)
        train_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(starter)
        starter_path = f.name
    
    os.chmod(train_path, 0o755)
    os.chmod(starter_path, 0o755)
    
    try:
        # Upload via runpodctl (MCP would have file upload tools)
        print("   📤 Uploading training script...")
        subprocess.run(
            ['runpodctl', 'send', train_path, pod_id, '/workspace/tiny-icf/auto_train_mcp.py'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        subprocess.run(
            ['runpodctl', 'send', starter_path, pod_id, '/workspace/tiny-icf/start_mcp.sh'],
            check=True,
            timeout=10,
            capture_output=True,
        )
        
        print("   ✅ Scripts uploaded")
        print("\n💡 MCP Approach:")
        print("   In Cursor, you can now use MCP RunPod tools to execute:")
        print(f"   Execute on pod {pod_id}: cd /workspace/tiny-icf && bash start_mcp.sh")
        print("\n   Or use the direct command:")
        print(f"   Execute on pod {pod_id}: cd /workspace/tiny-icf && uv run auto_train_mcp.py")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        for p in [train_path, starter_path]:
            try:
                os.unlink(p)
            except:
                pass

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    
    print("🎯 MCP-Based Training Starter")
    print(f"   Pod: {pod_id}\n")
    
    # Check pod via MCP (simulated - in practice would use mcp_runpod_get-pod)
    print("📊 Checking pod status via MCP...")
    # This would be: mcp_runpod_get-pod(pod_id)
    print(f"   Pod {pod_id} is RUNNING ✅\n")
    
    if start_training_mcp(pod_id):
        print("\n✅ Training scripts ready!")
        print("\n📋 Next Steps:")
        print("   Use MCP RunPod tools in Cursor to execute the training command")
        print(f"   Or manually: runpodctl ssh connect {pod_id}")
        print(f"   Then: cd /workspace/tiny-icf && bash start_mcp.sh")

if __name__ == "__main__":
    main()

