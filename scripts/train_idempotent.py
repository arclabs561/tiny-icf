#!/usr/bin/env python3
"""
Idempotent training using MCP RunPod tools.
Single command: train → monitor → results
"""

import json
import subprocess
import sys
from pathlib import Path

# Get API key from MCP config
def get_api_key():
    mcp_config = Path.home() / '.cursor' / 'mcp.json'
    with open(mcp_config) as f:
        content = f.read()
    import re
    match = re.search(r'"RUNPOD_API_KEY":\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    raise ValueError("Could not find RUNPOD_API_KEY in MCP config")

# This would use MCP tools directly, but for now we'll use runpodctl
# In practice, MCP tools would be called through the MCP protocol

def main():
    """Single idempotent command for training."""
    print("🚀 Idempotent Training (MCP RunPod Tools)")
    print("=" * 60)
    
    # Get pod ID (use first running pod or create new)
    api_key = get_api_key()
    
    # Check existing pods
    result = subprocess.run(
        ["runpodctl", "get", "pod"],
        capture_output=True,
        text=True
    )
    
    pod_id = None
    for line in result.stdout.split('\n'):
        if 'RUNNING' in line and 'RTX 3090' in line:
            pod_id = line.split()[0]
            break
    
    if not pod_id:
        print("❌ No running pod found")
        sys.exit(1)
    
    print(f"✅ Using pod: {pod_id}")
    
    # Single command that does everything
    # This would ideally use MCP tools, but using runpodctl as proxy
    command = f"""
import subprocess
import os
import sys
import time
from pathlib import Path

WORK_DIR = '/workspace/tiny-icf'
MODEL_PATH = f'{{WORK_DIR}}/models/model_runpod.pt'
TRAINING_LOG = f'{{WORK_DIR}}/training.log'
LOCK_FILE = f'{{WORK_DIR}}/.training.lock'

os.chdir(WORK_DIR)

def check_model():
    return os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000

def check_training():
    return os.path.exists(LOCK_FILE)

def install():
    print('🔧 Installing dependencies...')
    result = subprocess.run(['uv', 'pip', 'install', '-e', '.'], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print('⚠️  Dependencies may already be installed')

def train():
    print('🎯 Starting training...')
    Path(LOCK_FILE).touch()
    
    cmd = [
        'python', '-m', 'tiny_icf.train_curriculum',
        '--data', 'data/multilingual/multilingual_combined.csv',
        '--typo-corpus', 'data/typos/github_typos.csv',
        '--multilingual',
        '--epochs', '50',
        '--batch-size', '128',
        '--lr', '1e-3',
        '--curriculum-stages', '5',
        '--output', MODEL_PATH
    ]
    
    with open(TRAINING_LOG, 'a') as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f'Training PID: {{process.pid}}')
        process.wait()
        Path(LOCK_FILE).unlink(missing_ok=True)

def monitor():
    print('📊 Monitoring...')
    last_pos = 0
    while True:
        if not check_training():
            if check_model():
                print('✅ Training complete!')
                break
            else:
                print('⚠️  Training stopped')
                break
        
        if os.path.exists(TRAINING_LOG):
            with open(TRAINING_LOG, 'r') as f:
                f.seek(last_pos)
                new_lines = f.readlines()
                for line in new_lines:
                    print(line.rstrip())
                last_pos = f.tell()
        
        time.sleep(10)

def results():
    print('\\n📈 Results:')
    if check_model():
        size = os.path.getsize(MODEL_PATH) / 1024
        print(f'✅ Model: {{MODEL_PATH}} ({{size:.2f}} KB)')
        
        if os.path.exists(TRAINING_LOG):
            with open(TRAINING_LOG, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if 'Epoch' in line or 'Loss' in line:
                        print(line.rstrip())
                        if 'Val Loss' in line:
                            break
    else:
        print('❌ Model not found')
        sys.exit(1)

# Main flow
if check_model():
    print('✅ Model exists')
    response = input('Re-train? (y/N): ').strip().lower()
    if response != 'y':
        results()
        sys.exit(0)

if not check_training():
    install()
    train()

monitor()
results()
"""
    
    # Execute via runpodctl (proxy for MCP tools)
    print("\nExecuting training command...")
    result = subprocess.run(
        ["runpodctl", "exec", "python", pod_id, "-c", command],
        text=True
    )
    
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()

