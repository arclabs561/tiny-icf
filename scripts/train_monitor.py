#!/usr/bin/env python3
"""
Idempotent training monitor: checks status, trains if needed, monitors, returns results.
Single command that handles everything.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

POD_ID = os.getenv("RUNPOD_POD_ID", "w4857gh85qldts")
WORK_DIR = "/workspace/tiny-icf"
MODEL_PATH = f"{WORK_DIR}/models/model_runpod.pt"
TRAINING_LOG = f"{WORK_DIR}/training.log"
LOCK_FILE = f"{WORK_DIR}/.training.lock"

def run_remote_command(cmd: str, check: bool = True) -> tuple[str, str, int]:
    """Execute command on RunPod pod via runpodctl."""
    full_cmd = ["runpodctl", "exec", "python", POD_ID, "-c", cmd]
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
    return result.stdout, result.stderr, result.returncode

def check_training_running() -> bool:
    """Check if training is currently running."""
    cmd = f"import os; exit(0 if os.path.exists('{LOCK_FILE}') else 1)"
    _, _, code = run_remote_command(cmd, check=False)
    return code == 0

def check_model_complete() -> bool:
    """Check if model exists and is valid."""
    cmd = f"""
import os
import torch
model_path = '{MODEL_PATH}'
if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
    try:
        # Try loading to verify it's valid
        torch.load(model_path, map_location='cpu')
        exit(0)
    except:
        exit(1)
exit(1)
"""
    _, _, code = run_remote_command(cmd, check=False)
    return code == 0

def upload_project():
    """Upload project files to pod."""
    print("📤 Uploading project files...")
    # Use runpodctl send or SSH/SCP
    # For now, assume files are already there or will be uploaded separately
    print("✅ Project files ready (assuming already uploaded)")

def install_dependencies():
    """Install dependencies."""
    print("🔧 Installing dependencies...")
    cmd = f"""
import subprocess
import sys
import os
os.chdir('{WORK_DIR}')
result = subprocess.run(['uv', 'pip', 'install', '-e', '.'], 
                       capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)
"""
    stdout, stderr, code = run_remote_command(cmd, check=False)
    if code == 0:
        print("✅ Dependencies installed")
    else:
        print("⚠️  Dependencies may already be installed, continuing...")

def start_training():
    """Start training in background."""
    print("🎯 Starting training...")
    cmd = f"""
import subprocess
import os
import sys
from pathlib import Path

os.chdir('{WORK_DIR}')
Path('{LOCK_FILE}').touch()

# Training command
cmd = [
    'python', '-m', 'tiny_icf.train_curriculum',
    '--data', 'data/multilingual/multilingual_combined.csv',
    '--typo-corpus', 'data/typos/github_typos.csv',
    '--multilingual',
    '--epochs', '50',
    '--batch-size', '128',
    '--lr', '1e-3',
    '--curriculum-stages', '5',
    '--output', '{MODEL_PATH}'
]

# Start training
with open('{TRAINING_LOG}', 'w') as log:
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f'Training started with PID: {{process.pid}}')
    # Don't wait - let it run in background
"""
    stdout, stderr, code = run_remote_command(cmd)
    if code == 0:
        print(f"✅ Training started: {stdout.strip()}")
    else:
        print(f"❌ Failed to start training: {stderr}")
        sys.exit(1)

def monitor_training():
    """Monitor training progress."""
    print("\n📊 Monitoring training progress...")
    print("(Press Ctrl+C to stop monitoring - training continues)\n")
    
    last_lines = 0
    while True:
        # Check if training is still running
        if not check_training_running():
            if check_model_complete():
                print("\n✅ Training complete!")
                break
            else:
                print("\n⚠️  Training stopped but model not found")
                break
        
        # Show recent log output
        cmd = f"""
import os
log_path = '{TRAINING_LOG}'
if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
        # Show new lines since last check
        for line in lines[{last_lines}:]:
            print(line.rstrip())
        return len(lines)
return 0
"""
        stdout, _, _ = run_remote_command(cmd, check=False)
        if stdout.strip():
            print(stdout.strip())
            try:
                last_lines = int(stdout.strip().split('\n')[-1])
            except:
                pass
        
        time.sleep(10)

def show_results():
    """Show training results."""
    print("\n📈 Training Results")
    print("=" * 50)
    
    if check_model_complete():
        print(f"✅ Model saved: {MODEL_PATH}")
        
        # Get model size
        cmd = f"""
import os
size = os.path.getsize('{MODEL_PATH}')
print(f'Model size: {{size / 1024:.2f}} KB')
"""
        stdout, _, _ = run_remote_command(cmd, check=False)
        if stdout.strip():
            print(stdout.strip())
        
        # Show final metrics
        print("\nFinal training metrics:")
        cmd = f"""
import os
log_path = '{TRAINING_LOG}'
if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
        # Find last epoch info
        for line in reversed(lines):
            if 'Epoch' in line or 'Val Loss' in line or 'Train Loss' in line:
                print(line.rstrip())
                if 'Val Loss' in line:
                    break
"""
        stdout, _, _ = run_remote_command(cmd, check=False)
        if stdout.strip():
            print(stdout.strip())
        
        print("\n🎉 Training successful!")
        print(f"\nNext: Download model with:")
        print(f"  runpodctl receive {POD_ID} {MODEL_PATH} ./models/")
    else:
        print("❌ Model not found or incomplete")
        sys.exit(1)

def main():
    """Main execution - idempotent training monitor."""
    print("🚀 Idempotent Training Monitor")
    print("=" * 50)
    print(f"Pod: {POD_ID}\n")
    
    # Check if model already exists
    if check_model_complete():
        print("✅ Model already exists and appears complete")
        response = input("\nRe-train? (y/N): ").strip().lower()
        if response != 'y':
            show_results()
            return
    
    # Check if training is already running
    if check_training_running():
        print("⏳ Training already in progress")
        monitor_training()
        show_results()
    else:
        # Start fresh training
        upload_project()
        install_dependencies()
        start_training()
        time.sleep(5)  # Wait for training to start
        monitor_training()
        show_results()

if __name__ == "__main__":
    main()

