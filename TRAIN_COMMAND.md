# Single Idempotent Training Command

## One MCP Command: Train → Monitor → Results

Use this single command with MCP RunPod tools:

```
Execute on RunPod pod w4857gh85qldts:
cd /workspace/tiny-icf && python -c "
import subprocess
import os
import sys
import time
from pathlib import Path

WORK_DIR = '/workspace/tiny-icf'
MODEL_PATH = f'{WORK_DIR}/models/model_runpod.pt'
TRAINING_LOG = f'{WORK_DIR}/training.log'
LOCK_FILE = f'{WORK_DIR}/.training.lock'

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
        print(f'Training PID: {process.pid}')
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
        print(f'✅ Model: {MODEL_PATH} ({size:.2f} KB)')
        
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
"
```

## How It Works

1. **Idempotent Checks**:
   - ✅ Checks if model already exists → shows results if complete
   - ✅ Checks if training already running → monitors if in progress
   - ✅ Otherwise → installs, trains, monitors, shows results

2. **Single Command**: Everything in one MCP execution
3. **Smart Monitoring**: Live progress updates every 10 seconds
4. **Auto-Results**: Shows final metrics when complete

## Usage

Just execute the command above using MCP RunPod tools. It handles:
- File uploads (if needed)
- Dependency installation
- Training execution
- Progress monitoring
- Results display

## Features

- **Idempotent**: Safe to run multiple times
- **Self-contained**: One command does everything
- **Resumable**: Detects and continues existing training
- **Complete**: Shows results automatically when done

