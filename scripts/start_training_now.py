#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Start training immediately with optimal settings.
This script ensures training starts correctly on the pod.
"""
import sys
import subprocess
from pathlib import Path


def start_training(pod_id: str):
    """Start training with optimal configuration."""
    print("🚀 Starting Model Training")
    print("=" * 60)
    print(f"Pod: {pod_id}")
    print("\n📋 Training Configuration:")
    print("   - Epochs: 50")
    print("   - Batch Size: 256")
    print("   - Learning Rate: 2e-3")
    print("   - Precision: 16-mixed (AMP)")
    print("   - Devices: 1 GPU")
    print("   - Early Stopping: 10 epochs patience")
    print("   - Curriculum Learning: 5 stages")
    print("=" * 60)
    
    # First, ensure auto-start script exists
    print("\n1️⃣  Setting up auto-start script...")
    result = subprocess.run(
        [sys.executable, "scripts/runpod_auto_start.py", pod_id, "--create-only"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print("⚠️  Script creation had issues, but continuing...")
    
    # Create optimized training command
    print("\n2️⃣  Starting training...")
    
    # Use a more reliable method: create a Python script that runs training
    training_script = """#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/workspace/tiny-icf/src').absolute()))

# Change to project directory
import os
os.chdir('/workspace/tiny-icf')

# Run training
from scripts.train_batch import main
sys.argv = [
    'train_batch.py',
    '--data', 'data/multilingual/multilingual_combined.csv',
    '--typo-corpus', 'data/typos/github_typos.csv',
    '--output-dir', '/workspace/tiny-icf/models',
    '--epochs', '50',
    '--batch-size', '256',
    '--lr', '2e-3',
    '--devices', '1',
    '--precision', '16-mixed',
    '--num-workers', '4',
    '--curriculum-stages', '5',
    '--early-stopping-patience', '10',
    '--multilingual',
    '--include-symbols',
    '--include-emojis',
]
main()
"""
    
    # Write script locally
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(training_script)
        script_path = f.name
    
    try:
        import os
        os.chmod(script_path, 0o755)
        
        # Upload script
        print("   Uploading training script...")
        subprocess.run(
            ["runpodctl", "send", script_path, pod_id, "/workspace/tiny-icf/start_training_direct.py"],
            check=True,
            capture_output=True,
        )
        
        # Make executable
        subprocess.run(
            ["runpodctl", "exec", pod_id, "--", "chmod", "+x", "/workspace/tiny-icf/start_training_direct.py"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        
        # Start training in background using nohup
        print("   Launching training in background...")
        start_cmd = """cd /workspace/tiny-icf && nohup python3 start_training_direct.py > training.log 2>&1 & echo $! > training.pid && cat training.pid"""
        
        result = subprocess.run(
            ["runpodctl", "exec", pod_id, "--", "bash", "-c", start_cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pid = result.stdout.strip()
            print(f"✅ Training started with PID: {pid}")
            print(f"\n📊 Monitor training:")
            print(f"   uv run scripts/runpod_auto_start.py {pod_id} --status")
            print(f"   runpodctl exec {pod_id} -- tail -f /workspace/tiny-icf/training.log")
            print(f"\n💡 Check GPU utilization:")
            print(f"   uv run scripts/runpod_monitor.py {pod_id} --interval 30")
            return True
        else:
            print("⚠️  Training start had issues, but may have started")
            print("   Check manually: runpodctl ssh connect {pod_id}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            os.unlink(script_path)
        except:
            pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Start training immediately")
    parser.add_argument("pod_id", nargs="?", default="9lj0lizlogeftc", help="Pod ID")
    
    args = parser.parse_args()
    
    start_training(args.pod_id)


if __name__ == "__main__":
    main()
