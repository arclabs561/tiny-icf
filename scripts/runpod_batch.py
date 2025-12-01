#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Non-interactive RunPod batch training orchestration.
PEP 723 script - run with: uv run scripts/runpod_batch.py
"""

import subprocess
import sys
import time
from pathlib import Path

# Import shared utilities
_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))
from runpod_utils import get_api_key, configure_runpodctl, extract_pod_id


def get_or_create_pod(gpu_type="NVIDIA GeForce RTX 4080 SUPER", use_api=False, volume_path="/workspace"):
    """Get existing running pod or create new one.
    
    Args:
        gpu_type: GPU type ID
        use_api: If True, use GraphQL API. If False, use runpodctl (default).
        volume_path: Volume mount path for persistent storage
    """
    # Check for existing pods
    try:
        if use_api:
            # Use API to list pods
            try:
                from scripts.runpod_api_client import RunPodAPIClient
                client = RunPodAPIClient()
                pods = client.list_pods()
                for pod in pods:
                    if pod.get('desiredStatus') == 'RUNNING':
                        pod_id = pod['id']
                        print(f"✅ Using existing pod: {pod_id}")
                        return pod_id
            except Exception as e:
                print(f"⚠️  API check failed: {e}, falling back to runpodctl")
        
        # Fallback to runpodctl
        result = subprocess.run(
            ["runpodctl", "get", "pod"],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.split("\n"):
            if "RUNNING" in line.upper() and ("4080" in line or "4090" in line or "3090" in line):
                parts = line.split()
                if parts:
                    pod_id = parts[0]
                    if pod_id.startswith("pod-"):
                        pod_id = pod_id[4:]
                    print(f"✅ Using existing pod: {pod_id}")
                    return pod_id
    except subprocess.CalledProcessError:
        pass
    
    # Create new pod
    print(f"Creating new pod with {gpu_type}...")
    pod_name = f"tiny-icf-batch-{int(time.time())}"
    
    if use_api:
        # Use GraphQL API
        try:
            from scripts.runpod_api_client import RunPodAPIClient
            client = RunPodAPIClient()
            pod = client.create_pod(
                name=pod_name,
                gpu_type_id=gpu_type,
                image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
                gpu_count=1,
                volume_in_gb=50,
                container_disk_in_gb=30,
                min_vcpu_count=2,
                min_memory_in_gb=15,
                volume_mount_path=volume_path,
                env=[{"key": "PYTHONUNBUFFERED", "value": "1"}],
            )
            pod_id = pod['id']
            print(f"✅ Pod created via API: {pod_id}")
            print("   Waiting 30 seconds for pod to be ready...")
            time.sleep(30)
            return pod_id
        except Exception as e:
            print(f"⚠️  API creation failed: {e}, falling back to runpodctl")
    
    # Fallback to runpodctl
    cmd = [
        "runpodctl", "create", "pod",
        "--name", pod_name,
        "--imageName", "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "--gpuType", gpu_type,
        "--containerDiskSize", "30",
        "--mem", "32",
        "--volumePath", volume_path,  # Persistent volume for checkpoints
        "--env", "PYTHONUNBUFFERED=1",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout + result.stderr
        
        pod_id = extract_pod_id(output)
        if pod_id:
            print(f"✅ Pod created via runpodctl: {pod_id}")
            print("   Waiting 30 seconds for pod to be ready...")
            time.sleep(30)
            return pod_id
        else:
            print(f"❌ Could not extract pod ID from: {output}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create pod: {e.stderr}")
        return None


def upload_project(pod_id):
    """Upload project to pod (excluding .venv)."""
    print(f"📤 Uploading project to {pod_id}...")
    
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp(prefix="tiny-icf-upload-"))
    try:
        essential_items = ["src", "data", "scripts", "pyproject.toml", "README.md"]
        for item in essential_items:
            src = Path(item)
            if src.exists():
                dst = temp_dir / item
                if src.is_dir():
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
                else:
                    shutil.copy2(src, dst)
        
        result = subprocess.run(
            ["runpodctl", "send", str(temp_dir), pod_id, "/workspace/tiny-icf"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("✅ Upload complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed: {e.stderr}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_training(pod_id, background=False):
    """Run non-interactive training on pod using runpodctl exec.
    
    Falls back to auto-start script method if exec fails.
    
    Args:
        pod_id: Pod identifier
        background: If True, run in background (nohup). If False, wait for completion.
    
    Returns:
        bool: True if training started successfully (or completed if not background)
    """
    print("🚀 Starting batch training...")
    
    # Training command - runs and exits cleanly
    train_cmd = """
cd /workspace/tiny-icf
uv run scripts/train_batch.py \\
    --data data/multilingual/multilingual_combined.csv \\
    --typo-corpus data/typos/github_typos.csv \\
    --output-dir /workspace/tiny-icf/models \\
    --epochs 50 \\
    --batch-size 256 \\
    --devices 1 \\
    --precision 16-mixed \\
    2>&1 | tee /workspace/tiny-icf/training.log
echo "Training exit code: $?"
"""
    
    if background:
        # Run in background using nohup
        train_cmd = f"""
cd /workspace/tiny-icf
nohup bash -c '{train_cmd.replace(chr(10), "; ")}' > /dev/null 2>&1 &
echo $! > /workspace/tiny-icf/training.pid
echo "Training started in background with PID: $(cat /workspace/tiny-icf/training.pid)"
"""
    
    # Try runpodctl exec first
    try:
        print("   Method 1: runpodctl exec (direct)")
        result = subprocess.run(
            ["runpodctl", "exec", pod_id, "--", "bash", "-c", train_cmd],
            capture_output=True,
            text=True,
            timeout=3600 if not background else 30,
        )
        
        if background:
            if result.returncode == 0:
                print("✅ Training started in background")
                print(f"   PID: {result.stdout.strip() if result.stdout.strip() else 'check training.pid'}")
                print(f"\n📊 Monitor training:")
                print(f"   runpodctl exec {pod_id} -- tail -f /workspace/tiny-icf/training.log")
                return True
            else:
                print(f"⚠️  Method 1 failed, trying fallback...")
        else:
            if result.returncode == 0:
                print("✅ Training completed successfully")
                if "Training exit code: 0" in result.stdout:
                    return True
                else:
                    print("⚠️  Training may have failed - check logs")
                    return False
            else:
                print(f"⚠️  Method 1 failed, trying fallback...")
                
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
        print(f"⚠️  Method 1 failed ({type(e).__name__}), trying fallback...")
    
    # Fallback: Use auto-start script method (more reliable)
    print("   Method 2: Auto-start script (fallback)")
    try:
        # Import and use auto-start script
        import importlib.util
        auto_start_path = Path(__file__).parent / "runpod_auto_start.py"
        if auto_start_path.exists():
            # Create auto-start script on pod
            result = subprocess.run(
                [sys.executable, str(auto_start_path), pod_id, "--create-only"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                # Trigger it
                result = subprocess.run(
                    ["runpodctl", "exec", pod_id, "--", "bash", "/workspace/tiny-icf/auto_start_training.sh"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0 or "Training started" in result.stdout:
                    print("✅ Training started via auto-start script")
                    print(f"\n📊 Monitor training:")
                    print(f"   runpodctl exec {pod_id} -- tail -f /workspace/tiny-icf/training.log")
                    return True
        
        # Last resort: manual instructions
        print("⚠️  Both methods had issues")
        print(f"\n💡 Manual start:")
        print(f"   1. runpodctl ssh connect {pod_id}")
        print(f"   2. cd /workspace/tiny-icf")
        print(f"   3. bash auto_start_training.sh  # (if exists)")
        print(f"   OR: uv run scripts/train_batch.py --epochs 50 --batch-size 256")
        return False
        
    except Exception as e:
        print(f"❌ Fallback method also failed: {e}")
        return False


def check_training_status(pod_id):
    """Check if training is running or completed."""
    print(f"📊 Checking training status on {pod_id}...")
    
    # Check if process is running
    result = subprocess.run(
        ["runpodctl", "exec", pod_id, "--", "pgrep", "-f", "train_batch"],
        capture_output=True,
    )
    is_running = result.returncode == 0
    
    # Check for final model
    result = subprocess.run(
        ["runpodctl", "exec", pod_id, "--", "test", "-f", "/workspace/tiny-icf/models/model_final.pt"],
        capture_output=True,
    )
    is_complete = result.returncode == 0
    
    # Get last few lines of log
    result = subprocess.run(
        ["runpodctl", "exec", pod_id, "--", "tail", "-n", "5", "/workspace/tiny-icf/training.log"],
        capture_output=True,
        text=True,
    )
    log_tail = result.stdout if result.returncode == 0 else ""
    
    if is_complete:
        print("✅ Training completed (model_final.pt exists)")
        return "complete"
    elif is_running:
        print("🔄 Training is running")
        if log_tail:
            print(f"   Last log: {log_tail.strip().split(chr(10))[-1]}")
        return "running"
    else:
        print("❓ Training status unknown")
        if log_tail:
            print(f"   Last log: {log_tail.strip().split(chr(10))[-1]}")
        return "unknown"


def download_results(pod_id, remote_path="/workspace/tiny-icf/models", local_path="models"):
    """Download trained models."""
    print(f"📥 Downloading results from {pod_id}...")
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            ["runpodctl", "receive", pod_id, remote_path, local_path],
            check=True,
            capture_output=True,
        )
        print("✅ Results downloaded")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e.stderr}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RunPod batch training (PEP 723)")
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4080 SUPER", help="GPU type")
    parser.add_argument("--pod-id", help="Use existing pod ID")
    parser.add_argument("--upload-only", action="store_true", help="Only upload, don't train")
    parser.add_argument("--download-only", action="store_true", help="Only download results")
    parser.add_argument("--status", action="store_true", help="Check training status")
    parser.add_argument("--background", action="store_true", help="Run training in background")
    parser.add_argument("--stop-after", action="store_true", help="Stop pod after download")
    parser.add_argument("--use-api", action="store_true", help="Use GraphQL API for pod management (instead of runpodctl)")
    parser.add_argument("--volume-path", default="/workspace", help="Volume mount path")
    
    args = parser.parse_args()
    
    # Configure runpodctl (still needed for file transfer and command execution)
    if not configure_runpodctl():
        print("⚠️  runpodctl configuration failed, but continuing...")
        print("   Some operations may fail without API key")
    
    # Get or create pod
    if args.pod_id:
        pod_id = args.pod_id
        if pod_id.startswith("pod-"):
            pod_id = pod_id[4:]
        print(f"Using pod: {pod_id}")
    else:
        pod_id = get_or_create_pod(
            gpu_type=args.gpu_type,
            use_api=args.use_api,
            volume_path=args.volume_path,
        )
        if not pod_id:
            sys.exit(1)
    
    # Status check
    if args.status:
        check_training_status(pod_id)
        return
    
    # Download only
    if args.download_only:
        download_results(pod_id)
        if args.stop_after:
            subprocess.run(["runpodctl", "stop", "pod", pod_id], check=False)
        return
    
    # Upload
    if not upload_project(pod_id):
        sys.exit(1)
    
    if args.upload_only:
        print(f"\n✅ Upload complete. Pod ready: {pod_id}")
        print(f"   Run training: uv run scripts/runpod_batch.py --pod-id {pod_id}")
        return
    
    # Run training
    if not run_training(pod_id, background=args.background):
        sys.exit(1)
    
    if args.background:
        print(f"\n⏳ Training running in background on pod {pod_id}")
        print(f"   Check status: uv run scripts/runpod_batch.py --status --pod-id {pod_id}")
        print(f"   Download when complete: uv run scripts/runpod_batch.py --download-only --pod-id {pod_id}")
    else:
        # Training completed synchronously
        print(f"\n✅ Training completed on pod {pod_id}")
        print(f"   Download results: uv run scripts/runpod_batch.py --download-only --pod-id {pod_id}")


if __name__ == "__main__":
    main()

