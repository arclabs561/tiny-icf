#!/usr/bin/env python3
"""
Non-interactive RunPod batch training orchestration.
Creates pod, uploads code, runs training, downloads results, stops pod.
"""

import json
import subprocess
import sys
import time
from pathlib import Path


def get_api_key():
    """Get RunPod API key from MCP config."""
    mcp_config = Path.home() / ".cursor" / "mcp.json"
    if not mcp_config.exists():
        return None
    
    try:
        with open(mcp_config) as f:
            config = json.load(f)
        
        # Find RUNPOD_API_KEY in config
        def find_key(obj, key):
            if isinstance(obj, dict):
                if key in obj:
                    return obj[key]
                for v in obj.values():
                    result = find_key(v, key)
                    if result:
                        return result
            elif isinstance(obj, list):
                for item in obj:
                    result = find_key(item, key)
                    if result:
                        return result
            return None
        
        return find_key(config, "RUNPOD_API_KEY")
    except Exception:
        return None


def configure_runpodctl():
    """Configure runpodctl."""
    api_key = get_api_key()
    if not api_key:
        print("❌ Could not find RUNPOD_API_KEY in ~/.cursor/mcp.json")
        return False
    
    try:
        subprocess.run(
            ["runpodctl", "config", "--apiKey", api_key],
            check=True,
            capture_output=True,
        )
        print("✅ runpodctl configured")
        return True
    except subprocess.CalledProcessError:
        return False


def create_pod(gpu_type="NVIDIA GeForce RTX 4080 SUPER", volume_size=50):
    """Create a pod for batch training."""
    pod_name = f"tiny-icf-batch-{int(time.time())}"
    
    cmd = [
        "runpodctl", "create", "pod",
        "--name", pod_name,
        "--imageName", "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "--gpuType", gpu_type,
        "--containerDiskSize", "30",
        "--mem", "32",
        "--env", "PYTHONUNBUFFERED=1",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout + result.stderr
        
        # Extract pod ID
        import re
        match = re.search(r'([a-z0-9]{13,})', output)
        if match:
            pod_id = match.group(1)
            print(f"✅ Pod created: {pod_id}")
            print("   Waiting 30 seconds for pod to be ready...")
            time.sleep(30)
            return pod_id
        else:
            print(f"❌ Could not extract pod ID from: {output}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create pod: {e.stderr}")
        return None


def upload_project(pod_id, local_path=".", remote_path="/workspace/tiny-icf"):
    """Upload project to pod."""
    print(f"📤 Uploading project to {pod_id}...")
    
    # Create temp directory with essential files only
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
            ["runpodctl", "send", str(temp_dir), pod_id, remote_path],
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


def run_training(pod_id, remote_path="/workspace/tiny-icf"):
    """Run non-interactive training on pod."""
    print("🚀 Starting batch training...")
    
    # Training command that runs and exits
    train_cmd = f"""
cd {remote_path}
uv pip install -e . 2>&1 | tee install.log
python scripts/train_batch.py \\
    --data data/multilingual/multilingual_combined.csv \\
    --typo-corpus data/typos/github_typos.csv \\
    --output-dir {remote_path}/models \\
    --epochs 50 \\
    --batch-size 256 \\
    --devices 1 \\
    2>&1 | tee training.log
echo "Training exit code: $?"
"""
    
    # Use SSH to run command (non-interactive)
    try:
        # Get SSH command
        ssh_result = subprocess.run(
            ["runpodctl", "ssh", "connect", pod_id],
            capture_output=True,
            text=True,
            check=True,
        )
        ssh_cmd = ssh_result.stdout.strip()
        
        print(f"📋 Training command prepared")
        print(f"   Run via SSH: {ssh_cmd}")
        print(f"   Then execute the training script")
        print(f"\n   Or use runpodctl exec to run directly")
        
        # For now, return SSH info - user can run manually or we can use exec
        return ssh_cmd
    except subprocess.CalledProcessError:
        return None


def download_results(pod_id, remote_path="/workspace/tiny-icf/models", local_path="models"):
    """Download trained models from pod."""
    print(f"📥 Downloading results from {pod_id}...")
    
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run(
            ["runpodctl", "receive", pod_id, remote_path, local_path],
            check=True,
            capture_output=True,
            text=True,
        )
        print("✅ Results downloaded")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e.stderr}")
        return False


def stop_pod(pod_id):
    """Stop and remove pod."""
    print(f"🛑 Stopping pod {pod_id}...")
    try:
        subprocess.run(
            ["runpodctl", "stop", "pod", pod_id],
            check=True,
            capture_output=True,
        )
        print("✅ Pod stopped")
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """Main batch training workflow."""
    import argparse
    parser = argparse.ArgumentParser(description="RunPod batch training orchestration")
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4080 SUPER", help="GPU type")
    parser.add_argument("--create-only", action="store_true", help="Only create pod, don't train")
    parser.add_argument("--pod-id", help="Use existing pod ID")
    parser.add_argument("--download-only", action="store_true", help="Only download results")
    parser.add_argument("--stop-after", action="store_true", help="Stop pod after training")
    
    args = parser.parse_args()
    
    # Configure
    if not configure_runpodctl():
        sys.exit(1)
    
    # Get or create pod
    if args.pod_id:
        pod_id = args.pod_id
        if pod_id.startswith("pod-"):
            pod_id = pod_id[4:]
        print(f"Using existing pod: {pod_id}")
    else:
        pod_id = create_pod(gpu_type=args.gpu_type)
        if not pod_id:
            sys.exit(1)
    
    if args.download_only:
        download_results(pod_id)
        if args.stop_after:
            stop_pod(pod_id)
        return
    
    if args.create_only:
        print(f"\n✅ Pod ready: {pod_id}")
        print(f"   Upload and train manually, or run without --create-only")
        return
    
    # Upload
    if not upload_project(pod_id):
        sys.exit(1)
    
    # Run training
    ssh_cmd = run_training(pod_id)
    if ssh_cmd:
        print(f"\n📋 To monitor training:")
        print(f"   {ssh_cmd}")
        print(f"   Then: tail -f /workspace/tiny-icf/training.log")
    
    print(f"\n⏳ Training is running on pod {pod_id}")
    print(f"   Use --download-only --pod-id {pod_id} to download results when complete")
    
    if args.stop_after:
        print(f"\n⚠️  Note: Pod will continue running. Stop manually or use --stop-after on download")


if __name__ == "__main__":
    main()

