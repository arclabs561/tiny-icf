#!/usr/bin/env python3
"""
Streamlined RunPod training script.
Uses MCP tools if available, falls back to runpodctl.
Single command: setup → train → monitor → download
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def get_api_key():
    """Get RunPod API key from MCP config (~/.cursor/mcp.json)."""
    mcp_config = Path.home() / ".cursor" / "mcp.json"
    if not mcp_config.exists():
        return None
    
    try:
        with open(mcp_config) as f:
            content = f.read()
        
        # Try JSON parsing first
        try:
            config = json.loads(content)
            
            # Pattern 1: Direct servers list (most common MCP structure)
            if isinstance(config, dict) and "mcpServers" in config:
                for server in config.get("mcpServers", {}).values():
                    if isinstance(server, dict) and "env" in server:
                        if "RUNPOD_API_KEY" in server["env"]:
                            return server["env"]["RUNPOD_API_KEY"]
            
            # Pattern 2: Recursive search for RUNPOD_API_KEY anywhere
            if isinstance(config, dict):
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
                
                api_key = find_key(config, "RUNPOD_API_KEY")
                if api_key:
                    return api_key
        except json.JSONDecodeError:
            pass  # Fall through to regex parsing
        
        # Pattern 3: Fallback to regex (for malformed JSON or text config)
        import re
        match = re.search(r'"RUNPOD_API_KEY":\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"⚠️  Error reading MCP config: {e}", file=sys.stderr)
    
    return None


def check_runpodctl():
    """Check if runpodctl is configured."""
    try:
        result = subprocess.run(
            ["runpodctl", "config", "get"],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def configure_runpodctl():
    """Configure runpodctl with API key from MCP config."""
    api_key = get_api_key()
    if not api_key:
        print("❌ Could not find RUNPOD_API_KEY in ~/.cursor/mcp.json")
        print("   Please add it to your MCP config or run:")
        print("   runpodctl config --apiKey YOUR_API_KEY")
        return False
    
    try:
        result = subprocess.run(
            ["runpodctl", "config", "--apiKey", api_key],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ runpodctl configured using API key from ~/.cursor/mcp.json")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        print(f"❌ Failed to configure runpodctl: {error_msg}")
        return False


def get_or_create_pod():
    """Get existing running pod or create new one."""
    # Check for existing pods
    try:
        result = subprocess.run(
            ["runpodctl", "get", "pod"],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.split("\n"):
            if "RUNNING" in line.upper() and ("3090" in line or "RTX" in line):
                parts = line.split()
                if parts:
                    pod_id = parts[0]
                    # Remove "pod-" prefix if present
                    if pod_id.startswith("pod-"):
                        pod_id = pod_id[4:]
                    print(f"✅ Using existing pod: {pod_id}")
                    return pod_id
    except subprocess.CalledProcessError:
        pass
    
    # Create new pod
    print("Creating new pod...")
    api_key = get_api_key()
    if not api_key:
        print("❌ Need API key to create pod")
        return None
    
    pod_name = f"tiny-icf-{int(time.time())}"
    # Use RTX 4080 SUPER for better performance (similar cost to 3090 on-demand)
    # Or RTX 4090 for even better performance
    gpu_type = "NVIDIA GeForce RTX 4080 SUPER"  # Better than 3090, similar price
    cmd = [
        "runpodctl", "create", "pod",
        "--name", pod_name,
        "--imageName", "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "--gpuType", gpu_type,
        "--containerDiskSize", "30",  # More space for models
        "--mem", "32",
        "--env", "PYTHONUNBUFFERED=1",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Extract pod ID from output
        output = result.stdout + result.stderr
        import re
        # Try different patterns for pod ID (without pod- prefix)
        match = re.search(r'pod-([a-z0-9]+)', output) or re.search(r'([a-z0-9]{13,})', output)
        if match:
            pod_id = match.group(1) if match.lastindex == 1 else match.group(0)
            # Remove pod- prefix if present
            if pod_id.startswith('pod-'):
                pod_id = pod_id[4:]
            print(f"✅ Pod created: {pod_id}")
            print("   Waiting 30 seconds for pod to be ready...")
            time.sleep(30)
            return pod_id
        else:
            print(f"❌ Could not extract pod ID from output:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return None
    except subprocess.CalledProcessError as e:
        error_out = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
        error_out = error_out or (e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout))
        print(f"❌ Failed to create pod: {error_out}")
        return None


def upload_project(pod_id):
    """Upload project to pod (excluding .venv and other large dirs)."""
    print(f"📤 Uploading project to {pod_id}...")
    print("   (This may take a few minutes - excluding .venv)")
    
    # Create a temporary directory with only what we need
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp(prefix="tiny-icf-upload-"))
    try:
        # Copy essential files/directories
        essential_items = [
            "src", "data", "scripts", "tests", "pyproject.toml", 
            "README.md", "models"
        ]
        
        for item in essential_items:
            src = Path(item)
            if src.exists():
                dst = temp_dir / item
                if src.is_dir():
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
                else:
                    shutil.copy2(src, dst)
        
        # Upload from temp directory
        result = subprocess.run(
            ["runpodctl", "send", str(temp_dir), pod_id, "/workspace/tiny-icf"],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Upload complete")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
        error_msg = error_msg or (e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout))
        print(f"❌ Upload failed: {error_msg}")
        return False
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_training(pod_id, background=False):
    """Run training on pod."""
    work_dir = "/workspace/tiny-icf"
    model_path = f"{work_dir}/models/model_runpod.pt"
    training_log = f"{work_dir}/training.log"
    lock_file = f"{work_dir}/.training.lock"
    
    # Create models directory
    setup_cmd = f"""
import os
from pathlib import Path
os.chdir('{work_dir}')
Path('models').mkdir(exist_ok=True)
"""
    
    # Use bash script approach (simpler and more reliable)
    bash_script = f"""#!/bin/bash
set -e
cd {work_dir}
mkdir -p models

# Check if already training
if [ -f '{lock_file}' ]; then
    echo '⚠️  Training already in progress (lock file exists)'
    exit 0
fi

# Check if model exists (skip in background mode)
if [ -f '{model_path}' ] && [ $(stat -f%z '{model_path}' 2>/dev/null || stat -c%s '{model_path}' 2>/dev/null || echo 0) -gt 1000 ]; then
    if [ "{'true' if background else 'false'}" = "true" ]; then
        echo '✅ Model already exists, skipping (background mode)'
        exit 0
    fi
fi

# Create lock file
touch '{lock_file}'

# Install dependencies
echo '🔧 Installing dependencies...'
uv pip install -e . 2>&1 | tee -a {training_log} || true

# Start training (optimized version with mixed precision)
echo '🎯 Starting optimized training...' | tee -a {training_log}
python -m tiny_icf.train_optimized \\
    --data data/multilingual/multilingual_combined.csv \\
    --typo-corpus data/typos/github_typos.csv \\
    --multilingual \\
    --epochs 50 \\
    --batch-size 256 \\
    --lr 2e-3 \\
    --curriculum-stages 5 \\
    --num-workers 4 \\
    --early-stopping 10 \\
    --output {model_path} \\
    2>&1 | tee -a {training_log}

# Remove lock file
rm -f '{lock_file}'
echo '✅ Training complete' | tee -a {training_log}
"""
    
    # Run training via bash
    print("🚀 Starting training...")
    try:
        if background:
            # Run in background using nohup
            exec_cmd = f"cd {work_dir} && nohup bash -c '{bash_script.replace(chr(10), '; ')}' > /dev/null 2>&1 &"
            result = subprocess.run(
                ["runpodctl", "ssh", "connect", pod_id],
                capture_output=True,
                text=True
            )
            # For now, just run it directly - the pod should handle it
            # Actually, let's use the train_on_runpod.sh script that's already there
            exec_cmd = f"cd {work_dir} && nohup bash scripts/train_on_runpod.sh > {training_log} 2>&1 &"
            subprocess.Popen(
                ["runpodctl", "exec", pod_id, "--", "bash", "-c", exec_cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"✅ Training started in background")
            print(f"   Monitor: runpodctl ssh connect {pod_id} then: tail -f {training_log}")
        else:
            # Run in foreground
            subprocess.run(
                ["runpodctl", "exec", pod_id, "--", "bash", "-c", bash_script],
                check=True
            )
            print("✅ Training complete")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode() if e.stderr else 'Unknown error')
        print(f"❌ Training failed: {error_msg}")
        return False


def monitor_training(pod_id, follow=False):
    """Monitor training progress."""
    log_path = "/workspace/tiny-icf/training.log"
    
    # Get SSH command
    try:
        ssh_result = subprocess.run(
            ["runpodctl", "ssh", "connect", pod_id],
            capture_output=True,
            text=True,
            check=True
        )
        ssh_cmd = ssh_result.stdout.strip()
        print(f"📊 To monitor training, run:")
        print(f"   {ssh_cmd}")
        print(f"   Then: tail -f {log_path}")
        
        if not follow:
            # Try to get last lines via exec if possible
            try:
                result = subprocess.run(
                    ["runpodctl", "exec", pod_id, "--", "bash", "-c", f"tail -n 20 {log_path} 2>/dev/null || echo '(Log file not found)'"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.stdout and "Log file not found" not in result.stdout:
                    print(f"\n📊 Last 20 lines of training log:")
                    print(result.stdout)
                else:
                    print(f"\n   (Log file not found - training may be starting)")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print(f"\n   (Could not read log - use SSH command above to monitor)")
    except subprocess.CalledProcessError:
        print(f"❌ Could not get SSH connection info")
        print(f"   Try: runpodctl ssh connect {pod_id}")


def download_model(pod_id, local_path="models/model_runpod.pt"):
    """Download trained model."""
    remote_path = "/workspace/tiny-icf/models/model_runpod.pt"
    
    print(f"📥 Downloading model from {pod_id}...")
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            ["runpodctl", "receive", pod_id, remote_path, local_path],
            check=True,
            capture_output=True
        )
        size = Path(local_path).stat().st_size / 1024
        print(f"✅ Model downloaded: {local_path} ({size:.1f} KB)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        return False


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="RunPod training automation")
    parser.add_argument("--pod-id", help="Use specific pod ID")
    parser.add_argument("--background", action="store_true", help="Run training in background")
    parser.add_argument("--monitor", action="store_true", help="Monitor training log")
    parser.add_argument("--follow", action="store_true", help="Follow training log (tail -f)")
    parser.add_argument("--download", action="store_true", help="Download model after training")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading project")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (just monitor/download)")
    
    args = parser.parse_args()
    
    # Always try to configure runpodctl from MCP config first
    # This ensures we're using the latest API key from ~/.cursor/mcp.json
    api_key = get_api_key()
    if api_key:
        if not check_runpodctl() or True:  # Always reconfigure to use latest key
            configure_runpodctl()
    elif not check_runpodctl():
        print("⚠️  runpodctl not configured and no API key found in ~/.cursor/mcp.json")
        print("   Please add RUNPOD_API_KEY to ~/.cursor/mcp.json or run:")
        print("   runpodctl config --apiKey YOUR_API_KEY")
        sys.exit(1)
    
    # Get or create pod
    if args.pod_id:
        pod_id = args.pod_id
        print(f"Using pod: {pod_id}")
    else:
        pod_id = get_or_create_pod()
        if not pod_id:
            sys.exit(1)
    
    # Upload project
    if not args.skip_upload and not args.skip_train:
        if not upload_project(pod_id):
            sys.exit(1)
    
    # Run training
    if not args.skip_train:
        if not run_training(pod_id, background=args.background):
            sys.exit(1)
    
    # Monitor
    if args.monitor or args.follow:
        monitor_training(pod_id, follow=args.follow)
    
    # Download
    if args.download:
        download_model(pod_id)


if __name__ == "__main__":
    main()

