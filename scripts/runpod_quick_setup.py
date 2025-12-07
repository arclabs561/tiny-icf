#!/usr/bin/env -S uv run
"""
Quick RunPod setup using API key from MCP config.
Creates pod, uploads files, and starts training.
"""

import subprocess
import sys
import time
import re
from pathlib import Path

def get_api_key():
    """Extract RunPod API key from MCP config."""
    mcp_config = Path.home() / '.cursor' / 'mcp.json'
    with open(mcp_config) as f:
        content = f.read()
    match = re.search(r'"RUNPOD_API_KEY":\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    raise ValueError("Could not find RUNPOD_API_KEY in MCP config")

def run_command(cmd, check=True):
    """Run a command and return output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip(), result.stderr.strip()

def main():
    print("🚀 RunPod Quick Setup")
    print("=" * 60)
    
    # Get API key
    api_key = get_api_key()
    print(f"✅ Found API key: {api_key[:20]}...")
    
    # Configure runpodctl
    print("\n📝 Configuring runpodctl...")
    run_command(["runpodctl", "config", "--apiKey", api_key])
    print("✅ runpodctl configured")
    
    # Check existing pods
    print("\n📋 Checking existing pods...")
    pods_out, _ = run_command(["runpodctl", "get", "pod"], check=False)
    if pods_out and "ID" not in pods_out:
        print("No existing pods")
    else:
        print(pods_out)
    
    # Create pod
    pod_name = f"tiny-icf-training-{int(time.time())}"
    print(f"\n🔄 Creating pod: {pod_name}")
    print("This may take 1-2 minutes...")
    
    create_cmd = [
        "runpodctl", "create", "pod",
        "--name", pod_name,
        "--imageName", "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "--gpuType", "NVIDIA GeForce RTX 3090",
        "--volumeSize", "50",
        "--containerDiskSize", "20",
        "--startSSH",
        "--env", "PYTHONUNBUFFERED=1",
    ]
    
    output, error = run_command(create_cmd, check=False)
    print(output)
    if error:
        print(f"Error output: {error}")
    
    # Extract pod ID
    pod_id_match = re.search(r'pod-[a-zA-Z0-9]+', output + error)
    if pod_id_match:
        pod_id = pod_id_match.group(0)
        print(f"\n✅ Pod created: {pod_id}")
        
        print("\n⏳ Waiting for pod to be ready (30 seconds)...")
        time.sleep(30)
        
        print("\n📤 Uploading project files...")
        run_command(["runpodctl", "send", pod_id, ".", "/workspace/tiny-icf"], check=False)
        
        print("\n✅ Setup complete!")
        print(f"\nPod ID: {pod_id}")
        print("\nNext steps:")
        print(f"1. SSH: runpodctl ssh {pod_id}")
        print("2. Install: cd /workspace/tiny-icf && uv pip install -e .")
        print("3. Train: python -m tiny_icf.train_curriculum \\")
        print("     --data data/multilingual/multilingual_combined.csv \\")
        print("     --typo-corpus data/typos/github_typos.csv \\")
        print("     --multilingual --epochs 50 --batch-size 128 \\")
        print("     --output models/model_runpod.pt")
    else:
        print("\n⚠️  Could not extract pod ID from output")
        print("Check pod status: runpodctl get pod")

if __name__ == "__main__":
    main()

