#!/usr/bin/env python3
"""
RunPod training automation script.
Uses runpodctl or MCP tools to create a GPU pod and run training.
"""

import json
import subprocess
import sys
from pathlib import Path

def check_runpodctl():
    """Check if runpodctl is configured."""
    try:
        result = subprocess.run(
            ["runpodctl", "config", "get"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ runpodctl configured")
        return True
    except subprocess.CalledProcessError:
        print("❌ runpodctl not configured")
        print("Run: runpodctl config --apiKey YOUR_API_KEY")
        return False

def create_training_pod():
    """Create a RunPod pod for training."""
    config = {
        "name": "tiny-icf-training",
        "image": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "gpuTypeId": "NVIDIA RTX 3090",
        "cloudType": "ALL",
        "volumeInGb": 50,
        "containerDiskInGb": 20,
        "minVcpuCount": 4,
        "minMemoryInGb": 32,
        "gpuCount": 1,
        "env": [
            {"key": "PYTHONUNBUFFERED", "value": "1"},
        ],
        "startJupyter": False,
        "startSsh": True,
    }
    
    config_path = Path("runpod_training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created pod config: {config_path}")
    return config_path

def main():
    print("🚀 RunPod Training Setup")
    print("=" * 60)
    
    if not check_runpodctl():
        sys.exit(1)
    
    config_path = create_training_pod()
    
    print("\n📋 To create pod and start training:")
    print(f"1. Create pod: runpodctl pod create -f {config_path}")
    print("2. Wait for pod to be ready")
    print("3. Get pod ID: runpodctl pod get")
    print("4. SSH into pod: runpodctl pod ssh <pod-id>")
    print("5. Upload project: runpodctl pod send <pod-id> . /workspace/tiny-icf")
    print("6. Run training: cd /workspace/tiny-icf && bash scripts/train_on_runpod.sh")
    
    print("\n💡 Or use MCP RunPod tools if configured in Cursor")

if __name__ == "__main__":
    main()

