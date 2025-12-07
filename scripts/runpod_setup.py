#!/usr/bin/env -S uv run
"""
RunPod setup script for training Universal ICF model.
Prepares the environment and training configuration for RunPod GPU instances.
"""

import json
import subprocess
import sys
from pathlib import Path

def check_runpodctl():
    """Check if runpodctl is available."""
    try:
        result = subprocess.run(
            ["runpodctl", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ runpodctl found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ runpodctl not found")
        print("Install with: pip install runpodctl")
        return False

def create_runpod_config():
    """Create RunPod pod configuration."""
    config = {
        "name": "tiny-icf-training",
        "image": "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel",
        "gpuTypeId": "NVIDIA RTX 3090",  # or other GPU type
        "cloudType": "ALL",
        "networkVolumeId": None,
        "dockerArgs": "",
        "volumeInGb": 20,
        "containerDiskInGb": 10,
        "minVcpuCount": 2,
        "minMemoryInGb": 16,
        "gpuCount": 1,
        "env": [
            {"key": "DATA_FILE", "value": "/workspace/data/word_frequency.csv"},
            {"key": "EPOCHS", "value": "50"},
            {"key": "BATCH_SIZE", "value": "64"},
            {"key": "LR", "value": "1e-3"},
            {"key": "OUTPUT", "value": "/workspace/models/model_runpod.pt"},
        ],
        "startJupyter": False,
        "startSsh": True,
        "ports": "",
    }
    
    config_path = Path("runpod_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created RunPod config: {config_path}")
    return config_path

def main():
    print("🔧 RunPod Setup for IDF Estimation Training")
    print("=" * 60)
    
    if not check_runpodctl():
        sys.exit(1)
    
    config_path = create_runpod_config()
    
    print("\n📋 Next Steps:")
    print("1. Review runpod_config.json")
    print("2. Create pod: runpodctl pod create -f runpod_config.json")
    print("3. SSH into pod: runpodctl pod ssh <pod-id>")
    print("4. Upload code and data to /workspace")
    print("5. Run: bash scripts/train_on_runpod.sh")
    
    print("\n💡 Or use MCP RunPod tools if available:")
    print("   - Create pod via MCP")
    print("   - Upload files via MCP")
    print("   - Execute training via MCP")

if __name__ == "__main__":
    main()

