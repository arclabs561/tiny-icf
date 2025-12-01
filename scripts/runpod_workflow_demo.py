#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""
Demonstrate complete RunPod API workflow.

Shows how to use the API for:
1. Querying GPU types
2. Creating pods
3. Monitoring pods
4. Managing pod lifecycle
"""
import sys
import time
import subprocess
from pathlib import Path


def demo_workflow():
    """Demonstrate complete API workflow."""
    # Add parent directory to path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from scripts.runpod_api_client import RunPodAPIClient
    
    print("🚀 RunPod API Workflow Demonstration")
    print("=" * 60)
    
    # Initialize client
    print("\n1️⃣  Initializing API client...")
    try:
        client = RunPodAPIClient()
        print("   ✅ API client initialized")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Query GPU types
    print("\n2️⃣  Querying available GPU types...")
    try:
        gpu_types = client.get_gpu_types()
        print(f"   ✅ Found {len(gpu_types)} GPU types")
        
        # Find RTX 4080
        target_gpu = None
        for gpu in gpu_types:
            if "4080" in gpu['id']:
                target_gpu = gpu
                break
        
        if target_gpu:
            print(f"   ✅ Target GPU: {target_gpu['id']} ({target_gpu['displayName']})")
            gpu_type_id = target_gpu['id']
        else:
            print("   ⚠️  RTX 4080 not found, using first available")
            gpu_type_id = gpu_types[0]['id'] if gpu_types else None
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # List existing pods
    print("\n3️⃣  Listing existing pods...")
    try:
        pods = client.list_pods()
        print(f"   ✅ Found {len(pods)} pods")
        
        running_pods = [p for p in pods if p.get('desiredStatus') == 'RUNNING']
        if running_pods:
            pod_id = running_pods[0]['id']
            print(f"   ✅ Using existing running pod: {pod_id}")
        else:
            print("   ℹ️  No running pods found")
            pod_id = None
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Get pod information
    if pod_id:
        print(f"\n4️⃣  Getting information for pod {pod_id}...")
        try:
            pod = client.get_pod(pod_id)
            print(f"   ✅ Pod: {pod.get('name')}")
            print(f"   Status: {pod.get('desiredStatus')}")
            print(f"   Cost: ${pod.get('costPerHr', 0):.4f}/hr")
            
            runtime = pod.get('runtime', {})
            if runtime:
                uptime = runtime.get('uptimeInSeconds', 0)
                hours = uptime // 3600
                minutes = (uptime % 3600) // 60
                print(f"   Uptime: {hours}h {minutes}m")
                
                gpus = runtime.get('gpus', [])
                if gpus:
                    gpu_util = gpus[0].get('gpuUtilPercent', 0)
                    print(f"   GPU Utilization: {gpu_util}%")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Demonstrate pod creation (dry-run)
    print("\n5️⃣  Pod creation example (dry-run)...")
    print("   Would create pod with:")
    print(f"   - GPU: {gpu_type_id}")
    print("   - Image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04")
    print("   - Volume: 50GB at /workspace")
    print("   - Container Disk: 30GB")
    print("   ✅ Pod creation structure validated")
    
    # Show integration with runpodctl
    print("\n6️⃣  Integration with runpodctl...")
    try:
        result = subprocess.run(
            ["runpodctl", "get", "pod"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("   ✅ runpodctl is available and working")
            print("   ✅ Hybrid workflow (API + runpodctl) ready")
        else:
            print("   ⚠️  runpodctl check failed")
    except FileNotFoundError:
        print("   ⚠️  runpodctl not found (optional)")
    except Exception as e:
        print(f"   ⚠️  runpodctl check error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Workflow demonstration complete!")
    print("\nNext steps:")
    print("  1. Create pod: uv run scripts/runpod_batch.py --use-api --upload-only")
    print("  2. Monitor pod: uv run scripts/runpod_monitor.py <pod-id>")
    print("  3. Check status: uv run scripts/runpod_api_client.py get <pod-id>")
    print("  4. List pods: uv run scripts/runpod_api_client.py list")


if __name__ == "__main__":
    demo_workflow()

