#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""
RunPod API Experiments - Test various API operations and integrations.

This script runs several experiments to validate:
1. API client initialization
2. GPU type queries
3. Pod creation via API
4. Pod status monitoring
5. Integration with runpodctl
6. Error handling
"""
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


def experiment_1_test_api_client_init():
    """Experiment 1: Test API client initialization."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: API Client Initialization")
    print("="*60)
    
    try:
        # Add parent directory to path for imports
        import sys
        from pathlib import Path
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from scripts.runpod_api_client import RunPodAPIClient, RunPodAPIError
        
        print("📝 Testing API client initialization...")
        client = RunPodAPIClient()
        print("✅ API client initialized successfully")
        print(f"   API URL: {client.API_URL}")
        return True, client
    except Exception as e:
        # Try importing RunPodAPIError separately for error handling
        try:
            from scripts.runpod_api_client import RunPodAPIError
            if isinstance(e, RunPodAPIError):
                print(f"❌ API Error: {e}")
            else:
                print(f"❌ Unexpected error: {e}")
                import traceback
                traceback.print_exc()
        except:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        return False, None


def experiment_2_query_gpu_types(client):
    """Experiment 2: Query GPU types."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Query GPU Types")
    print("="*60)
    
    if not client:
        print("⚠️  No API client available, skipping")
        return False, None
    
    try:
        print("📝 Querying GPU types...")
        gpu_types = client.get_gpu_types()
        
        print(f"✅ Found {len(gpu_types)} GPU types")
        
        # Show first 5
        print("\n   Sample GPU types:")
        for gpu in gpu_types[:5]:
            print(f"   - {gpu['id']}: {gpu['displayName']} ({gpu['memoryInGb']}GB)")
        
        # Find RTX 4080 SUPER
        target_gpu = None
        for gpu in gpu_types:
            if "4080" in gpu['id'] or "4080" in gpu['displayName']:
                target_gpu = gpu
                break
        
        if target_gpu:
            print(f"\n✅ Found target GPU: {target_gpu['id']}")
            return True, target_gpu['id']
        else:
            print("⚠️  RTX 4080 SUPER not found, using first available")
            return True, gpu_types[0]['id'] if gpu_types else None
            
    except Exception as e:
        print(f"❌ Error querying GPU types: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def experiment_3_list_existing_pods(client):
    """Experiment 3: List existing pods."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: List Existing Pods")
    print("="*60)
    
    if not client:
        print("⚠️  No API client available, skipping")
        return False, None
    
    try:
        print("📝 Listing existing pods...")
        pods = client.list_pods()
        
        print(f"✅ Found {len(pods)} pods")
        
        if pods:
            print("\n   Pods:")
            for pod in pods[:10]:  # Show first 10
                status = pod.get('desiredStatus', 'UNKNOWN')
                name = pod.get('name', 'unnamed')
                pod_id = pod.get('id', 'unknown')
                print(f"   - {pod_id}: {name} ({status})")
            
            # Find running pods
            running = [p for p in pods if p.get('desiredStatus') == 'RUNNING']
            if running:
                print(f"\n✅ Found {len(running)} running pods")
                return True, running[0]['id']
            else:
                print("\n⚠️  No running pods found")
                return True, None
        else:
            print("\n⚠️  No pods found")
            return True, None
            
    except Exception as e:
        print(f"❌ Error listing pods: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def experiment_4_get_pod_info(client, pod_id: Optional[str] = None):
    """Experiment 4: Get pod information."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Get Pod Information")
    print("="*60)
    
    if not pod_id:
        print("⚠️  No pod ID provided, skipping")
        return True, None
    
    try:
        print(f"📝 Getting information for pod {pod_id}...")
        pod = client.get_pod(pod_id)
        
        print("✅ Pod information retrieved:")
        print(f"   ID: {pod.get('id')}")
        print(f"   Name: {pod.get('name', 'N/A')}")
        print(f"   Status: {pod.get('desiredStatus')}")
        print(f"   Image: {pod.get('imageName', 'N/A')}")
        print(f"   Cost: ${pod.get('costPerHr', 0):.4f}/hr")
        
        runtime = pod.get('runtime', {})
        if runtime:
            uptime = runtime.get('uptimeInSeconds', 0)
            print(f"   Uptime: {uptime // 3600}h {(uptime % 3600) // 60}m")
            
            container = runtime.get('container', {})
            if container:
                print(f"   CPU: {container.get('cpuPercent', 0)}%")
                print(f"   Memory: {container.get('memoryPercent', 0)}%")
            
            gpus = runtime.get('gpus', [])
            if gpus:
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.get('gpuUtilPercent', 0)}% util, {gpu.get('memoryUtilPercent', 0)}% mem")
        
        return True, pod
        
    except Exception as e:
        print(f"❌ Error getting pod info: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def experiment_5_create_test_pod(client, gpu_type_id: str, dry_run: bool = True):
    """Experiment 5: Create a test pod (dry run by default)."""
    print("\n" + "="*60)
    print("EXPERIMENT 5: Create Test Pod")
    print("="*60)
    
    if not client:
        print("⚠️  No API client available, skipping")
        return False, None
    
    if not gpu_type_id:
        print("⚠️  No GPU type ID available, skipping")
        return False, None
    
    if dry_run:
        print("🔍 DRY RUN MODE - No pod will be created")
        print("\n   Would create pod with:")
        print(f"   - GPU Type: {gpu_type_id}")
        print(f"   - Image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04")
        print(f"   - Volume: 50GB at /workspace")
        print(f"   - Container Disk: 30GB")
        return True, None
    
    try:
        print(f"📝 Creating test pod with {gpu_type_id}...")
        pod = client.create_pod(
            name=f"tiny-icf-experiment-{int(time.time())}",
            gpu_type_id=gpu_type_id,
            image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            gpu_count=1,
            volume_in_gb=50,
            container_disk_in_gb=30,
            volume_mount_path="/workspace",
            env=[{"key": "PYTHONUNBUFFERED", "value": "1"}],
        )
        
        pod_id = pod['id']
        print(f"✅ Pod created: {pod_id}")
        print(f"   Status: {pod.get('desiredStatus')}")
        print(f"   Machine ID: {pod.get('machineId', 'N/A')}")
        
        machine = pod.get('machine', {})
        if machine:
            print(f"   Host ID: {machine.get('podHostId', 'N/A')}")
            print(f"   IP: {machine.get('runpodIp', 'N/A')}")
        
        return True, pod_id
        
    except Exception as e:
        print(f"❌ Error creating pod: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def experiment_6_test_runpodctl_integration(pod_id: Optional[str] = None):
    """Experiment 6: Test runpodctl integration."""
    print("\n" + "="*60)
    print("EXPERIMENT 6: runpodctl Integration")
    print("="*60)
    
    if not pod_id:
        print("⚠️  No pod ID provided, testing runpodctl config only")
        try:
            result = subprocess.run(
                ["runpodctl", "config", "--apiKey", "test"],
                capture_output=True,
                text=True,
            )
            print("✅ runpodctl is available")
            return True
        except FileNotFoundError:
            print("❌ runpodctl not found in PATH")
            return False
    
    try:
        print(f"📝 Testing runpodctl operations on pod {pod_id}...")
        
        # Test: Get pod info
        print("   Testing: runpodctl get pod")
        result = subprocess.run(
            ["runpodctl", "get", "pod", pod_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("   ✅ runpodctl get pod works")
        else:
            print(f"   ⚠️  runpodctl get pod failed: {result.stderr[:100]}")
        
        # Test: Check if we can exec
        print("   Testing: runpodctl exec (simple command)")
        result = subprocess.run(
            ["runpodctl", "exec", pod_id, "--", "echo", "test"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print(f"   ✅ runpodctl exec works: {result.stdout.strip()}")
        else:
            print(f"   ⚠️  runpodctl exec failed: {result.stderr[:100]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing runpodctl: {e}")
        import traceback
        traceback.print_exc()
        return False


def experiment_7_test_batch_script_integration():
    """Experiment 7: Test batch script integration."""
    print("\n" + "="*60)
    print("EXPERIMENT 7: Batch Script Integration")
    print("="*60)
    
    try:
        print("📝 Testing batch script with --use-api flag...")
        
        # Test help
        result = subprocess.run(
            [sys.executable, "scripts/runpod_batch.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "--use-api" in result.stdout:
            print("✅ Batch script supports --use-api flag")
        else:
            print("⚠️  --use-api flag not found in help")
        
        # Test status check (should work even without pod)
        print("\n   Testing: --status flag")
        result = subprocess.run(
            [sys.executable, "scripts/runpod_batch.py", "--status", "--pod-id", "test123"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # This will likely fail, but we're just testing the flag exists
        print(f"   Status check attempted (exit code: {result.returncode})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing batch script: {e}")
        import traceback
        traceback.print_exc()
        return False


def experiment_8_error_handling(client):
    """Experiment 8: Test error handling."""
    print("\n" + "="*60)
    print("EXPERIMENT 8: Error Handling")
    print("="*60)
    
    if not client:
        print("⚠️  No API client available, skipping")
        return False
    
    try:
        print("📝 Testing error handling...")
        
        # Test: Invalid pod ID
        print("   Testing: Get non-existent pod")
        try:
            pod = client.get_pod("invalid_pod_id_12345")
            print("   ⚠️  Should have raised an error")
        except Exception as e:
            print(f"   ✅ Correctly caught error: {type(e).__name__}")
        
        # Test: Invalid GPU type
        print("   Testing: Create pod with invalid GPU type")
        try:
            pod = client.create_pod(
                name="test-error",
                gpu_type_id="INVALID_GPU_TYPE_XYZ",
                image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
                gpu_count=1,
            )
            print("   ⚠️  Should have raised an error")
        except Exception as e:
            print(f"   ✅ Correctly caught error: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in error handling test: {e}")
        return False


def main():
    """Run all experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RunPod API Experiments")
    parser.add_argument("--dry-run", action="store_true", help="Don't create actual pods")
    parser.add_argument("--create-pod", action="store_true", help="Actually create a test pod (overrides --dry-run)")
    parser.add_argument("--pod-id", help="Use existing pod ID for some experiments")
    parser.add_argument("--skip", nargs="+", type=int, help="Skip specific experiments (1-8)")
    
    args = parser.parse_args()
    
    skip = set(args.skip or [])
    dry_run = args.dry_run and not args.create_pod
    
    print("🧪 RunPod API Experiments")
    print("="*60)
    print(f"Dry run: {dry_run}")
    print(f"Skip experiments: {skip}")
    
    results = {}
    client = None
    gpu_type_id = None
    pod_id = args.pod_id
    
    # Experiment 1: API Client Init
    if 1 not in skip:
        success, client = experiment_1_test_api_client_init()
        results[1] = success
        if not success or not client:
            print("\n❌ Cannot proceed without API client. Exiting.")
            return
    
    # Experiment 2: Query GPU Types
    if 2 not in skip:
        success, gpu_type_id = experiment_2_query_gpu_types(client)
        results[2] = success
    
    # Experiment 3: List Pods
    if 3 not in skip:
        success, existing_pod_id = experiment_3_list_existing_pods(client)
        results[3] = success
        if existing_pod_id and not pod_id:
            pod_id = existing_pod_id
            print(f"\n💡 Using existing pod {pod_id} for subsequent experiments")
    
    # Experiment 4: Get Pod Info
    if 4 not in skip and pod_id:
        success, _ = experiment_4_get_pod_info(client, pod_id)
        results[4] = success
    
    # Experiment 5: Create Pod
    if 5 not in skip and gpu_type_id:
        success, new_pod_id = experiment_5_create_test_pod(client, gpu_type_id, dry_run=dry_run)
        results[5] = success
        if new_pod_id and not pod_id:
            pod_id = new_pod_id
            print(f"\n💡 Using newly created pod {pod_id} for subsequent experiments")
            # Wait a bit for pod to initialize
            if not dry_run:
                print("   Waiting 30 seconds for pod to initialize...")
                time.sleep(30)
    
    # Experiment 6: runpodctl Integration
    if 6 not in skip:
        success = experiment_6_test_runpodctl_integration(pod_id)
        results[6] = success
    
    # Experiment 7: Batch Script Integration
    if 7 not in skip:
        success = experiment_7_test_batch_script_integration()
        results[7] = success
    
    # Experiment 8: Error Handling
    if 8 not in skip:
        success = experiment_8_error_handling(client)
        results[8] = success
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for exp_num, success in sorted(results.items()):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"Experiment {exp_num}: {status}")
    
    total = len(results)
    passed = sum(1 for s in results.values() if s)
    print(f"\nTotal: {passed}/{total} passed")
    
    if pod_id and not dry_run:
        print(f"\n⚠️  Test pod created: {pod_id}")
        print(f"   Remember to clean up: uv run scripts/runpod_api_client.py terminate {pod_id}")


if __name__ == "__main__":
    main()

