#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""
Real-time pod monitoring using RunPod API.

Monitors pod status, CPU, GPU, and memory usage in real-time.
"""
import sys
import time
import json
from pathlib import Path


def monitor_pod(pod_id: str, interval: int = 10, duration: int = 300):
    """Monitor pod in real-time."""
    try:
        # Add parent directory to path for imports
        import sys
        from pathlib import Path
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from scripts.runpod_api_client import RunPodAPIClient
        
        client = RunPodAPIClient()
        
        print(f"📊 Monitoring pod {pod_id}")
        print(f"   Interval: {interval}s")
        print(f"   Duration: {duration}s")
        print("=" * 60)
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration:
            iteration += 1
            try:
                pod = client.get_pod(pod_id)
                
                # Basic info
                status = pod.get('desiredStatus', 'UNKNOWN')
                name = pod.get('name', 'unnamed')
                cost = pod.get('costPerHr', 0)
                
                # Runtime info
                runtime = pod.get('runtime', {})
                uptime = runtime.get('uptimeInSeconds', 0)
                container = runtime.get('container', {})
                gpus = runtime.get('gpus', [])
                
                # Format uptime
                hours = uptime // 3600
                minutes = (uptime % 3600) // 60
                seconds = uptime % 60
                
                # Display
                print(f"\n[{iteration}] {time.strftime('%H:%M:%S')}")
                print(f"  Status: {status} | Cost: ${cost:.4f}/hr")
                print(f"  Uptime: {hours}h {minutes}m {seconds}s")
                
                if container:
                    cpu = container.get('cpuPercent', 0)
                    mem = container.get('memoryPercent', 0)
                    print(f"  CPU: {cpu}% | Memory: {mem}%")
                
                if gpus:
                    for i, gpu in enumerate(gpus):
                        gpu_util = gpu.get('gpuUtilPercent', 0)
                        mem_util = gpu.get('memoryUtilPercent', 0)
                        print(f"  GPU {i}: {gpu_util}% util | {mem_util}% mem")
                
                # Check if training is running (heuristic)
                if gpus and any(g.get('gpuUtilPercent', 0) > 10 for g in gpus):
                    print("  🔥 GPU active - training likely running")
                elif container and container.get('cpuPercent', 0) > 5:
                    print("  ⚙️  CPU active - processing")
                else:
                    print("  💤 Idle")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n\n⏹️  Monitoring stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                time.sleep(interval)
        
        print("\n" + "=" * 60)
        print("✅ Monitoring complete")
        
    except Exception as e:
        print(f"❌ Failed to start monitoring: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor RunPod pod in real-time")
    parser.add_argument("pod_id", help="Pod ID to monitor")
    parser.add_argument("--interval", type=int, default=10, help="Update interval in seconds (default: 10)")
    parser.add_argument("--duration", type=int, default=300, help="Monitoring duration in seconds (default: 300)")
    
    args = parser.parse_args()
    
    monitor_pod(args.pod_id, args.interval, args.duration)


if __name__ == "__main__":
    main()

