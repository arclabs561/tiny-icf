#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Fast monitoring - no hanging, shows progress immediately.
"""
import sys
import subprocess
import time

def check_via_ssh(pod_id):
    """Check status via SSH command (if supported)."""
    # Try to get SSH connection and execute command
    # This is a fallback - main method is file-based
    pass

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    
    print(f"📊 Fast Monitor for {pod_id}")
    print("   (Shows status every 10s, Ctrl+C to stop)\n")
    
    print("💡 To start training:")
    print(f"   runpodctl ssh connect {pod_id}")
    print(f"   cd /workspace/tiny-icf && bash start.sh\n")
    
    print("📊 Monitoring (checking pod status)...")
    
    for i in range(1, 100):
        # Check pod status via MCP/runpodctl
        try:
            result = subprocess.run(
                ["runpodctl", "get", "pod", pod_id],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if "RUNNING" in result.stdout:
                print(f"[{i}] Pod: RUNNING ✅")
            else:
                print(f"[{i}] Pod: {result.stdout[:50]}")
        except:
            print(f"[{i}] Status check failed")
        
        time.sleep(10)
        
        if i % 6 == 0:  # Every minute
            print(f"   💡 Training should be running. Check logs via SSH")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏸️  Monitoring stopped")

