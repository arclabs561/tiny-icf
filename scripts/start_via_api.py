#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""
Start training using RunPod API directly - bypasses runpodctl exec issues.
"""
import sys
import requests
import json
import re
from pathlib import Path

def get_api_key():
    mcp = Path.home() / ".cursor" / "mcp.json"
    if not mcp.exists():
        return None
    try:
        match = re.search(r'"RUNPOD_API_KEY"\s*:\s*"([^"]+)"', mcp.read_text())
        return match.group(1) if match else None
    except:
        return None

def start_training_api(pod_id):
    """Start training via RunPod GraphQL API."""
    api_key = get_api_key()
    if not api_key:
        print("❌ No API key found")
        return False
    
    print(f"🚀 Starting training via RunPod API on {pod_id}...")
    
    # RunPod GraphQL API endpoint
    url = "https://api.runpod.io/graphql"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    # GraphQL mutation to execute command on pod
    # Note: This is a simplified version - actual API may differ
    mutation = """
    mutation ExecuteCommand($podId: String!, $command: String!) {
        podExecCommand(podId: $podId, command: $command) {
            result
            error
        }
    }
    """
    
    command = "cd /workspace/tiny-icf && bash start.sh"
    
    payload = {
        "query": mutation,
        "variables": {
            "podId": pod_id,
            "command": command,
        }
    }
    
    try:
        print("   📡 Calling RunPod API...")
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and data["data"]:
                result = data["data"].get("podExecCommand", {})
                if result.get("result"):
                    print(f"   ✅ {result['result']}")
                    return True
                elif result.get("error"):
                    print(f"   ⚠️  {result['error']}")
            else:
                print(f"   ⚠️  API response: {data}")
        else:
            print(f"   ⚠️  API error: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
        
        # If API doesn't support this, fall back to manual
        print("\n💡 API method not available, using manual approach")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"   ⚠️  API request failed: {e}")
        print("\n💡 Using manual SSH approach")
        return False

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    
    print("🎯 API-Based Auto-Start")
    print(f"   Pod: {pod_id}\n")
    
    if not start_training_api(pod_id):
        print("\n📋 Manual Start (Most Reliable):")
        print(f"   1. runpodctl ssh connect {pod_id}")
        print(f"   2. cd /workspace/tiny-icf && bash start.sh")
        print("\n   Or use the pre-uploaded scripts on the pod:")
        print(f"   - /workspace/tiny-icf/start.sh")
        print(f"   - /workspace/tiny-icf/start_simple.sh")
        print(f"   - /tmp/api_start.py")

if __name__ == "__main__":
    main()

