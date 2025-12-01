#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""
Execute training command using RunPod API directly (MCP approach).
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

def execute_command_api(pod_id, command):
    """Execute command on pod using RunPod GraphQL API."""
    api_key = get_api_key()
    if not api_key:
        print("❌ No API key found")
        return False
    
    url = "https://api.runpod.io/graphql"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    # Try different GraphQL mutations
    mutations = [
        # Try podExec mutation
        {
            "query": """
            mutation PodExec($input: PodExecInput!) {
                podExec(input: $input) {
                    result
                    error
                }
            }
            """,
            "variables": {
                "input": {
                    "podId": pod_id,
                    "command": command,
                }
            }
        },
        # Try executeCommand
        {
            "query": """
            mutation ExecuteCommand($podId: String!, $command: String!) {
                executeCommand(podId: $podId, command: $command) {
                    output
                    error
                }
            }
            """,
            "variables": {
                "podId": pod_id,
                "command": command,
            }
        },
    ]
    
    for i, payload in enumerate(mutations, 1):
        try:
            print(f"   Trying API method {i}...")
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "errors" in data:
                    print(f"   ⚠️  GraphQL errors: {data['errors'][0].get('message', 'Unknown')}")
                    continue
                
                if "data" in data:
                    result = data["data"]
                    # Check various possible result structures
                    for key in ["podExec", "executeCommand"]:
                        if key in result:
                            exec_result = result[key]
                            if exec_result.get("result") or exec_result.get("output"):
                                output = exec_result.get("result") or exec_result.get("output")
                                print(f"   ✅ {output}")
                                return True
                            elif exec_result.get("error"):
                                print(f"   ⚠️  {exec_result['error']}")
                else:
                    print(f"   ⚠️  Unexpected response: {data}")
            else:
                print(f"   ⚠️  HTTP {response.status_code}: {response.text[:100]}")
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Request failed: {e}")
            continue
    
    return False

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    
    print("🎯 Execute Training via RunPod API (MCP Approach)")
    print(f"   Pod: {pod_id}\n")
    
    # Command to execute
    command = "cd /workspace/tiny-icf && bash start_mcp.sh"
    
    print(f"🚀 Executing: {command}\n")
    
    if execute_command_api(pod_id, command):
        print("\n✅ Training started via API!")
    else:
        print("\n⚠️  API execution not available")
        print("   Using alternative: File-based trigger or manual SSH")
        print(f"\n📋 Manual execution:")
        print(f"   runpodctl ssh connect {pod_id}")
        print(f"   {command}")

if __name__ == "__main__":
    main()

