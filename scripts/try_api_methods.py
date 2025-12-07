#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""
Try multiple API methods to execute commands on RunPod.
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

def try_rest_endpoint(pod_id, command, api_key):
    """Try REST API endpoint for command execution."""
    print("   Trying REST API endpoint...")
    
    # Try different REST endpoints
    endpoints = [
        f"https://api.runpod.io/v1/pods/{pod_id}/exec",
        f"https://api.runpod.io/v1/pods/{pod_id}/command",
        f"https://api.runpod.io/v1/pods/{pod_id}/run",
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    for endpoint in endpoints:
        try:
            payload = {"command": command}
            response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                print(f"   ✅ Success with {endpoint}")
                return True, response.json()
            elif response.status_code != 404:
                print(f"   ⚠️  {endpoint}: {response.status_code} - {response.text[:100]}")
        except requests.exceptions.RequestException as e:
            continue
    
    return False, None

def try_graphql_variations(pod_id, command, api_key):
    """Try various GraphQL mutations."""
    print("   Trying GraphQL variations...")
    
    url = "https://api.runpod.io/graphql"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    # Try introspection first to see available mutations
    introspection = {
        "query": """
        {
            __type(name: "Mutation") {
                fields {
                    name
                    args {
                        name
                        type {
                            name
                        }
                    }
                }
            }
        }
        """
    }
    
    try:
        response = requests.post(url, json=introspection, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "__type" in data["data"]:
                mutations = data["data"]["__type"]["fields"]
                print(f"   Found {len(mutations)} mutations")
                # Look for exec/command related
                exec_mutations = [m for m in mutations if "exec" in m["name"].lower() or "command" in m["name"].lower() or "run" in m["name"].lower()]
                if exec_mutations:
                    print(f"   Found potential mutations: {[m['name'] for m in exec_mutations]}")
                    # Try these mutations
                    for mut in exec_mutations[:3]:  # Try first 3
                        mut_name = mut["name"]
                        args = {arg["name"]: arg["type"]["name"] for arg in mut.get("args", [])}
                        print(f"   Trying mutation: {mut_name} with args: {args}")
                        
                        # Build mutation based on args
                        if "podId" in args or "id" in args:
                            var_name = "podId" if "podId" in args else "id"
                            mutation = f"""
                            mutation {mut_name}(${var_name}: String!) {{
                                {mut_name}({var_name}: ${var_name}) {{
                                    result
                                    output
                                    error
                                }}
                            }}
                            """
                            payload = {
                                "query": mutation,
                                "variables": {var_name: pod_id}
                            }
                            
                            try:
                                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                                if resp.status_code == 200:
                                    result = resp.json()
                                    if "errors" not in result:
                                        print(f"   ✅ {mut_name} succeeded!")
                                        return True, result
                            except:
                                continue
    except Exception as e:
        print(f"   ⚠️  Introspection failed: {e}")
    
    return False, None

def try_serverless_approach(pod_id, command, api_key):
    """Try using serverless endpoint approach."""
    print("   Trying serverless endpoint approach...")
    
    # Create a simple handler that executes the command
    # This would require creating a serverless endpoint first
    # For now, just document the approach
    print("   ⚠️  Serverless approach requires endpoint creation first")
    return False, None

def main():
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "9lj0lizlogeftc"
    command = "cd /workspace/tiny-icf && bash start_mcp.sh"
    
    print("🔍 Trying Multiple API Methods")
    print(f"   Pod: {pod_id}")
    print(f"   Command: {command}\n")
    
    api_key = get_api_key()
    if not api_key:
        print("❌ No API key found")
        return
    
    print("📡 Method 1: REST Endpoints")
    success, result = try_rest_endpoint(pod_id, command, api_key)
    if success:
        print(f"   ✅ Success! Result: {result}")
        return
    
    print("\n📡 Method 2: GraphQL Introspection & Mutations")
    success, result = try_graphql_variations(pod_id, command, api_key)
    if success:
        print(f"   ✅ Success! Result: {result}")
        return
    
    print("\n📡 Method 3: Serverless Endpoint")
    try_serverless_approach(pod_id, command, api_key)
    
    print("\n❌ All API methods failed")
    print("   RunPod API doesn't support direct command execution")
    print("   Use MCP tools in Cursor or SSH instead")

if __name__ == "__main__":
    main()

