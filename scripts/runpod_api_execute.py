#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""
Execute commands on RunPod pods using GraphQL API directly.
Bypasses runpodctl exec limitations - more reliable for non-interactive jobs.
"""

import json
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("❌ requests library required: uv pip install requests")
    sys.exit(1)


def get_api_key():
    """Get RunPod API key from MCP config."""
    mcp_config = Path.home() / ".cursor" / "mcp.json"
    if not mcp_config.exists():
        return None
    
    try:
        content = mcp_config.read_text()
        # Try JSON parsing first
        try:
            config = json.loads(content)
            def find_key(obj, key):
                if isinstance(obj, dict):
                    if key in obj:
                        return obj[key]
                    for v in obj.values():
                        result = find_key(v, key)
                        if result:
                            return result
                elif isinstance(obj, list):
                    for item in obj:
                        result = find_key(item, key)
                        if result:
                            return result
                return None
            api_key = find_key(config, "RUNPOD_API_KEY")
            if api_key:
                return api_key
        except json.JSONDecodeError:
            pass
        
        # Fallback: regex
        match = re.search(r'"RUNPOD_API_KEY"\s*:\s*"([^"]+)"', content)
        return match.group(1) if match else None
    except Exception:
        return None


def execute_command_api(pod_id: str, command: str, timeout: int = 3600) -> tuple[bool, str, str]:
    """
    Execute command on pod using RunPod GraphQL API.
    
    Returns:
        (success, stdout, stderr)
    """
    api_key = get_api_key()
    if not api_key:
        return False, "", "No API key found"
    
    url = "https://api.runpod.io/graphql"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    # RunPod GraphQL mutation for pod exec
    # Based on RunPod API documentation
    mutation = """
    mutation PodExec($input: PodExecInput!) {
        podExec(input: $input) {
            id
            result
            error
        }
    }
    """
    
    payload = {
        "query": mutation,
        "variables": {
            "input": {
                "podId": pod_id,
                "command": command,
            }
        }
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=30,
        )
        
        if response.status_code != 200:
            return False, "", f"API error {response.status_code}: {response.text[:200]}"
        
        data = response.json()
        
        if "errors" in data:
            error_msg = "; ".join([e.get("message", str(e)) for e in data["errors"]])
            return False, "", f"GraphQL error: {error_msg}"
        
        if "data" in data and data["data"]:
            result = data["data"].get("podExec", {})
            exec_id = result.get("id")
            
            if exec_id:
                # Poll for result (async execution)
                return poll_exec_result(api_key, exec_id, timeout)
            elif result.get("result"):
                # Immediate result
                return True, result["result"], ""
            elif result.get("error"):
                return False, "", result["error"]
        
        return False, "", f"Unexpected API response: {json.dumps(data)[:200]}"
        
    except requests.exceptions.Timeout:
        return False, "", "API request timeout"
    except requests.exceptions.RequestException as e:
        return False, "", f"Request failed: {e}"


def poll_exec_result(api_key: str, exec_id: str, timeout: int = 3600) -> tuple[bool, str, str]:
    """Poll for execution result."""
    url = "https://api.runpod.io/graphql"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    query = """
    query PodExecStatus($id: String!) {
        podExecStatus(id: $id) {
            status
            result
            error
            stdout
            stderr
        }
    }
    """
    
    start_time = time.time()
    poll_interval = 2  # seconds
    
    while time.time() - start_time < timeout:
        try:
            payload = {
                "query": query,
                "variables": {"id": exec_id}
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "data" in data and data["data"]:
                    status_data = data["data"].get("podExecStatus", {})
                    status = status_data.get("status", "UNKNOWN")
                    
                    if status == "COMPLETED":
                        return True, status_data.get("stdout", ""), status_data.get("stderr", "")
                    elif status == "FAILED":
                        return False, status_data.get("stdout", ""), status_data.get("error", status_data.get("stderr", ""))
                    elif status in ["RUNNING", "PENDING"]:
                        time.sleep(poll_interval)
                        continue
            
            time.sleep(poll_interval)
            
        except Exception as e:
            time.sleep(poll_interval)
            continue
    
    return False, "", f"Timeout after {timeout}s"


def execute_training(pod_id: str, background: bool = False) -> bool:
    """Execute training command on pod."""
    command = """
cd /workspace/tiny-icf
uv run scripts/train_batch.py \\
    --data data/multilingual/multilingual_combined.csv \\
    --typo-corpus data/typos/github_typos.csv \\
    --output-dir /workspace/tiny-icf/models \\
    --epochs 50 \\
    --batch-size 256 \\
    --devices 1 \\
    --precision 16-mixed \\
    2>&1 | tee /workspace/tiny-icf/training.log
echo "Training exit code: $?"
"""
    
    if background:
        # Run in background using nohup
        command = f"""
cd /workspace/tiny-icf
nohup bash -c '{command.replace(chr(10), "; ")}' > /dev/null 2>&1 &
echo $! > /workspace/tiny-icf/training.pid
echo "Training started with PID: $(cat /workspace/tiny-icf/training.pid)"
"""
    
    print(f"🚀 Executing training on pod {pod_id}...")
    print(f"   Method: RunPod GraphQL API (bypasses runpodctl)")
    
    success, stdout, stderr = execute_command_api(pod_id, command, timeout=3600 if not background else 30)
    
    if success:
        print("✅ Command executed successfully")
        if stdout:
            print(f"   Output: {stdout[:500]}")
        return True
    else:
        print(f"❌ Command execution failed")
        if stderr:
            print(f"   Error: {stderr[:500]}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Execute commands on RunPod pods via API")
    parser.add_argument("pod_id", help="Pod ID")
    parser.add_argument("--command", help="Command to execute (default: training)")
    parser.add_argument("--background", action="store_true", help="Run in background")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
    
    args = parser.parse_args()
    
    if args.command:
        success, stdout, stderr = execute_command_api(args.pod_id, args.command, args.timeout)
        if success:
            print(stdout)
            sys.exit(0)
        else:
            print(stderr, file=sys.stderr)
            sys.exit(1)
    else:
        # Default: run training
        success = execute_training(args.pod_id, background=args.background)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

