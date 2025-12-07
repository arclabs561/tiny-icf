#!/usr/bin/env -S uv run
"""
Shared utilities for RunPod scripts.
Harmonized API key extraction and configuration.
"""
import json
import re
import subprocess
from pathlib import Path


def get_api_key():
    """Get RunPod API key from MCP config (robust parsing).
    
    Returns:
        str: API key if found, None otherwise
    """
    mcp_config = Path.home() / ".cursor" / "mcp.json"
    if not mcp_config.exists():
        return None
    
    try:
        content = mcp_config.read_text()
        
        # Try JSON parsing first
        try:
            config = json.loads(content)
            
            def find_key(obj, key):
                """Recursively find key in nested structure."""
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
            pass  # Fall through to regex
        
        # Fallback: regex search (handles malformed JSON)
        patterns = [
            r'"RUNPOD_API_KEY"\s*:\s*"([^"]+)"',
            r'RUNPOD_API_KEY["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            r'runpod.*api.*key["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    except Exception:
        return None


def configure_runpodctl():
    """Configure runpodctl with API key from MCP config.
    
    Returns:
        bool: True if configured successfully, False otherwise
    """
    api_key = get_api_key()
    if not api_key:
        print("❌ Could not find RUNPOD_API_KEY in ~/.cursor/mcp.json")
        return False
    
    try:
        subprocess.run(
            ["runpodctl", "config", "--apiKey", api_key],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def extract_pod_id(output: str) -> str:
    """Extract pod ID from runpodctl output.
    
    Args:
        output: Output from runpodctl command
        
    Returns:
        str: Pod ID (without "pod-" prefix) or None
    """
    import re
    # Look for pod ID pattern (13+ alphanumeric chars)
    match = re.search(r'([a-z0-9]{13,})', output)
    if match:
        pod_id = match.group(1)
        # Remove "pod-" prefix if present
        if pod_id.startswith("pod-"):
            pod_id = pod_id[4:]
        return pod_id
    return None

