#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""
RunPod GraphQL API client for direct pod management.

This uses the RunPod GraphQL API directly, bypassing runpodctl for:
- Creating pods
- Starting/stopping pods
- Getting pod information
- Listing pods

Note: The API does NOT support direct command execution on pods.
For that, use SSH or runpodctl exec (or upload scripts and use entrypoints).
"""
import sys
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests


class RunPodAPIError(Exception):
    """Error from RunPod API."""
    pass


class RunPodAPIClient:
    """Client for RunPod GraphQL API."""
    
    API_URL = "https://api.runpod.io/graphql"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize client with API key."""
        self.api_key = api_key or self._get_api_key()
        if not self.api_key:
            raise RunPodAPIError("No API key found. Set RUNPOD_API_KEY or provide in ~/.cursor/mcp.json")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
    
    @staticmethod
    def _get_api_key() -> Optional[str]:
        """Get API key from ~/.cursor/mcp.json."""
        mcp = Path.home() / ".cursor" / "mcp.json"
        if not mcp.exists():
            return None
        try:
            match = re.search(r'"RUNPOD_API_KEY"\s*:\s*"([^"]+)"', mcp.read_text())
            return match.group(1) if match else None
        except Exception:
            return None
    
    def _execute_query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute GraphQL query/mutation."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        try:
            response = requests.post(
                self.API_URL,
                json=payload,
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            
            if "errors" in data:
                error_msg = "; ".join(e.get("message", str(e)) for e in data["errors"])
                raise RunPodAPIError(f"GraphQL errors: {error_msg}")
            
            return data.get("data", {})
        except requests.exceptions.RequestException as e:
            raise RunPodAPIError(f"API request failed: {e}")
    
    def create_pod(
        self,
        name: str,
        gpu_type_id: str,
        image_name: str,
        gpu_count: int = 1,
        volume_in_gb: int = 50,
        container_disk_in_gb: int = 30,
        min_vcpu_count: int = 2,
        min_memory_in_gb: int = 15,
        volume_mount_path: str = "/workspace",
        env: Optional[List[Dict[str, str]]] = None,
        cloud_type: str = "ALL",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create an on-demand pod.
        
        Args:
            name: Pod name
            gpu_type_id: GPU type ID (e.g., "NVIDIA GeForce RTX 4080 SUPER")
            image_name: Docker image name
            gpu_count: Number of GPUs
            volume_in_gb: Volume size in GB
            container_disk_in_gb: Container disk size in GB
            min_vcpu_count: Minimum vCPU count
            min_memory_in_gb: Minimum memory in GB
            volume_mount_path: Volume mount path
            env: Environment variables as [{"key": "VAR", "value": "val"}]
            cloud_type: "ALL", "SECURE", or "COMMUNITY"
            **kwargs: Additional pod configuration
        
        Returns:
            Pod information including id, machineId, etc.
        """
        env_vars = env or []
        env_input = [{"key": e["key"], "value": e["value"]} for e in env_vars]
        
        mutation = """
        mutation CreatePod($input: PodFindAndDeployOnDemandInput) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                imageName
                machineId
                desiredStatus
                costPerHr
                machine {
                    podHostId
                    runpodIp
                }
            }
        }
        """
        
        variables = {
            "input": {
                "name": name,
                "gpuTypeId": gpu_type_id,
                "imageName": image_name,
                "gpuCount": gpu_count,
                "volumeInGb": volume_in_gb,
                "containerDiskInGb": container_disk_in_gb,
                "minVcpuCount": min_vcpu_count,
                "minMemoryInGb": min_memory_in_gb,
                "volumeMountPath": volume_mount_path,
                "cloudType": cloud_type,
                "env": env_input,
                **kwargs
            }
        }
        
        result = self._execute_query(mutation, variables)
        pod = result.get("podFindAndDeployOnDemand")
        if not pod:
            raise RunPodAPIError("Failed to create pod")
        return pod
    
    def get_pod(self, pod_id: str) -> Dict[str, Any]:
        """Get pod information by ID."""
        query = """
        query GetPod($input: PodFilter) {
            pod(input: $input) {
                id
                name
                desiredStatus
                imageName
                machineId
                costPerHr
                runtime {
                    uptimeInSeconds
                    container {
                        cpuPercent
                        memoryPercent
                    }
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
                machine {
                    podHostId
                    runpodIp
                }
            }
        }
        """
        
        variables = {"input": {"podId": pod_id}}
        result = self._execute_query(query, variables)
        pod = result.get("pod")
        if not pod:
            raise RunPodAPIError(f"Pod {pod_id} not found")
        return pod
    
    def list_pods(self) -> List[Dict[str, Any]]:
        """List all pods."""
        query = """
        query ListPods {
            myself {
                pods {
                    id
                    name
                    desiredStatus
                    imageName
                    costPerHr
                    runtime {
                        uptimeInSeconds
                    }
                }
            }
        }
        """
        
        result = self._execute_query(query)
        user = result.get("myself", {})
        return user.get("pods", [])
    
    def start_pod(self, pod_id: str, gpu_count: Optional[int] = None) -> Dict[str, Any]:
        """Start (resume) a pod."""
        mutation = """
        mutation StartPod($input: PodResumeInput!) {
            podResume(input: $input) {
                id
                desiredStatus
                machineId
            }
        }
        """
        
        input_data = {"podId": pod_id}
        if gpu_count is not None:
            input_data["gpuCount"] = gpu_count
        
        variables = {"input": input_data}
        result = self._execute_query(mutation, variables)
        pod = result.get("podResume")
        if not pod:
            raise RunPodAPIError(f"Failed to start pod {pod_id}")
        return pod
    
    def stop_pod(self, pod_id: str) -> Dict[str, Any]:
        """Stop a pod."""
        mutation = """
        mutation StopPod($input: PodStopInput!) {
            podStop(input: $input) {
                id
                desiredStatus
            }
        }
        """
        
        variables = {"input": {"podId": pod_id}}
        result = self._execute_query(mutation, variables)
        pod = result.get("podStop")
        if not pod:
            raise RunPodAPIError(f"Failed to stop pod {pod_id}")
        return pod
    
    def terminate_pod(self, pod_id: str) -> None:
        """Terminate a pod (permanent deletion)."""
        mutation = """
        mutation TerminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        
        variables = {"input": {"podId": pod_id}}
        self._execute_query(mutation, variables)
    
    def get_gpu_types(self, gpu_type_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List GPU types, optionally filtered by ID."""
        query = """
        query GetGpuTypes($input: GpuTypeFilter) {
            gpuTypes(input: $input) {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
                securePrice
                communityPrice
            }
        }
        """
        
        variables = {"input": {"id": gpu_type_id}} if gpu_type_id else None
        result = self._execute_query(query, variables)
        return result.get("gpuTypes", [])


def main():
    """CLI interface for RunPod API client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RunPod GraphQL API client")
    parser.add_argument("--api-key", help="RunPod API key (or set in ~/.cursor/mcp.json)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create pod
    create_parser = subparsers.add_parser("create", help="Create a pod")
    create_parser.add_argument("--name", required=True, help="Pod name")
    create_parser.add_argument("--gpu-type", required=True, help="GPU type ID")
    create_parser.add_argument("--image", required=True, help="Docker image name")
    create_parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs")
    create_parser.add_argument("--volume-gb", type=int, default=50, help="Volume size in GB")
    create_parser.add_argument("--disk-gb", type=int, default=30, help="Container disk size in GB")
    create_parser.add_argument("--volume-path", default="/workspace", help="Volume mount path")
    
    # Get pod
    get_parser = subparsers.add_parser("get", help="Get pod information")
    get_parser.add_argument("pod_id", help="Pod ID")
    
    # List pods
    subparsers.add_parser("list", help="List all pods")
    
    # Start pod
    start_parser = subparsers.add_parser("start", help="Start a pod")
    start_parser.add_argument("pod_id", help="Pod ID")
    start_parser.add_argument("--gpu-count", type=int, help="GPU count")
    
    # Stop pod
    stop_parser = subparsers.add_parser("stop", help="Stop a pod")
    stop_parser.add_argument("pod_id", help="Pod ID")
    
    # Terminate pod
    term_parser = subparsers.add_parser("terminate", help="Terminate a pod")
    term_parser.add_argument("pod_id", help="Pod ID")
    
    # List GPU types
    gpu_parser = subparsers.add_parser("gpu-types", help="List GPU types")
    gpu_parser.add_argument("--id", help="Filter by GPU type ID")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        client = RunPodAPIClient(api_key=args.api_key)
        
        if args.command == "create":
            pod = client.create_pod(
                name=args.name,
                gpu_type_id=args.gpu_type,
                image_name=args.image,
                gpu_count=args.gpu_count,
                volume_in_gb=args.volume_gb,
                container_disk_in_gb=args.disk_gb,
                volume_mount_path=args.volume_path,
            )
            print(f"✅ Pod created: {pod['id']}")
            print(json.dumps(pod, indent=2))
        
        elif args.command == "get":
            pod = client.get_pod(args.pod_id)
            print(json.dumps(pod, indent=2))
        
        elif args.command == "list":
            pods = client.list_pods()
            print(f"Found {len(pods)} pods:")
            for pod in pods:
                print(f"  {pod['id']}: {pod['name']} ({pod['desiredStatus']})")
        
        elif args.command == "start":
            pod = client.start_pod(args.pod_id, gpu_count=getattr(args, 'gpu_count', None))
            print(f"✅ Pod {args.pod_id} started")
            print(json.dumps(pod, indent=2))
        
        elif args.command == "stop":
            pod = client.stop_pod(args.pod_id)
            print(f"✅ Pod {args.pod_id} stopped")
            print(json.dumps(pod, indent=2))
        
        elif args.command == "terminate":
            client.terminate_pod(args.pod_id)
            print(f"✅ Pod {args.pod_id} terminated")
        
        elif args.command == "gpu-types":
            gpus = client.get_gpu_types(gpu_type_id=getattr(args, 'id', None))
            print(f"Found {len(gpus)} GPU types:")
            for gpu in gpus:
                print(f"  {gpu['id']}: {gpu['displayName']} ({gpu['memoryInGb']}GB)")
    
    except RunPodAPIError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

