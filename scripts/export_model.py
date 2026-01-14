#!/usr/bin/env -S uv run
"""Export model for deployment: weights, metadata, and inference code."""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.model import UniversalICF


def export_weights(model_path: Path, output_dir: Path):
    """Export model weights to JSON for easy loading in other languages."""
    device = torch.device("cpu")
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    weights = {}
    
    # Export all parameters
    for name, param in model.named_parameters():
        weights[name] = {
            "shape": list(param.shape),
            "data": param.detach().cpu().numpy().tolist(),
        }
    
    # Export buffers (if any)
    for name, buffer in model.named_buffers():
        weights[name] = {
            "shape": list(buffer.shape),
            "data": buffer.detach().cpu().numpy().tolist(),
        }
    
    # Save weights
    weights_file = output_dir / "weights.json"
    with open(weights_file, "w") as f:
        json.dump(weights, f, indent=2)
    
    print(f"✓ Weights exported to {weights_file}")
    
    # Export model metadata
    metadata = {
        "model_type": "UniversalICF",
        "parameters": model.count_parameters(),
        "vocab_size": 256,
        "emb_dim": 48,
        "conv_channels": 24,
        "hidden_dim": 48,
        "max_length": 20,
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata exported to {metadata_file}")
    
    return weights_file, metadata_file


def export_onnx(model_path: Path, output_file: Path):
    """Export model to ONNX format."""
    try:
        import torch.onnx
    except ImportError:
        print("ONNX export requires torch.onnx (usually included with PyTorch)")
        return False
    
    device = torch.device("cpu")
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, 256, (1, 20), dtype=torch.long)
    
    # Export
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_file,
            input_names=["bytes"],
            output_names=["icf_score"],
            dynamic_axes={
                "bytes": {0: "batch_size"},
                "icf_score": {0: "batch_size"},
            },
            opset_version=11,
        )
        print(f"✓ ONNX model exported to {output_file}")
        return True
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False


def export_torchscript(model_path: Path, output_file: Path):
    """Export model to TorchScript format."""
    device = torch.device("cpu")
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, 256, (1, 20), dtype=torch.long)
    
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(output_file))
        print(f"✓ TorchScript model exported to {output_file}")
        return True
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export model for deployment")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--output-dir", type=Path, default=Path("export"), help="Output directory")
    parser.add_argument("--format", type=str, choices=["weights", "onnx", "torchscript", "all"],
                       default="all", help="Export format")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Model Export")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    print()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format in ["weights", "all"]:
        export_weights(args.model, args.output_dir)
    
    if args.format in ["onnx", "all"]:
        onnx_file = args.output_dir / "model.onnx"
        export_onnx(args.model, onnx_file)
    
    if args.format in ["torchscript", "all"]:
        torchscript_file = args.output_dir / "model.pt"
        export_torchscript(args.model, torchscript_file)
    
    print(f"\n✅ Export complete! Files in {args.output_dir}")


if __name__ == "__main__":
    main()

