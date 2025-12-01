"""Export Nano-CNN weights to binary format for Rust."""

import argparse
import json
import struct
from pathlib import Path

import torch

from tiny_icf.nano_model import NanoICF


def export_nano_weights(model_path: str, output_json: str, output_bin: str | None = None):
    """Export Nano-CNN weights to JSON and optionally binary format."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = NanoICF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    state = model.state_dict()
    
    # Extract weights
    weights = {
        "emb": state["emb.weight"].cpu().tolist(),  # [256, 16]
        "conv_w": state["conv.weight"].cpu().tolist(),  # [32, 16, 5]
        "conv_b": state["conv.bias"].cpu().tolist(),  # [32]
        "head_w": state["head.weight"].cpu().tolist(),  # [1, 32]
        "head_b": state["head.bias"].cpu().tolist(),  # [1]
    }
    
    # Metadata
    weights["metadata"] = {
        "vocab_size": 256,
        "emb_dim": state["emb.weight"].shape[1],
        "conv_channels": state["conv.weight"].shape[0],
        "kernel_size": state["conv.weight"].shape[2],
        "stride": 2,
        "max_length": 20,
    }
    
    # Save JSON
    with open(output_json, "w") as f:
        json.dump(weights, f, indent=2)
    
    json_size = Path(output_json).stat().st_size / 1024
    print(f"Exported JSON to {output_json}")
    print(f"JSON size: {json_size:.2f} KB")
    
    # Optionally save binary format (flat float32 array)
    if output_bin:
        # Flatten all weights into single array
        flat_weights = []
        
        # Embedding: [256, 16]
        flat_weights.extend(state["emb.weight"].cpu().flatten().tolist())
        
        # Conv weight: [32, 16, 5]
        flat_weights.extend(state["conv.weight"].cpu().flatten().tolist())
        
        # Conv bias: [32]
        flat_weights.extend(state["conv.bias"].cpu().tolist())
        
        # Head weight: [1, 32]
        flat_weights.extend(state["head.weight"].cpu().flatten().tolist())
        
        # Head bias: [1]
        flat_weights.extend(state["head.bias"].cpu().tolist())
        
        # Write as binary float32
        with open(output_bin, "wb") as f:
            # Write count first
            f.write(struct.pack("I", len(flat_weights)))
            # Write all floats
            for w in flat_weights:
                f.write(struct.pack("f", w))
        
        bin_size = Path(output_bin).stat().st_size / 1024
        print(f"Exported binary to {output_bin}")
        print(f"Binary size: {bin_size:.2f} KB")
    
    param_count = model.count_parameters()
    print(f"Total parameters: {param_count:,}")
    print(f"Expected size: ~{param_count * 4 / 1024:.2f} KB (float32)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Nano-CNN weights")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained Nano-CNN model")
    parser.add_argument("--json", type=Path, default=Path("nano_weights.json"), help="Output JSON path")
    parser.add_argument("--bin", type=Path, help="Optional binary output path")
    
    args = parser.parse_args()
    export_nano_weights(
        str(args.model),
        str(args.json),
        str(args.bin) if args.bin else None,
    )

