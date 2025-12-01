"""Export UniversalICF model weights for Rust inference."""

import argparse
import json
from pathlib import Path

import torch

from tiny_icf.model import UniversalICF


def export_weights(model_path: str, output_json: str, output_bin: str | None = None):
    """Export UniversalICF model weights to JSON and optional binary format."""
    device = torch.device("cpu")
    
    # Load model
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Extract weights
    weights = {
        "emb": model.emb.weight.detach().cpu().tolist(),  # [256, 48]
        "conv3_w": model.conv3.weight.detach().cpu().tolist(),  # [24, 48, 3]
        "conv3_b": model.conv3.bias.detach().cpu().tolist(),  # [24]
        "conv5_w": model.conv5.weight.detach().cpu().tolist(),  # [24, 48, 5]
        "conv5_b": model.conv5.bias.detach().cpu().tolist(),  # [24]
        "conv7_w": model.conv7.weight.detach().cpu().tolist(),  # [24, 48, 7]
        "conv7_b": model.conv7.bias.detach().cpu().tolist(),  # [24]
        "head_0_w": model.head[0].weight.detach().cpu().tolist(),  # [48, 72]
        "head_0_b": model.head[0].bias.detach().cpu().tolist(),  # [48]
        "head_3_w": model.head[3].weight.detach().cpu().tolist(),  # [1, 48]
        "head_3_b": model.head[3].bias.detach().cpu().tolist(),  # [1]
        "metadata": {
            "vocab_size": 256,
            "emb_dim": 48,
            "conv_channels": 24,
            "hidden_dim": 48,
            "kernel_sizes": [3, 5, 7],
            "max_length": 20,
        }
    }
    
    # Save JSON
    with open(output_json, "w") as f:
        json.dump(weights, f, indent=2)
    
    print(f"✓ Exported weights to {output_json}")
    print(f"  Embedding: {len(weights['emb'])} x {len(weights['emb'][0])}")
    print(f"  Conv layers: 3 (kernels 3, 5, 7)")
    print(f"  Head layers: 2")
    
    # Save binary if requested
    if output_bin:
        # Flatten all weights into a single array
        flat_weights = []
        flat_weights.extend([w for row in weights["emb"] for w in row])
        flat_weights.extend([w for c in weights["conv3_w"] for row in c for w in row])
        flat_weights.extend(weights["conv3_b"])
        flat_weights.extend([w for c in weights["conv5_w"] for row in c for w in row])
        flat_weights.extend(weights["conv5_b"])
        flat_weights.extend([w for c in weights["conv7_w"] for row in c for w in row])
        flat_weights.extend(weights["conv7_b"])
        flat_weights.extend([w for row in weights["head_0_w"] for w in row])
        flat_weights.extend(weights["head_0_b"])
        flat_weights.extend([w for row in weights["head_3_w"] for w in row])
        flat_weights.extend(weights["head_3_b"])
        
        # Write as binary float32
        import struct
        with open(output_bin, "wb") as f:
            for w in flat_weights:
                f.write(struct.pack("f", w))
        
        print(f"✓ Exported binary weights to {output_bin}")
        print(f"  Total floats: {len(flat_weights)}")
        print(f"  Size: {len(flat_weights) * 4 / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description="Export UniversalICF weights for Rust")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--json", type=str, default="rust/weights.json", help="Output JSON path")
    parser.add_argument("--bin", type=str, help="Output binary path (optional)")
    
    args = parser.parse_args()
    
    export_weights(args.model, args.json, args.bin)


if __name__ == "__main__":
    main()
