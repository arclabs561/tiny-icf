"""Export UniversalICF model weights for Rust inference."""

import argparse
import json

import torch
import torch.nn as nn

from tiny_icf.checkpoint import load_model
from tiny_icf.model import UniversalICF


def export_weights(model_path: str, output_json: str, output_bin: str | None = None):
    """Export UniversalICF model weights to JSON and optional binary format."""
    device = torch.device("cpu")

    # Load model
    model, checkpoint = load_model(model_path, device=device)
    if not isinstance(model, UniversalICF):
        model_type = checkpoint.get("model_type", type(model).__name__)
        raise ValueError(
            f"export_weights expects a UniversalICF checkpoint, got {model_type!r}. "
            "Use `tiny_icf.export_nano_weights` for NanoICF exports."
        )
    model.eval()

    def _fuse_conv_bn(seq: nn.Sequential) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse Conv1d + BatchNorm1d for inference-time export.

        UniversalICF uses `conv{k} = Sequential(Conv1d, BatchNorm1d)` and then applies ReLU.
        Exporting the fused conv weights lets downstream inference skip BatchNorm.
        """
        conv = seq[0]
        bn = seq[1]
        if not isinstance(conv, nn.Conv1d) or not isinstance(bn, nn.BatchNorm1d):
            raise TypeError("Expected Sequential(Conv1d, BatchNorm1d) for conv blocks.")

        w = conv.weight.detach().cpu()  # [out_c, in_c, k]
        if conv.bias is None:
            b = torch.zeros(w.size(0), dtype=w.dtype)
        else:
            b = conv.bias.detach().cpu()

        gamma = bn.weight.detach().cpu()
        beta = bn.bias.detach().cpu()
        mean = bn.running_mean.detach().cpu()
        var = bn.running_var.detach().cpu()
        eps = float(bn.eps)

        denom = torch.sqrt(var + eps)
        scale = gamma / denom  # [out_c]

        w_fused = w * scale.reshape(-1, 1, 1)
        b_fused = (b - mean) * scale + beta
        return w_fused, b_fused

    conv3_w, conv3_b = _fuse_conv_bn(model.conv3)
    conv5_w, conv5_b = _fuse_conv_bn(model.conv5)
    conv7_w, conv7_b = _fuse_conv_bn(model.conv7)

    head0 = model.head[0]
    head3 = model.head[3]
    if not isinstance(head0, nn.Linear) or not isinstance(head3, nn.Linear):
        raise TypeError(
            "Unexpected UniversalICF head structure; expected Linear layers at [0] and [3]."
        )

    # Extract weights
    weights = {
        "emb": model.emb.weight.detach().cpu().tolist(),  # [vocab_size, emb_dim]
        # Fused Conv+BN weights (Conv1d only; downstream should apply ReLU after conv).
        "conv3_w": conv3_w.tolist(),  # [conv_channels, emb_dim, 3]
        "conv3_b": conv3_b.tolist(),  # [conv_channels]
        "conv5_w": conv5_w.tolist(),  # [conv_channels, emb_dim, 5]
        "conv5_b": conv5_b.tolist(),  # [conv_channels]
        "conv7_w": conv7_w.tolist(),  # [conv_channels, emb_dim, 7]
        "conv7_b": conv7_b.tolist(),  # [conv_channels]
        "head_0_w": head0.weight.detach().cpu().tolist(),  # [hidden_dim, conv_channels*9]
        "head_0_b": head0.bias.detach().cpu().tolist(),  # [hidden_dim]
        "head_3_w": head3.weight.detach().cpu().tolist(),  # [1, hidden_dim]
        "head_3_b": head3.bias.detach().cpu().tolist(),  # [1]
        "metadata": {
            "model_type": "UniversalICF",
            "vocab_size": int(model.emb.num_embeddings),
            "emb_dim": int(model.emb.embedding_dim),
            "conv_channels": int(conv3_w.shape[0]),
            "hidden_dim": int(head0.out_features),
            "head_in_dim": int(head0.in_features),
            "kernel_sizes": [3, 5, 7],
            "max_length": 20,
            # Architectural notes for downstream implementations.
            "pooling": "multi_scale(max, mean, last) per kernel; concat => conv_channels*9",
            "output_activation": getattr(model, "output_activation", "clamp"),
            "sigmoid_temperature": float(getattr(model, "sigmoid_temperature", 1.0)),
            "use_attention": bool(getattr(model, "use_attention", False)),
        },
    }

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"✓ Exported weights to {output_json}")
    print(f"  Embedding: {len(weights['emb'])} x {len(weights['emb'][0])}")
    print("  Conv layers: 3 (kernels 3, 5, 7)")
    print("  Head layers: 2")

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
