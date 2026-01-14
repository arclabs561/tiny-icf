"""Compress model using quantization and pruning."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.quantization

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def quantize_model(model: nn.Module, device: str = 'cpu') -> nn.Module:
    """
    Quantize model to int8 (8-bit integers).
    
    Reduces model size by ~4x (float32 → int8).
    """
    print("Quantizing model to int8...")
    
    # Set quantization config
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # For CPU
    
    # Prepare for quantization
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate (run dummy data through model)
    print("  Calibrating...")
    dummy_input = torch.randint(0, 256, (1, 20), dtype=torch.long)
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Convert to quantized
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    print("✅ Model quantized to int8")
    return quantized_model


def prune_model(model: nn.Module, pruning_ratio: float = 0.3) -> nn.Module:
    """
    Prune model by removing least important weights.
    
    Args:
        model: Model to prune
        pruning_ratio: Fraction of weights to prune (0.3 = 30%)
    
    Returns:
        Pruned model
    """
    print(f"Pruning model ({pruning_ratio*100:.1f}% of weights)...")
    
    # Use magnitude-based pruning
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            parameters_to_prune.append((module, 'weight'))
    
    # Prune
    torch.nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=pruning_ratio,
    )
    
    # Make pruning permanent
    for module, param_name in parameters_to_prune:
        torch.nn.utils.prune.remove(module, param_name)
    
    print(f"✅ Model pruned ({pruning_ratio*100:.1f}% of weights removed)")
    return model


def get_model_size(model: nn.Module, quantized: bool = False) -> int:
    """Get model size in bytes."""
    if quantized:
        # Quantized models store weights as int8
        total_params = sum(p.numel() for p in model.parameters())
        return total_params  # 1 byte per parameter (int8)
    else:
        # Float32 models: 4 bytes per parameter
        total_params = sum(p.numel() for p in model.parameters())
        return total_params * 4


def compress_model(
    model_path: Path,
    output_path: Path,
    quantization: bool = True,
    pruning: bool = True,
    pruning_ratio: float = 0.3,
    device: str = 'cpu',
):
    """
    Compress model using quantization and/or pruning.
    
    Args:
        model_path: Path to original model checkpoint
        output_path: Path to save compressed model
        quantization: If True, quantize to int8
        pruning: If True, prune weights
        pruning_ratio: Fraction of weights to prune (if pruning=True)
        device: Device to run on ('cpu' or 'cuda')
    """
    print("=" * 70)
    print("Model Compression")
    print("=" * 70)
    
    # Load model
    print(f"Loading model from: {model_path}")
    if device == 'cuda' and torch.cuda.is_available():
        device_obj = torch.device('cuda')
        checkpoint = torch.load(model_path, map_location=device_obj)
    else:
        device_obj = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract model
    from tiny_icf.model import UniversalICF
    model = UniversalICF()
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device_obj)
    model.eval()
    
    # Get original size
    original_size = get_model_size(model, quantized=False)
    print(f"\nOriginal model size: {original_size:,} bytes ({original_size/1024:.2f} KB)")
    
    # Apply compression
    if pruning:
        model = prune_model(model, pruning_ratio=pruning_ratio)
        pruned_size = get_model_size(model, quantized=False)
        print(f"After pruning: {pruned_size:,} bytes ({pruned_size/1024:.2f} KB)")
        print(f"  Reduction: {(1 - pruned_size/original_size)*100:.1f}%")
    
    if quantization:
        model = quantize_model(model, device=device)
        quantized_size = get_model_size(model, quantized=True)
        print(f"After quantization: {quantized_size:,} bytes ({quantized_size/1024:.2f} KB)")
        print(f"  Total reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    
    # Save compressed model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\n✅ Compressed model saved to: {output_path}")
    
    # Final size
    final_size = get_model_size(model, quantized=quantization)
    print(f"Final model size: {final_size:,} bytes ({final_size/1024:.2f} KB)")
    
    return model


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compress model using quantization and pruning')
    parser.add_argument('--model', type=Path, required=True,
                        help='Path to original model checkpoint')
    parser.add_argument('--output', type=Path, required=True,
                        help='Path to save compressed model')
    parser.add_argument('--quantization', action='store_true', default=True,
                        help='Apply quantization (default: True)')
    parser.add_argument('--no-quantization', dest='quantization', action='store_false',
                        help='Disable quantization')
    parser.add_argument('--pruning', action='store_true', default=True,
                        help='Apply pruning (default: True)')
    parser.add_argument('--no-pruning', dest='pruning', action='store_false',
                        help='Disable pruning')
    parser.add_argument('--pruning-ratio', type=float, default=0.3,
                        help='Fraction of weights to prune (default: 0.3 = 30%%)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run on (default: cpu)')
    
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"❌ Error: Model file not found: {args.model}")
        return 1
    
    compress_model(
        model_path=args.model,
        output_path=args.output,
        quantization=args.quantization,
        pruning=args.pruning,
        pruning_ratio=args.pruning_ratio,
        device=args.device,
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

