# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
# ]
# ///
"""Quick test of different model variants to compare parameter counts and basic functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from tiny_icf.model import UniversalICF
from tiny_icf.nano_model import NanoICF
from tiny_icf.model_hierarchical import HierarchicalICF, BoxEmbeddingICF

def test_model(model, name, dummy_input):
    """Test a model variant."""
    try:
        params = model.count_parameters()
        output = model(dummy_input)
        output_range = (output.min().item(), output.max().item())
        output_mean = output.mean().item()
        return {
            'name': name,
            'params': params,
            'output_range': output_range,
            'output_mean': output_mean,
            'status': 'OK'
        }
    except Exception as e:
        return {
            'name': name,
            'params': 0,
            'status': f'ERROR: {str(e)}'
        }

def main():
    dummy_input = torch.randint(0, 256, (4, 10))  # Batch of 4, max length 10
    
    models_to_test = [
        (UniversalICF(emb_dim=48, conv_channels=24, hidden_dim=48, dropout=0.3), 'UniversalICF (original)'),
        (UniversalICF(emb_dim=36, conv_channels=18, hidden_dim=36, dropout=0.4), 'UniversalICF (reduced)'),
        (UniversalICF(emb_dim=36, conv_channels=18, hidden_dim=36, dropout=0.4), 'UniversalICF (BatchNorm)'),
        (NanoICF(), 'NanoICF'),
        (HierarchicalICF(), 'HierarchicalICF'),
        (BoxEmbeddingICF(), 'BoxEmbeddingICF'),
    ]
    
    print("=" * 80)
    print("MODEL VARIANT COMPARISON")
    print("=" * 80)
    print()
    
    results = []
    for model, name in models_to_test:
        result = test_model(model, name, dummy_input)
        results.append(result)
        
        print(f"{result['name']:<35} {result['params']:>8,} params", end="")
        if result['status'] == 'OK':
            print(f"  Range: [{result['output_range'][0]:.4f}, {result['output_range'][1]:.4f}], Mean: {result['output_mean']:.4f}")
        else:
            print(f"  {result['status']}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    working_models = [r for r in results if r['status'] == 'OK']
    if working_models:
        smallest = min(working_models, key=lambda x: x['params'])
        largest = max(working_models, key=lambda x: x['params'])
        print(f"Smallest: {smallest['name']} ({smallest['params']:,} params)")
        print(f"Largest: {largest['name']} ({largest['params']:,} params)")
        print(f"Reduction potential: {((largest['params'] - smallest['params']) / largest['params'] * 100):.1f}%")

if __name__ == "__main__":
    main()

