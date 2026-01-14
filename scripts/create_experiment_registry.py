#!/usr/bin/env python3
"""Create and maintain an experiment registry for all training experiments."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from tiny_icf.flexible_lightning_module import FlexibleIDFLightningModule
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False


def extract_experiment_configs() -> List[Dict[str, Any]]:
    """Extract all experiment configurations from train_flexible_opportunistic.py."""
    script_path = PROJECT_ROOT.parent / "trainctl" / "training" / "scripts" / "train_flexible_opportunistic.py"
    
    if not script_path.exists():
        print(f"⚠️  Training script not found: {script_path}")
        return []
    
    # Read the script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find all configs.append blocks
    import re
    configs = []
    
    # Pattern to match config dictionaries
    pattern = r"configs\.append\(\{([^}]+(?:\{[^}]*\}[^}]*)*)\}\)"
    
    # More robust: find configs.append and extract the dict
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        if "configs.append({" in lines[i]:
            # Collect the config dict
            config_lines = []
            brace_count = 0
            j = i
            while j < len(lines):
                line = lines[j]
                config_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0 and '})' in line:
                    break
                j += 1
            
            # Extract name and key fields
            config_text = '\n'.join(config_lines)
            name_match = re.search(r"'name':\s*'([^']+)'", config_text)
            desc_match = re.search(r"'description':\s*'([^']+)'", config_text)
            aim_match = re.search(r"'aim_experiment':\s*'([^']+)'", config_text)
            
            if name_match:
                config = {
                    'name': name_match.group(1),
                    'description': desc_match.group(1) if desc_match else 'No description',
                    'aim_experiment': aim_match.group(1) if aim_match else 'icf-training',
                    'category': _categorize_experiment(name_match.group(1)),
                }
                
                # Extract key config values
                for key in ['model_type', 'use_research_aligned_loss', 'use_unified_loss', 
                           'use_distillation', 'spearman_weight', 'rank_weight', 
                           'dropout', 'weight_decay', 'lr', 'batch_size', 'epochs']:
                    match = re.search(rf"'{key}':\s*([^,\n]+)", config_text)
                    if match:
                        val = match.group(1).strip()
                        # Try to parse as appropriate type
                        try:
                            if val.replace('.', '').replace('-', '').isdigit():
                                config[key] = eval(val)  # Safe for numbers/booleans
                            elif val in ['True', 'False']:
                                config[key] = val == 'True'
                            else:
                                config[key] = val
                        except:
                            config[key] = val
                
                configs.append(config)
            
            i = j + 1
        else:
            i += 1
    
    return configs


def _categorize_experiment(name: str) -> str:
    """Categorize experiment based on name."""
    if 'research_aligned' in name:
        return 'research_aligned'
    elif 'distillation' in name:
        return 'distillation'
    elif 'multitask' in name:
        return 'multitask'
    elif 'residual' in name:
        return 'residual'
    elif 'standard' in name or 'baseline' in name:
        return 'baseline'
    else:
        return 'other'


def load_existing_results() -> Dict[str, Any]:
    """Load existing experiment results from model directories."""
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        return {}
    
    results = {}
    for exp_dir in models_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        
        # Check for metrics CSV
        metrics_csv = exp_dir / "lightning_logs" / "version_0" / "metrics.csv"
        if metrics_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(metrics_csv)
                val_df = df[df['val_spearman_corr'].notna()]
                if len(val_df) > 0:
                    best_spearman = val_df['val_spearman_corr'].max()
                    final_spearman = val_df['val_spearman_corr'].iloc[-1] if len(val_df) > 0 else None
                    results[exp_name] = {
                        'best_spearman': float(best_spearman),
                        'final_spearman': float(final_spearman) if final_spearman is not None else None,
                        'epochs_trained': len(val_df),
                        'status': 'completed' if len(val_df) >= 10 else 'in_progress',
                    }
            except Exception as e:
                results[exp_name] = {'status': 'error', 'error': str(e)}
        else:
            # Check if training is in progress
            log_file = exp_dir / "training.log"
            if log_file.exists():
                results[exp_name] = {'status': 'in_progress'}
            else:
                results[exp_name] = {'status': 'not_started'}
    
    return results


def create_registry() -> Dict[str, Any]:
    """Create complete experiment registry."""
    configs = extract_experiment_configs()
    results = load_existing_results()
    
    registry = {
        'created_at': datetime.now().isoformat(),
        'total_experiments': len(configs),
        'experiments': {},
    }
    
    for config in configs:
        exp_name = config['name']
        registry['experiments'][exp_name] = {
            'config': config,
            'results': results.get(exp_name, {'status': 'not_started'}),
        }
    
    return registry


def main():
    """Main entry point."""
    registry = create_registry()
    
    # Save registry
    registry_path = PROJECT_ROOT / "models" / "experiment_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Experiment registry created: {registry_path}")
    print(f"   Total experiments: {registry['total_experiments']}")
    
    # Print summary by category
    categories = {}
    for exp_name, exp_data in registry['experiments'].items():
        category = exp_data['config'].get('category', 'other')
        if category not in categories:
            categories[category] = []
        categories[category].append(exp_name)
    
    print("\n📊 Experiments by category:")
    for category, exp_names in sorted(categories.items()):
        print(f"   {category}: {len(exp_names)}")
        for name in exp_names[:5]:  # Show first 5
            status = registry['experiments'][name]['results'].get('status', 'unknown')
            print(f"      - {name} ({status})")
        if len(exp_names) > 5:
            print(f"      ... and {len(exp_names) - 5} more")
    
    # Print research-aligned experiments
    ra_experiments = [name for name in registry['experiments'].keys() if 'research_aligned' in name]
    if ra_experiments:
        print(f"\n🔬 Research-aligned experiments: {len(ra_experiments)}")
        for name in ra_experiments:
            status = registry['experiments'][name]['results'].get('status', 'not_started')
            print(f"   - {name} ({status})")


if __name__ == '__main__':
    main()

