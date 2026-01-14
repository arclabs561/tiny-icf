# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Update imports after consolidation.

This script updates imports in scripts that reference archived files
to use the consolidated versions instead.
"""

import re
from pathlib import Path

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    # Loss files
    "from tiny_icf.loss import": "from tiny_icf.loss import",
    "from tiny_icf.loss import": "from tiny_icf.loss import",
    "from tiny_icf.loss import": "from tiny_icf.loss import",  # Note: CombinedLoss  # TODO: Review - migrated from EnhancedMultiLoss not consolidated yet
    "from tiny_icf.loss import": "from tiny_icf.loss import",  # Note: CombinedLoss  # TODO: Review - migrated from DifferentiableSortingLoss not consolidated yet
    
    # Predict files
    "from tiny_icf.predict_consolidated import": "from tiny_icf.predict_consolidated import",
    "from tiny_icf.predict_consolidated import": "from tiny_icf.predict_consolidated import",
}

# Specific class/function mappings
CLASS_MAPPINGS = {
    # Loss classes
    "CombinedLoss  # TODO: Review - migrated from ResearchBasedLoss": "CombinedLoss",  # Approximate - may need manual adjustment
    "CombinedLoss  # TODO: Review - migrated from CombinedListwiseLoss": "CombinedLoss",  # With use_listwise=True
    "CombinedLoss  # TODO: Review - migrated from EnhancedMultiLoss": "CombinedLoss",  # Approximate - may need manual adjustment
    "CombinedLoss  # TODO: Review - migrated from DifferentiableSortingLoss": "CombinedLoss",  # Not directly supported - may need manual adjustment
    
    # Predict functions
    "predict_batch  # TODO: Review - migrated from predict_batch": "predict_batch  # TODO: Review - migrated from predict_batch",  # Same name in consolidated
    "predict  # TODO: Review - migrated from predict_with_analysis": "predict",  # With advanced=True
}

def update_file(file_path: Path) -> bool:
    """Update imports in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception:
        return False
    
    original_content = content
    updated = False
    
    # Update import statements
    for old_import, new_import in IMPORT_MAPPINGS.items():
        if old_import in content:
            content = content.replace(old_import, new_import)
            updated = True
    
    # Update class references (basic - may need manual review)
    for old_class, new_class in CLASS_MAPPINGS.items():
        # Only replace if it's a class instantiation or type hint
        pattern = rf'\b{old_class}\b'
        if re.search(pattern, content):
            # Add comment about manual review needed
            content = re.sub(
                pattern,
                f'{new_class}  # TODO: Review - migrated from {old_class}',
                content
            )
            updated = True
    
    if updated and content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    
    return False

def main():
    print("Updating imports after consolidation...")
    print("=" * 70)
    
    # Find all Python files in scripts/
    scripts_dir = Path("scripts")
    python_files = list(scripts_dir.glob("*.py"))
    
    updated_count = 0
    for script_file in python_files:
        if update_file(script_file):
            print(f"  ✅ Updated {script_file.name}")
            updated_count += 1
    
    print("=" * 70)
    print(f"✅ Updated {updated_count} files")
    print("\n⚠️  Note: Some class mappings may need manual review:")
    print("   - CombinedLoss  # TODO: Review - migrated from ResearchBasedLoss → CombinedLoss (may need different config)")
    print("   - CombinedLoss  # TODO: Review - migrated from EnhancedMultiLoss → CombinedLoss (may need different config)")
    print("   - CombinedLoss  # TODO: Review - migrated from DifferentiableSortingLoss → Not directly supported in CombinedLoss")
    print("\n   Please review updated files and adjust as needed.")

if __name__ == "__main__":
    main()

