# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Script to consolidate evaluation functions into eval.py.

This moves functions from eval_calibration.py and eval_stratified.py
into eval.py to reduce fragmentation.
"""

import re
from pathlib import Path

def consolidate_eval():
    """Consolidate eval files into eval.py."""
    
    eval_file = Path("src/tiny_icf/eval.py")
    calib_file = Path("src/tiny_icf/eval_calibration.py")
    strat_file = Path("src/tiny_icf/eval_stratified.py")
    
    # Read eval.py
    with open(eval_file, 'r') as f:
        eval_content = f.read()
    
    # Read calibration functions
    with open(calib_file, 'r') as f:
        calib_content = f.read()
        # Extract function definitions (skip imports and docstrings)
        calib_functions = []
        in_function = False
        current_function = []
        indent_level = 0
        
        for line in calib_content.split('\n'):
            if line.strip().startswith('def ') and not in_function:
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                current_function = [line]
            elif in_function:
                if line.strip() == '' or line.startswith(' ' * indent_level) or line.startswith('\t'):
                    current_function.append(line)
                else:
                    # Function ended
                    calib_functions.append('\n'.join(current_function))
                    current_function = []
                    in_function = False
                    if line.strip().startswith('def '):
                        in_function = True
                        indent_level = len(line) - len(line.lstrip())
                        current_function = [line]
        
        if current_function:
            calib_functions.append('\n'.join(current_function))
    
    # Read stratified functions
    with open(strat_file, 'r') as f:
        strat_content = f.read()
        # Similar extraction for stratified functions
        strat_functions = []
        in_function = False
        current_function = []
        indent_level = 0
        
        for line in strat_content.split('\n'):
            if line.strip().startswith('def ') and not in_function:
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                current_function = [line]
            elif in_function:
                if line.strip() == '' or line.startswith(' ' * indent_level) or line.startswith('\t'):
                    current_function.append(line)
                else:
                    strat_functions.append('\n'.join(current_function))
                    current_function = []
                    in_function = False
                    if line.strip().startswith('def '):
                        in_function = True
                        indent_level = len(line) - len(line.lstrip())
                        current_function = [line]
        
        if current_function:
            strat_functions.append('\n'.join(current_function))
    
    # Add consolidated functions to eval.py
    # Insert before the last return statement or at the end
    
    # Find insertion point (before last function or at end)
    insertion_point = eval_content.rfind('\n\n')
    if insertion_point == -1:
        insertion_point = len(eval_content)
    
    # Add section header and functions
    consolidated = eval_content[:insertion_point]
    consolidated += "\n\n# ============================================================================\n"
    consolidated += "# Calibration Metrics (consolidated from eval_calibration.py)\n"
    consolidated += "# ============================================================================\n\n"
    consolidated += '\n\n'.join(calib_functions)
    consolidated += "\n\n# ============================================================================\n"
    consolidated += "# Stratified Evaluation (consolidated from eval_stratified.py)\n"
    consolidated += "# ============================================================================\n\n"
    consolidated += '\n\n'.join(strat_functions)
    consolidated += eval_content[insertion_point:]
    
    # Write back
    with open(eval_file, 'w') as f:
        f.write(consolidated)
    
    print("✅ Consolidated evaluation functions into eval.py")
    print(f"   Added {len(calib_functions)} calibration functions")
    print(f"   Added {len(strat_functions)} stratified functions")
    print("\n⚠️  Next steps:")
    print("   1. Update imports in eval.py to include calibration/stratified dependencies")
    print("   2. Update eval.py to use consolidated functions directly")
    print("   3. Archive eval_calibration.py and eval_stratified.py")
    print("   4. Test that everything still works")

if __name__ == "__main__":
    consolidate_eval()

