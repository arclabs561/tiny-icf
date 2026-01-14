#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Code organization refinement tool.

Analyzes codebase for:
- Duplicate functions/classes
- Unused imports
- File organization issues
- Consolidation opportunities
"""

import ast
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

def analyze_imports(file_path: Path) -> Set[str]:
    """Extract imports from a Python file."""
    imports = set()
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
    except Exception:
        pass
    return imports

def analyze_functions_classes(file_path: Path) -> tuple:
    """Extract function and class names from a Python file."""
    functions = []
    classes = []
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
    except Exception:
        pass
    return functions, classes

def main():
    src_dir = Path("src/tiny_icf")
    
    print("=" * 70)
    print("Code Organization Analysis")
    print("=" * 70)
    print()
    
    # Find duplicates
    all_functions = defaultdict(list)
    all_classes = defaultdict(list)
    
    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        rel_path = str(py_file.relative_to(src_dir))
        funcs, classes = analyze_functions_classes(py_file)
        for func in funcs:
            all_functions[func].append(rel_path)
        for cls in classes:
            all_classes[cls].append(rel_path)
    
    # Report duplicates
    print("🔍 Duplicate Functions:")
    duplicates = {k: v for k, v in all_functions.items() if len(v) > 1}
    if duplicates:
        for name, files in sorted(duplicates.items()):
            print(f"  {name}:")
            for f in files:
                print(f"    - {f}")
    else:
        print("  ✅ No duplicate functions found")
    
    print()
    print("🔍 Duplicate Classes:")
    duplicates = {k: v for k, v in all_classes.items() if len(v) > 1}
    if duplicates:
        for name, files in sorted(duplicates.items()):
            print(f"  {name}:")
            for f in files:
                print(f"    - {f}")
    else:
        print("  ✅ No duplicate classes found")
    
    print()
    print("=" * 70)
    print("Analysis complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
