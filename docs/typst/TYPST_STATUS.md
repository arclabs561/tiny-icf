# Typst Documentation Status

## Created Files

1. **`loss_component_bounds.typ`** - Theoretical bounds for all loss components
2. **`ceiling_analysis.typ`** - Analysis of the performance ceiling (~0.18-0.19 Spearman)
3. **`multi_task_bounds.typ`** - Bounds for multi-task learning outputs

## Build System

- **`scripts/build_typst_docs.sh`** - Build script for all Typst documents
- **`justfile`** - Added `build-docs` command

## Current Status

The Typst files are structurally complete but have some syntax issues with variable names in math mode. Typst interprets certain identifiers (like `rank`, `sign`, `margin`, `if`) as reserved words or functions.

### Remaining Issues

1. Variable name conflicts in math mode need to be quoted
2. Some function names (`sign`, `max`, `relu`) may need special handling
3. The `cases()` function syntax may need adjustment

### Quick Fixes Needed

For math mode variables, use quotes:
- `L_rank` → `L_"rank"`
- `sign(x)` → `"sign"(x)` or use `text(sign)(x)`

For conditional expressions, use text mode or different notation:
- `if condition` → `"when" condition` or use separate equations

## Benefits

Once compiled, these Typst documents will provide:

1. **Better math rendering** than Markdown + KaTeX
2. **Native PDF generation** with proper typesetting
3. **Code readability** - easier for LLMs to parse mathematical constraints
4. **Professional output** for documentation and papers

## Next Steps

1. Fix remaining syntax issues (variable name quoting)
2. Test PDF generation
3. Set up HTML export (if typst web is available)
4. Integrate into documentation workflow

## Usage

```bash
# Build all documents
just build-docs

# Or manually
typst compile docs/typst/loss_component_bounds.typ docs/typst/output/loss_component_bounds.pdf
```

