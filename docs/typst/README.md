# Typst Documentation

This directory contains Typst-formatted documentation for the ICF estimation project, focusing on mathematical content that benefits from proper typesetting.

## Files

### Core Documents

- `loss_component_bounds.typ` - Theoretical bounds for all loss components
- `ceiling_analysis.typ` - Analysis of the performance ceiling (~0.18-0.19 Spearman)
- `multi_task_bounds.typ` - Bounds for multi-task learning outputs
- `problem_motivation.typ` - Problem motivation, use cases, and the wonder of generalization
- `theoretical_foundations.typ` - Information theory, Kolmogorov complexity, and fundamental constraints
- `technical_mathematical_description.typ` - Complete technical mathematical description
- `design_philosophy.typ` - Design philosophy: learning through experimentation

## Building

### Prerequisites

Install Typst:

```bash
# macOS
brew install typst

# Or via cargo
cargo install --git https://github.com/typst/typst typst-cli
```

### Build All Documents

```bash
./scripts/build_typst_docs.sh
```

Or via justfile:

```bash
just build-docs
```

This will generate PDF files in `docs/typst/output/`.

### Build Individual Documents

```bash
# Build PDF
typst compile docs/typst/loss_component_bounds.typ docs/typst/output/loss_component_bounds.pdf

# Build HTML (if typst web is available)
typst web docs/typst/loss_component_bounds.typ docs/typst/output/loss_component_bounds.html
```

## Current Status

✅ **Successfully Compiling:**
- `multi_task_bounds.pdf` (111K)
- `theoretical_foundations.pdf` (164K) - Enhanced with provenance and motivations

⚠️ **In Progress:**
- `loss_component_bounds.typ` - Minor syntax fixes needed (variable name conflicts in math mode)
- `ceiling_analysis.typ` - Minor syntax fixes needed
- `problem_motivation.typ` - Enhanced with Jabberwocky Protocol, generalization examples
- `technical_mathematical_description.typ` - Minor syntax fixes needed
- `design_philosophy.typ` - New document on experimental approach

## Content Enhancements

The Typst documents now include:

1. **Provenance and Motivations:**
   - The "flimjam" example demonstrating generalization
   - Jabberwocky Protocol for testing generalization
   - Why we don't use nearest-word mapping
   - The fun project philosophy

2. **Theoretical Foundations:**
   - Fundamental questions about feasibility
   - Why compression is marginal but generalization matters
   - The regularity assumption
   - Information-theoretic bounds

3. **Design Philosophy:**
   - Learning > perfection
   - Experimentation > optimization
   - Interesting results > perfect metrics

## Why Typst?

Typst provides:
- **Better math rendering** than Markdown + KaTeX
- **Native PDF generation** with proper typesetting
- **Code readability** - easier for LLMs to parse mathematical constraints
- **Professional output** for documentation and papers

## Integration

These Typst documents complement the Markdown documentation:
- Markdown docs are easier to edit and view in GitHub
- Typst docs provide formal mathematical typesetting
- Both are kept in sync for different use cases

## Known Issues

Some Typst syntax issues remain:
- Variable name conflicts in math mode (e.g., `rank`, `sign`, `max`, `mean` need quoting)
- Function names in math mode need special handling
- Some content blocks may need simplification

These are being fixed incrementally. The structure and content are complete.

## Style Principles

The documents follow principles from:
- **Chris Olah**: Visual, stepwise explanations with wonder and curiosity
- **Martin Gardner**: Clear exposition of paradox and mystery, friendly narratives
- **Julia Evans**: Approachable, demystifying, admits gaps and struggles
- **Donald Knuth**: Mathematical rigor with practical beauty, complementary expression
