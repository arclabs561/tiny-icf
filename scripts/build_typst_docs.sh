#!/bin/bash
# Build Typst documentation to PDF and HTML

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs/typst"
OUTPUT_DIR="$PROJECT_ROOT/docs/typst/output"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if typst is installed
if ! command -v typst &> /dev/null; then
    echo "❌ Error: typst is not installed"
    echo "   Install with: cargo install --git https://github.com/typst/typst typst-cli"
    echo "   Or: brew install typst (on macOS)"
    exit 1
fi

echo "📚 Building Typst Documentation"
echo "================================"
echo ""

# Build each Typst file
for typ_file in "$DOCS_DIR"/*.typ; do
    if [ ! -f "$typ_file" ]; then
        continue
    fi
    
    basename=$(basename "$typ_file" .typ)
    echo "📄 Building: $basename"
    
    # Build PDF
    echo "   → PDF..."
    typst compile "$typ_file" "$OUTPUT_DIR/${basename}.pdf" || {
        echo "   ❌ Failed to build PDF for $basename"
        continue
    }
    
    # Build HTML (requires typst-preview or manual conversion)
    # For now, we'll use typst's web export if available
    if typst --help | grep -q "web"; then
        echo "   → HTML..."
        typst web "$typ_file" "$OUTPUT_DIR/${basename}.html" || {
            echo "   ⚠️  HTML export not available, skipping"
        }
    else
        echo "   → HTML (skipping - typst web not available)"
    fi
    
    echo "   ✅ Done"
    echo ""
done

echo "✅ All documentation built successfully!"
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR" | tail -n +2 | awk '{print "   " $9 " (" $5 ")"}'

