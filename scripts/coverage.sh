#!/bin/bash

# Local code coverage script for vidoc
# Requires: cargo-tarpaulin

set -e

echo "Installing cargo-tarpaulin (if not already installed)..."
cargo install cargo-tarpaulin --quiet || true

echo "Running tests with coverage..."
cargo tarpaulin \
    --out Html \
    --out Xml \
    --all-features \
    --workspace \
    --timeout 300 \
    --exclude-files "examples/*" \
    --exclude-files "benches/*" \
    --exclude-files "tests/*"

echo ""
echo "✅ Coverage report generated!"
echo "📊 HTML report: tarpaulin-report.html"
echo "📄 XML report: cobertura.xml"
echo ""
echo "Open HTML report with:"
echo "  open tarpaulin-report.html  # macOS"
echo "  xdg-open tarpaulin-report.html  # Linux"
