#!/usr/bin/env bash
# Build the manuscript PDF. Run `python paper/make_assets.py` first so that generated/ and figures/ are current.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
run() { pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build sarra.tex > build/pdflatex.log; }
run
bibtex build/sarra > build/bibtex.log
run
run
# Repeat until cross-references are stable.
for _ in 1 2 3; do
  grep -q "Rerun to get cross-references right" build/sarra.log || break
  run
done
cp build/sarra.pdf ../docs/sarra.pdf
# Earlier links to the manuscript point at these names.
cp build/sarra.pdf ../docs/emnlp-manuscript.pdf
cp build/sarra.pdf ../docs/manuscript.pdf
grep -E "Warning|undefined|Overfull" build/sarra.log | grep -v "Font shape" || true
echo "wrote docs/sarra.pdf"
