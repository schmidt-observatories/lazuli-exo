#!/usr/bin/env bash
# Live-reloading docs server (like wcc-etc's docs/rundocs.sh). Falls back to a
# one-shot build if sphinx-autobuild is not installed.
set -euo pipefail
cd "$(dirname "$0")/sphinx"
if python -c "import sphinx_autobuild" 2>/dev/null; then
  python -m sphinx_autobuild . _build/html --open-browser
else
  make html && echo "open $(pwd)/_build/html/index.html"
fi
