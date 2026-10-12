#!/usr/bin/env bash
# The wheel must ship exactly the tables git tracks: a stale local build/ dir
# once leaked 70 deleted tables into a wheel (#31). Run from the repo root
# with the built wheel as $1.
set -euo pipefail
wheel="$1"
in_wheel=$(unzip -l "$wheel" | grep -c 'lazuli_exo/data/tables/.*\.npz$' || true)
in_git=$(git ls-files src/lazuli_exo/data/tables | grep -c '\.npz$' || true)
echo "tables in wheel: $in_wheel; tracked in git: $in_git"
if [ "$in_wheel" -ne "$in_git" ] || [ "$in_git" -eq 0 ]; then
  echo "::error::wheel table count ($in_wheel) != git ($in_git); build from a clean checkout"
  exit 1
fi
