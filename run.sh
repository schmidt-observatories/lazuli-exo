#!/usr/bin/env bash
# Dev server for the terminator webapp. Usage: ./run.sh [port]
set -euo pipefail
cd "$(dirname "$0")"
exec flask --app webapp/app run --debug --port "${1:-5000}"
