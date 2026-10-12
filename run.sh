#!/usr/bin/env bash
# Local dev server for the lazuli-exo webapp. Usage: ./run.sh [port]
# Binds to 127.0.0.1 only. Set FLASK_DEBUG=1 yourself if you want the
# Werkzeug debugger; it allows code execution from the browser, so never
# combine it with --host 0.0.0.0.
set -euo pipefail
cd "$(dirname "$0")"
exec flask --app webapp/app run --host 127.0.0.1 --port "${1:-5000}"
