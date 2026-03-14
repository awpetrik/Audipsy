#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[quick-native][error] Python not found: $PYTHON_BIN" >&2
  echo "[quick-native][hint] Install Python 3.10+ or set PYTHON env var (e.g. PYTHON=python3.11)." >&2
  exit 1
fi

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" quick_native.py "$@"
