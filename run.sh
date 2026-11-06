#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$root/.venv/bin/python" ]]; then
  python="$root/.venv/bin/python"
elif [[ -x "$root/.venv/Scripts/python.exe" ]]; then
  python="$root/.venv/Scripts/python.exe"
else
  python="${PYTHON:-python3}"
fi
export HF_HUB_OFFLINE=1
exec "$python" "$root/code/safe_run.py" "$@"
