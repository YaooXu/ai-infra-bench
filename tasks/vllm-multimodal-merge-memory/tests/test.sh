#!/usr/bin/env bash
set -euo pipefail
python_bin=
for candidate in /opt/venv/bin/python /usr/local/bin/python /usr/local/bin/python3 /usr/bin/python3; do
  if [ -x "$candidate" ] && [ "$(stat -Lc '%U:%G' "$candidate")" = "root:root" ]; then
    python_bin="$candidate"
    break
  fi
done
if [ -z "$python_bin" ]; then
  exit 0
fi
cd /workspace/repo
exec "$python_bin" -I /tests/supervise_multimodal_merge.py "$python_bin"
