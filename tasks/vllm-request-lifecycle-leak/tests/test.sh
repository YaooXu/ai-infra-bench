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
  echo "trusted verifier Python is unavailable" >&2
  exit 1
fi
cd /workspace/repo
# Harbor bind-mounts /tests from the host checkout, so these scripts carry the
# host user's ownership: root where the checkout is root-owned, but uid 1001 on
# GitHub Actions runners. The supervisor requires root-owned, non-writable
# scripts, so stage a private root-owned copy instead of trusting the mount.
stage=/opt/ai-infra-verifier
install -d -m 0755 -o 0 -g 0 "$stage"
install -m 0644 -o 0 -g 0 \
  /tests/supervise_retention.py /tests/verify_retention.py "$stage/"

exec "$python_bin" -I "$stage/supervise_retention.py" "$python_bin"
