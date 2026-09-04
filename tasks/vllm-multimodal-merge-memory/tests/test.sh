#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier
log=/logs/verifier/verify_multimodal_merge.log
python_bin=
for candidate in /opt/venv/bin/python /usr/local/bin/python /usr/local/bin/python3 /usr/bin/python3; do
    if [ -x "$candidate" ] && [ "$(stat -Lc '%U:%G' "$candidate")" = "root:root" ]; then
        python_bin="$candidate"
        break
    fi
done

if [ -z "$python_bin" ]; then
    printf '0\n' > /logs/verifier/reward.txt
    exit 0
fi
if cd /workspace/repo \
    && "$python_bin" -I /tests/verify_multimodal_merge.py \
        >"${log}" 2>&1; then
    cat "${log}"
    printf '1\n' > /logs/verifier/reward.txt
else
    cat "${log}" 2>/dev/null || true
    printf '0\n' > /logs/verifier/reward.txt
fi
