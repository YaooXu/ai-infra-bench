#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier
log=/logs/verifier/verify_multimodal_merge.log

if cd /workspace/repo \
    && PYTHONPATH=/workspace/repo python3 /tests/verify_multimodal_merge.py \
        >"${log}" 2>&1; then
    cat "${log}"
    printf '1\n' > /logs/verifier/reward.txt
else
    cat "${log}" 2>/dev/null || true
    printf '0\n' > /logs/verifier/reward.txt
fi
