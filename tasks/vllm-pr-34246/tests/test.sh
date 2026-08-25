#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier
log=/logs/verifier/verify_multimodal_merge.log

if cd /workspace/repo \
    && PYTHONPATH=/workspace/repo python3 /tests/verify_multimodal_merge.py \
        >"${log}" 2>&1 \
    && ! grep -Eiq \
        'registered multiple times|overriding a previously registered kernel|duplicate registration' \
        "${log}"; then
    cat "${log}"
    printf '1\n' > /logs/verifier/reward.txt
else
    rc=$?
    cat "${log}" 2>/dev/null || true
    printf '0\n' > /logs/verifier/reward.txt
    exit "${rc:-1}"
fi
