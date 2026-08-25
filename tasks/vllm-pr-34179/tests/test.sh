#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier
if cd /workspace/repo && python3 /tests/verify_dcp_slot_mapping.py; then
  printf '1\n' > /logs/verifier/reward.txt
else
  printf '0\n' > /logs/verifier/reward.txt
fi
