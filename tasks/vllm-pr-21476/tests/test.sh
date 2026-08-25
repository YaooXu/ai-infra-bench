#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier
if cd /workspace/repo && python3 -I /tests/verify_int8_quant.py --mode candidate; then
  printf '1\n' > /logs/verifier/reward.txt
else
  printf '0\n' > /logs/verifier/reward.txt
fi

