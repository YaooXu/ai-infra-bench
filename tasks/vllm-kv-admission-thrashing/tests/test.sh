#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier
cd /workspace/vllm
rc=0
timeout 900 bash -lc "pytest -v -s /tests/test_regression.py" || rc=$?
if [ "$rc" -eq 0 ]; then
  printf '1
' > /logs/verifier/reward.txt
else
  printf '0
' > /logs/verifier/reward.txt
fi
printf '{"reward":%s,"command_exit_code":%s}
' "$([ "$rc" -eq 0 ] && printf 1 || printf 0)" "$rc" > /logs/verifier/reward.json
exit 0
