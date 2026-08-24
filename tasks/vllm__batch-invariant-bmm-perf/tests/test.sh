#!/bin/bash
# Harbor verifier entrypoint.
# Correctness is a hard gate: if it fails, reward=0 and we short-circuit.
# Otherwise we compute a continuous performance reward in [0,1].
# Final score is written to /logs/verifier/reward.txt.
set +e
mkdir -p /logs/verifier

cd /app/vllm 2>/dev/null || cd /app

# --- Gate 1: correctness (bitwise batch-invariance) ---
# 容器基础镜像只提供 python3（无 python 别名），统一用 python3。
PY=$(command -v python3 || command -v python)
"$PY" /tests/required/test_correctness.py
CORRECT=$?

if [ "$CORRECT" -ne 0 ]; then
  echo "0" > /logs/verifier/reward.txt
  echo "[verifier] correctness gate FAILED -> reward=0"
  exit 0
fi
echo "[verifier] correctness gate PASSED"

# --- Gate 2: continuous performance reward vs frozen baselines ---
"$PY" /tests/heldout/perf_reward.py \
    --baselines /tests/baselines.json \
    --out /logs/verifier/reward.txt \
    --report /logs/verifier/perf_report.json

echo "[verifier] final reward:"
cat /logs/verifier/reward.txt
exit 0
