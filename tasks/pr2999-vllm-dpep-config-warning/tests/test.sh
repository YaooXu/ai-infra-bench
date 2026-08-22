#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier
cd /workspace/repo

if python -m pytest -q -p no:cacheprovider /tests/required /tests/heldout; then
  echo 1 > /logs/verifier/reward.txt
  exit 0
fi

echo 0 > /logs/verifier/reward.txt
exit 1
