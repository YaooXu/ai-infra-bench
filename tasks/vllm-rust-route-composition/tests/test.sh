#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier
cd /workspace/vllm
rc=0
mkdir -p rust/src/server/tests
cp /tests/router_extension_api.rs rust/src/server/tests/router_extension_api.rs
timeout 900 bash -lc "cargo nextest run --manifest-path rust/Cargo.toml -p vllm-server --test router_extension_api --locked --no-fail-fast" || rc=$?
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
