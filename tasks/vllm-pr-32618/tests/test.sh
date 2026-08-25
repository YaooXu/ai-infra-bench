#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier
cd /workspace/repo || exit 1
if python3 /tests/verify_async_pp.py --check-api && \
   python3 -m torch.distributed.run --nnodes=1 --nproc-per-node=2 \
     --master-addr=127.0.0.1 --master-port=29618 /tests/verify_async_pp.py; then
  printf '1\n' > /logs/verifier/reward.txt
else
  printf '0\n' > /logs/verifier/reward.txt
fi
