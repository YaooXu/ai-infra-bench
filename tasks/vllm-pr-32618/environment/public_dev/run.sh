#!/usr/bin/env bash
set -euo pipefail

python3 /workspace/public_dev/reproduce_async_pp_token_broadcast.py --check-api
torchrun --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29518 \
  /workspace/public_dev/reproduce_async_pp_token_broadcast.py
