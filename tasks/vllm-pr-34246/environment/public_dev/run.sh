#!/usr/bin/env bash
set -euo pipefail

cd /workspace/repo
exec python3 /workspace/public_dev/reproduce_multimodal_merge.py
