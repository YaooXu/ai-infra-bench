#!/usr/bin/env bash
set -euo pipefail

cd /workspace/repo
exec python3 /opt/ai-infra-bench/dev/reproduce_retention.py
