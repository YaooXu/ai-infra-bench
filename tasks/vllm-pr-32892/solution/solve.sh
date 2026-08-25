#!/usr/bin/env bash
set -euo pipefail
cd /app
git apply /solution/fix.patch
/opt/bench/rebuild_native.sh

