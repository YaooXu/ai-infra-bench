#!/usr/bin/env bash
set -euo pipefail

cd /app
git apply /solution/plausible-wrong.patch
