#!/usr/bin/env bash
set -uo pipefail

cd /app
python3 /tests/grader.py
