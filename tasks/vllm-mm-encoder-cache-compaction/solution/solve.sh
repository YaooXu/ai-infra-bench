#!/bin/sh
set -eu

WORKSPACE="${BENCH_WORKSPACE:-/app}"
SOLUTION_DIR="${BENCH_SOLUTION_DIR:-/solution}"

cd "$WORKSPACE"
git apply --unidiff-zero --check "$SOLUTION_DIR/hardened-reference.patch"
git apply --unidiff-zero "$SOLUTION_DIR/hardened-reference.patch"
git diff --check
