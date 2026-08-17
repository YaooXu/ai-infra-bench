#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

analysis_root="analysis/RQ1"
tagging_input="artifacts/rq1/vllm_pr_tagging/output/vllm_merged_prs_2026-02-01_2026-07-31.tagging_input.ci_complete.jsonl"
: "${RQ1_TAGGING_SNAPSHOT:?Set RQ1_TAGGING_SNAPSHOT to the frozen task snapshot}"
tagging_snapshot="${RQ1_TAGGING_SNAPSHOT}"

python3 "${analysis_root}/scripts/extract_tagging_snapshot.py" \
  --input "${tagging_input}" \
  --snapshot "${tagging_snapshot}" \
  --output "${analysis_root}/data/tagging_compact.jsonl" \
  --audit "${analysis_root}/manifests/tagging_snapshot_audit.json" \
  --task-id 348689 \
  --snapshot-as-of '2026-08-17T00:26:17+08:00' \
  --snapshot-artifact 'frozen-task-348689-snapshot-20260817' \
  --expected-input-sha256 936f6fe84f484005bd5b32b797ba8ae9b0f2ebc931c9cc9240327746b3fb5626

uv run --python 3.12 \
  --with pandas==2.3.1 \
  --with numpy==2.5.2 \
  --with matplotlib==3.10.5 \
  --with seaborn==0.13.2 \
  --with pyyaml==6.0.3 \
  python "${analysis_root}/scripts/analyze_rq1.py"

python3 "${analysis_root}/scripts/validate_outputs.py"
