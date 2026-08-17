# vLLM RQ1: real engineering workload

This directory is the self-contained, release-aligned RQ1 analysis workspace.
The central question is:

> What work does vLLM engineering actually consist of?

The answer is built in two layers. The repository-wide layer measures the full
public work envelope (Issues, incoming PRs, contributors, outcomes, review and
backlog). The semantic layer characterizes every merged PR from 2026-02 through
2026-07 by engineering intent, project surface, architecture component and
affected hardware backend. The first layer prevents us from equating “work” with
merged code; the second gives RQ2 a technically meaningful workload distribution.

## Notion page

The canonical long-form publication is maintained directly in Notion. The
Notion export and local publication workspace are intentionally not committed.

- Title: `Can Frontier Coding Agents Solve Real-World Engineering Tasks in vLLM?`
- URL: <https://app.notion.com/p/Can-Frontier-Coding-Agents-Solve-Real-World-Engineering-Tasks-in-vLLM-3bf70448c2e1819982c9f80ca6325c8a>
- Page ID: `3bf70448-c2e1-8199-82c9-f80ca6325c8a`

Always read the current page before editing it, then update the same Page ID
with the Notion CLI:

```bash
export RQ1_NOTION_PAGE_ID=3bf70448-c2e1-8199-82c9-f80ca6325c8a
NOTION_KEYRING=0 ntn pages get "$RQ1_NOTION_PAGE_ID"
NOTION_KEYRING=0 ntn pages edit "$RQ1_NOTION_PAGE_ID"
```

## Read this first

- [Exact research questions](RQ1_QUESTIONS.md)
- [Methods and population definitions](METHODS.md)
- [Detailed findings in Chinese](FINDINGS_CN.md)
- [Bridge from RQ1 to RQ2](RQ2_BRIDGE.md)
- [Limitations](LIMITATIONS.md)
- [Source and branch reconciliation](SOURCE_PROVENANCE.md)

## Main generated artifacts

- `data/tagging_compact.jsonl`: 5,662-row compact analysis layer retaining all
  labels and Chinese evidence rationales.
- `data/tagging_evidence.jsonl.zst`: generated, source-aligned audit layer. Each
  row retains the complete public PR instance (body, commits, discussion,
  reviews, inline comments, patch, CI evidence, and GitHub metadata), complete
  source provenance, and complete tagging result. Only the Vela/Ludus execution
  envelope and infrastructure metadata are removed. Credential-shaped values
  in `Authorization: Bearer` headers are replaced by
  `<redacted_api_token>` before publication; the manifest records the affected
  record and replacement count. The compressed archive is tracked with Git LFS;
  the 2.46 GB uncompressed JSONL remains local-only.
- `data/rq1_summary.json`: machine-readable headline results.
- `tables/`: machine-readable CSV for every reported statistic.
- `figures/`: high-resolution PNG figures for web and document publication.
- `manifests/`: exact paths, SHA-256 hashes, counts and task-snapshot lineage.
- `tagging/vllm_pr_tagging_taxonomy.yaml` plus its public prompt and
  hardware catalog: the exact label definitions and evidence instructions.

The 5.0 GiB raw Vela result remains on JFS and is not duplicated in Git. Its
path, size and SHA-256 are recorded in
`manifests/tagging_snapshot_audit.json`.

The compact file is the analysis input, not the complete evidence archive. To
build the compressed evidence archive after `run_all.sh`:

```bash
python3 analysis/RQ1/scripts/build_tagging_evidence.py \
  --input artifacts/rq1/vllm_pr_tagging/output/vllm_merged_prs_2026-02-01_2026-07-31.tagging_input.ci_complete.jsonl \
  --compact analysis/RQ1/data/tagging_compact.jsonl \
  --output analysis/RQ1/data/tagging_evidence.jsonl.zst \
  --manifest analysis/RQ1/manifests/tagging_evidence_manifest.json \
  --expected-input-sha256 936f6fe84f484005bd5b32b797ba8ae9b0f2ebc931c9cc9240327746b3fb5626 \
  --expected-compact-sha256 990b9bd1e01421b9980203f798dc9d0922a7da7ecdcd57e3ab08a2e838200f0c
```

## Reproduce

Run from the repository root:

```bash
export RQ1_TAGGING_SNAPSHOT=/path/to/frozen-task-snapshot.jsonl
bash analysis/RQ1/run_all.sh
```

The runner pins CPython 3.12 and all analysis packages. It first reconstructs
the compact layer from the frozen task-348689 snapshot, then regenerates every
table, figure, summary and manifest. It never submits a Vela or Ludus task.

## Headline population

- Target: all 5,662 PRs merged during 2026-02-01 through 2026-07-31.
- Frozen task snapshot: 5,649 unique outputs.
- Entire schema valid: 5,636 outputs.
- Four RQ1 dimensions independently valid: 5,649 outputs (99.77%).
- Missing RQ1 labels: 13 outputs (0.23%); the maximum possible absolute shift
  to any reported share is therefore 0.23 percentage points.

Multi-label percentages use the 5,649 core-labeled PRs as denominator and may
sum above 100%. Counts involving the full GitHub history use their own named
population and never reuse the semantic-label denominator.
