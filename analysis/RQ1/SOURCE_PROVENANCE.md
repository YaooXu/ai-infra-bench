# Source provenance and remote-branch reconciliation

## Primary release data

| Artifact | Location | Integrity / state |
| --- | --- | --- |
| Release SQLite | `artifacts/rq1/vllm_pr_tagging/source/vllm_github_2026-07-31.sqlite` | 2ac86507…f8b5; 49,925 canonical artifacts |
| Six-month tagging input | `artifacts/rq1/vllm_pr_tagging/output/vllm_merged_prs_2026-02-01_2026-07-31.tagging_input.ci_complete.jsonl` | 936f6fe8…5626; 5,662 rows |
| Raw task snapshot | External frozen artifact, not published | 5950542d…9327; 5,312,373,209 bytes; 5,649 rows |
| Compact labels | `data/tagging_compact.jsonl` | hash and count in manifests; 5,662 source-aligned rows |
| Complete evidence archive | `data/tagging_evidence.jsonl.zst` | Git LFS artifact; hash, size, and counts in `manifests/tagging_evidence_manifest.json` |

The raw result was frozen at `2026-08-17 00:26:17 +08:00`. It is identified by
its size, SHA-256, row count, and logical artifact name without publishing the
internal transfer endpoint or storage path. All absent rows are listed rather
than hidden.

The complete evidence archive joins the six-month source input to the semantic
output by stable instance id. It preserves each source `instance` without
field-level pruning, including PR body, commit records, conversation and review
text, inline comments, full patch, CI check runs and commit statuses, and current
GitHub metadata. Before publication, credential-shaped values in
`Authorization: Bearer` headers are replaced by `<redacted_api_token>`; the
manifest records the exact affected record and replacement count. It also
preserves the complete structured tagging output and Chinese reasoning. It
removes the duplicated Vela/Ludus request envelope,
session identifier, source execution image/workspace, agent/tool/mount/sandbox
configuration, and pod/job/network paths. Global taxonomy, prompt, hardware
catalog, model/harness configuration, and hashes are stored once in the public
bundle and manifests rather than repeated in every PR row.

## Remote branch A: release-aligned RQ1 data

- ref: `origin/codex/vllm-rq1-analysis-data`
- inspected commit: `edfc277`
- local copies of the inspected branch files are not included in this package

This branch supplied release-aligned population definitions and prior Issue/PR
growth findings. Its path/title heuristic workload labels are not mixed with the
new semantic labels.

## Remote branch B: empirical study

- ref: `origin/codex/rq1-empirical-study`
- inspected commit: `69eb9e5`
- local copies of the inspected branch files are not included in this package

This branch supplied the broader ecosystem analysis: complete artifact outcomes,
contributor cohorts, review concentration, open/closed-unmerged burden and the
earlier path-based component study.

## Reconciliation rules

1. **Same database, explicit denominators.** The remote result “2,102.6 PRs/month”
   is all 14,718 PRs opened in 2026 Jan–Jul. “2,092.0” is the 14,644 non-bot PR
   mean. They are not conflicting estimates.
2. **New labels replace old heuristics for technical content.** Old `work_type`,
   `topic`, path-area and hardware regex fields remain sensitivity/context
   evidence. All headline intent/scope/architecture/hardware results come from
   taxonomy v11 and task 348689.
3. **Path-based and semantic component breadth are not pooled.** The old branch
   found 53.0% multi-component among its path-classifiable PRs. The new semantic
   result is 39.0% of all core-labeled merged PRs. Different components,
   populations and `support_only` rules make direct subtraction invalid.
4. **Review filters are reproduced.** The new script matches remote 2026 Jan–Jul
   review concentration exactly: 19,091 events, 77 active snapshot reviewers,
   top-five share 34.969% and Gini 0.66450.
5. **Merged-only limitation is preserved.** The remote estimate that merged PRs
   omit 18.6% of submitted reviews and 25.9% of inline comments remains ecosystem
   context; it is not overwritten by the merged semantic analysis.
6. **One bot-inference difference is explicit.** The empirical branch inferred
   missing-user bots from `-bot` and a short name allowlist and reports 75 bot /
   10,993 external PRs in 2026 Jan-Jul. The release-aligned conservative rule uses
   GitHub type `Bot` or `[bot]` and reports 74 / 10,994. This one-row difference is
   kept as a definition sensitivity, not silently normalized.

`data/legacy_reconciliation.json` is the machine-readable equality check for the
shared headline metrics.

## Taxonomy provenance

The public taxonomy bundle is committed under `analysis/RQ1/tagging/`:

- `vllm_pr_tagging_taxonomy.yaml`, SHA-256 `84d938a781638a312d82a7b889d5d6ee1886ce37f025e5562c1edea57dd48ecd`;
- `vllm_pr_tagging_prompt.md`, SHA-256 `2024aaf341615b77a38790d44cfef8091a920f5176a4bd7fcce46c05df052877`;
- `vllm_ci_hardware_catalog.yaml`, SHA-256 `46197091de87cd120d36c7e16260c66181cfc5e7e97c691568601946036b75f7`.

The internal CodePipe/Vela execution snippet and runtime-specific provenance
file are intentionally excluded from the public package. The schema definitions,
reasoning contract, examples, and hardware vocabulary required to audit the
labels are contained in the three public files above. The six-month population
and CI-completeness lineage are recorded separately in the RQ1 manifests.
