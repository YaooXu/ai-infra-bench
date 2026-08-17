# vLLM CPU candidate sample: 300 for a final 100

This directory contains a deterministic, three-times-overprovisioned candidate
sample for RQ2. It is a discovery and human-screening artifact, not a frozen
benchmark release.

## Population and eligibility

The full CPU workload contains 2,391 fully schema-valid merged PRs whose selected
reproduction platform is `cpu`. This is intentionally different from
`affected_platforms=cpu`: a CUDA, ROCm, XPU, or backend-agnostic maintenance
change can still be adequately verified on CPU.

Documentation-dominant work is excluded when either `change_type` is
`documentation` or the primary project scope is `documentation_examples`. This
removes 179 PRs from the workload reference and leaves 2,212 non-documentation
CPU PRs. Documentation may remain as a secondary scope of a substantive feature,
bug fix, build, or other engineering change.

The practical executable frame contains 1,626 CPU PRs before that workload
filter. It excludes bot authors, rows without a reproduction command,
low-confidence reproduction routes, rows without passed relevant verification,
patches outside 5--2,000 changed lines, patches with more than 30 changed files,
and routine revert/backport/merge PRs. Removing 116 documentation-dominant rows
leaves 1,510 eligible PRs from which the 300 candidates are selected.

## Threefold workload buckets

The intended released sample has 100 tasks. We first derive integer quotas for
those 100 tasks and then allocate exactly three times each quota to the 300-PR
candidate set.

The fixed joint bucket is:

```text
change_type x primary_project_scope x architecture_shape
```

`architecture_shape` is one of `support_only`, `single_component`, or
`multi_component`. There are 22 non-empty final buckets in this sample. Every
one contains exactly three candidates per intended final task.

The resulting high-level candidate counts are:

| Dimension | Candidate counts |
| --- | --- |
| Change type | 141 bug fixes, 51 features, 27 refactors, 21 CI, 21 build, 15 performance, 12 test, 9 maintenance, 3 other |
| Architecture shape | 63 support-only, 168 single-component, 69 multi-component |
| Scope integration | 141 single-scope, 159 multi-scope |
| Primary scope | 237 production code, 24 build, 15 CI, 18 tests, 6 benchmarks |

Within the fixed buckets, deterministic local search followed by an exact binary
integer program calibrates every concrete architecture component, all
multi-label project scopes, month, affected hardware, patch-size band, and
author association to the observed CPU workload. Every fixed and calibrated
integer target is satisfied exactly.

The 300-candidate set contains no documentation-dominant workload bucket. All
independently rounded change-type, primary-scope, architecture-shape, and
calibrated multi-label targets are satisfied exactly.

## Performance oversample

The representative core remains 300 PRs. A separate 15-PR CPU performance
top-up increases the available performance candidates from 15 to 30 without
changing any representative quota. The combined discovery pool therefore has
315 unique PRs:

```text
300 representative core + 15 performance reserve
```

The top-up contains six multi-component, six single-component, and three
support-only PRs, matching the architecture-shape mix of the original 15
performance candidates. Across the combined 30 performance candidates, 17 have
an executed benchmark with inspectable results, 12 have no recorded benchmark,
and one has an executed benchmark whose result is unavailable. All 15 added PRs
are CPU-runnable, disjoint from the core, and exclude documentation-primary and
`[Docs]`-titled work.

The performance reserve is deliberately oversampled. It must be reported as a
separate track or reweighted; it must not be treated as part of the
representative 300-PR distribution.

## Files

- `cpu_candidates_300.jsonl`: compact PR records plus selection metadata.
- `cpu_candidates_300.csv`: human-review index.
- `cpu_candidates_300.evidence.jsonl.zst`: complete PR evidence, including body,
  patch, commits, conversation, reviews, inline comments, CI, and tagging output.
- `distribution_comparison.csv`: population, eligible-frame, final-100 target,
  candidate-300 target, actual count, and share difference for every bucket.
- `workload_bucket_quotas.csv`: the 22 non-empty fixed buckets and their exact
  final-100 and candidate-300 quotas.
- `selection_manifest.json`: exact filters, quotas, input/output hashes, seed,
  optimizer diagnostics, and distribution-quality summary.
- `performance_topup_15.jsonl` / `.csv`: the additional performance candidates.
- `performance_topup_15.evidence.jsonl.zst`: complete evidence for the top-up.
- `performance_topup_manifest.json`: top-up constraints, provenance, and hashes.
- `cpu_candidates_315.jsonl` / `.csv`: combined discovery index with explicit
  `representative_core` and `performance_oversample` track labels.
- `../../scripts/sample_cpu_candidates.py`: deterministic generator.
- `../../scripts/sample_performance_topup.py`: deterministic performance top-up.

## Human screening rule

Human screening should retain approximately one PR out of each group of three
within every fixed workload bucket. It must also track the multi-label
architecture and scope totals in `distribution_comparison.csv`; simply choosing
the globally most attractive 100 PRs would destroy the calibrated distribution.

`change_type` is the engineering intent of a PR. It is not the future RQ2 task
contract. Diagnosis, implementation, patch review, review-driven revision, and
performance-engineering contracts should be assigned during human screening
from the complete evidence.

The 300 PRs have evidence-backed CPU routes, but they have not yet been rerun in
one standardized CPU sandbox. Actual execution, resource measurement, leakage
review, verifier construction, and expert semantic review remain release gates.

## Reproduce

From the repository root:

```bash
uv run --with-requirements analysis/RQ2/requirements.txt \
  python analysis/RQ2/scripts/sample_cpu_candidates.py

uv run --with-requirements analysis/RQ2/requirements.txt \
  python analysis/RQ2/scripts/sample_performance_topup.py
```

The generator uses seed `20260817`, derives every quota from the frozen RQ1
inputs, and fails if a fixed bucket cannot supply three times its final quota.
