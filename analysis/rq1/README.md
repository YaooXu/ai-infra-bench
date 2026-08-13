# RQ1 analysis

This directory contains the reproducible aggregate analysis for the vLLM workload study. The current results use the merged database through **2026-07-31 23:59:59 UTC**. The analysis never writes issue bodies, comment text, names, email addresses, or row-level actor identities to the repository.

## Reproduce the current analysis

Download and decompress `vllm_github_2026-07-31.sqlite.zst` from the [`vllm-github-data-2026-07-31` release](https://github.com/ai-infra-bench/ai-infra-bench/releases/tag/vllm-github-data-2026-07-31). The uncompressed database must have SHA-256 `2ac86507a95f9b8785e6ce0bbf2745e3fbba67c747e37b54020a7e57ce80f8b5`.

Run:

```bash
python3 analysis/rq1/analyze.py \
  --snapshot data/derived/vllm_github_2026-07-31.sqlite \
  --output analysis/rq1/outputs \
  --figures docs/assets/rq1 \
  --summary analysis/rq1/summary.json
```

Runtime dependencies are Python 3.10+, pandas, NumPy, SciPy, and Matplotlib. Row-level databases stay outside git; only aggregate, identity-free outputs are versioned. `load_merged.py` translates the merged database into the analyzer's canonical inputs and prevents duplicated base/delta events.

CSV outputs are ignored by git because they are reproducible. The aggregate summary and report figures are versioned. The title/label/path taxonomies are deterministic exploratory classifications; they are not substitutes for the preregistered human-coded gold sample.

The pipeline produces aggregate tables for:

- monthly intake, merge/closure throughput, backlog, and fixed-horizon response;
- current issue and PR queues, including assignment, review-request, review-state, and process-label signals;
- issue intent, PR work type, inference topic, subsystem, and hardware composition;
- observable snapshot-collaborator action volume and concentration;
- review burden by PR outcome, work type, subsystem, hardware, and topic;
- author-role work composition, engineering/review/merge concentration, and changed-path ownership;
- contributor intake, first-time and repeat-author lifecycle, return, and review-capacity sensitivity;
- competing PR outcomes, test/verifier signals, and merged-task complexity strata.

The separate [vLLM component taxonomy](../../docs/VLLM_COMPONENT_TAXONOMY.md) triangulates the cutoff source tree, CODEOWNERS, Buildkite test and hardware domains, and 2026 changed paths. Reproduce it with:

```bash
python3 analysis/rq1/analyze_vllm_components.py
```

This component analysis additionally uses PyYAML and pathspec to read the maintainer-authored CI and CODEOWNERS definitions.

The script never treats heterogeneous event counts as hours, never estimates patch-size effects on merge from outcome-dependent commit coverage, and never relabels the May 18 collaborator roster as a historical maintainer roster. See the [academic PDF report](../../output/pdf/vllm_rq1_workload_report_2026-07-31_zh.pdf), [English findings](../../docs/RQ1_FINDINGS.md), [Chinese findings](../../docs/RQ1_FINDINGS_ZH.md), or [May-to-July conclusion audit](../../docs/RQ1_CHANGE_AUDIT_2026-07-31.md) for estimands and limitations.

Regenerate the academic PDF with:

```bash
python3 analysis/rq1/render_academic_report.py
```

## Base snapshot provenance

The merged database starts from Simon Mo's [*vLLM GitHub Gym: vLLM GitHub Snapshot (Fivetran)*](https://gist.github.com/simon-mo/2b0f4e9f872d479a08ae53edac51ecb1), observed on 2026-05-18 with SHA-256 `1992a9f7011ebe35ba6f62511d5ccc727b233e21d7279db3d3496f9f4892c44d`, and extends it through July 31.

## July 31 extension

`collect_github_delta.py` extends the maintainer snapshot through 2026-07-31. It saves every API response as a gzip-compressed JSON envelope before normalization, then builds a queryable SQLite index and a checksum manifest:

```bash
python3 analysis/rq1/collect_github_delta.py \
  --output data/raw/vllm_github_2026-07-31
```

The local dataset contains updated issues and PRs, conversation and inline-review comments, maintenance timeline events, PR reviews/commits/files, default-branch delivery history, repository labels, and the repository tree at the cutoff. Collection is resumable and reads authentication from `gh auth token`; credentials are never written to disk. Raw files and derived row-level databases stay under ignored `data/raw/` because they contain public text, actor identifiers, and commit emails. The generated `manifest.json` records file families, checksums, row counts, cutoff-consistent counts, and known historical limitations.

The canonical cutoff is `2026-07-31T23:59:59Z`. Raw responses can contain later current-state representations when an older artifact changed after the cutoff; every analytical event table is separately counted and filtered at the cutoff. The public API cannot enumerate the complete current collaborator roster without write-level repository access, so the extension retains the May 18 snapshot roster only as a documented sensitivity definition, not as a historical maintainer identity label.

The identity-free [collection manifest](COLLECTION_2026-07-31.json) publishes coverage, validation results, fallback counts, and limitations without publishing row-level raw data.

## Merged database

The public database combines the May 18 snapshot and July 31 delta in one SQLite file. Download the compressed asset from the [`vllm-github-data-2026-07-31` release](https://github.com/ai-infra-bench/ai-infra-bench/releases/tag/vllm-github-data-2026-07-31), verify its SHA-256 against the release manifest, and decompress it with `unzstd`.

To reproduce it locally:

```bash
python3 analysis/rq1/build_merged_database.py \
  --base data/raw/vllm_github_2026-07-31/base/vllm_2026-05-18.sqlite \
  --delta data/raw/vllm_github_2026-07-31/derived/github_delta.sqlite \
  --output data/derived/vllm_github_2026-07-31.sqlite
```

The database has three query layers:

- the original Fivetran tables, preserved unchanged;
- `delta_*`, the complete normalized API delta including raw JSON;
- `canonical_*`, deduplicated tables materialized at the inclusive `2026-07-31T23:59:59Z` cutoff.

`dataset_source`, `dataset_metadata`, `dataset_table_inventory`, and `dataset_validation` make provenance, cutoff semantics, row counts, anomalies, and validation results queryable inside the database. Text edited after the cutoff remains the representation observed during collection and is flagged by `representation_may_postdate_cutoff`. PR file lists that cannot be proven stable at the cutoff are flagged by `files_cutoff_stable`/`cutoff_stable`.

This dataset uses Simon Mo's [*vLLM GitHub Gym: vLLM GitHub Snapshot (Fivetran)*](https://gist.github.com/simon-mo/2b0f4e9f872d479a08ae53edac51ecb1) as its base and extends the data through 2026-07-31.

The database contains public GitHub text, usernames, actor identifiers, and commit metadata including names and emails. It is not de-identified survey data. No new license is asserted over third-party GitHub content; users should follow the source and GitHub terms when redistributing or using it.
