# RQ1 analysis

This directory contains the reproducible aggregate analysis for the vLLM workload study. It consumes the maintainer-provided Fivetran SQLite snapshot and never writes issue bodies, comment text, names, email addresses, or row-level actor identities to the repository.

## Source snapshot

- Snapshot date: 2026-05-18
- Gist: <https://gist.github.com/simon-mo/2b0f4e9f872d479a08ae53edac51ecb1>
- SHA-256: `1992a9f7011ebe35ba6f62511d5ccc727b233e21d7279db3d3496f9f4892c44d`

Download the SQLite file using the URL in the gist README, then run:

```bash
python3 analysis/rq1/analyze.py \
  --snapshot /path/to/vllm-github-issues-pr-snapshot-2026-05-18.sqlite \
  --output analysis/rq1/outputs \
  --figures docs/assets/rq1 \
  --summary analysis/rq1/summary.json
```

Runtime dependencies are Python 3.10+, pandas, NumPy, SciPy, and Matplotlib. The 1.1 GB source snapshot stays outside the repository; only aggregate, identity-free outputs are versioned.

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

The script never treats heterogeneous event counts as hours, never estimates patch-size effects on merge from outcome-dependent commit coverage, and never relabels the current collaborator roster as a historical maintainer roster. See [the findings report](../../docs/RQ1_FINDINGS.md) for estimands and limitations.
