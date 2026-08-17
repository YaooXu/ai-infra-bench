# Methods

## 1. Study design

RQ1 uses a two-layer observational design.

1. **Repository workload envelope.** All canonical Issues and PRs in the July 31
   release are used to measure arrivals, integration, backlog, contributor growth
   and review activity.
2. **Integrated technical workload.** All PRs merged during February–July 2026
   are semantically labeled to characterize the technical work that entered
   vLLM.

This distinction is substantive. A merged-only sample can characterize integrated
changes but cannot measure all triage and review work. Conversely, repository
event counts do not reveal the architecture or hardware meaning of a change.

## 2. Data sources and cutoff

The primary source is the release-aligned SQLite database:

- release: `vllm-github-data-2026-07-31`;
- cutoff: `2026-07-31 23:59:59 UTC`;
- uncompressed database SHA-256:
  `2ac86507a95f9b8785e6ce0bbf2745e3fbba67c747e37b54020a7e57ce80f8b5`;
- canonical artifacts: 49,925, comprising 32,935 PRs and 16,990 Issues.

The database integrates the original Fivetran snapshot with delta retrieval and
cutoff reconstruction. The earlier remote empirical study's 19 release
validations passed. The current analysis hashes the exact database again and
checks its populations before producing outputs.

GitHub's stargazer table stops at exactly 40,000 records and at March 2025, so it
is not used to claim user growth. “User growth” here means observed non-bot PR and
Issue authors, not stars or repository visitors.

## 3. Populations and denominators

### 3.1 Ecosystem population

The ecosystem layer includes every canonical artifact at the cutoff. Monthly
comparisons use complete calendar year 2024, complete calendar year 2025 and the
seven observed months January–July 2026. Monthly means, rather than raw period
totals, make unequal windows comparable.

Bot status is assigned when the GitHub user type is `Bot` or the login ends in
`[bot]`. “Non-bot” is used deliberately: an untyped account is not asserted to be
human merely because it is not a known bot. The older empirical branch also
inferred bots from `-bot` and a short name allowlist; that broader rule moves one
2026 PR from external to bot (75 instead of 74) but does not change any all-PR
count or the six-month merged semantic population.

### 3.2 Semantic deep-label population

Selection is exhaustive rather than sampled:

```text
repository = vllm-project/vllm
state_at_cutoff = MERGED
merged_at >= 2026-02-01T00:00:00Z
merged_at <  2026-08-01T00:00:00Z
```

This yields 5,662 unique PRs: 862 in February, 1,074 in March, 802 in April,
855 in May, 1,041 in June and 1,028 in July. The source JSONL SHA-256 is
`936f6fe84f484005bd5b32b797ba8ae9b0f2ebc931c9cc9240327746b3fb5626`.

All marginal and joint label shares use 5,649 PRs with valid RQ1 core labels as
denominator. Tables repeat the denominator. Since `project_scope`, `architecture`
and `affected_platforms` are multi-label, their shares may sum above 100%.

## 4. Evidence supplied for each PR

Each task received substantially more than a title and diff:

- title, body, labels, author metadata and timestamps;
- base/head/merge refs and the repository checked out at the PR's base commit;
- complete reconstructed patch and changed-file list;
- commits;
- cutoff-visible conversation comments, submitted reviews and inline comments;
- complete CI check-runs and commit statuses for all associated head and merge
  commits, with pagination exhausted;
- a compact CI summary plus on-demand raw CI evidence.

All 5,662 patches are non-empty and reconstructable. The CI sidecar contains
24,667 unique commits, 147,539 check-runs and 2,435,292 commit statuses; every
selected PR has complete associated-commit endpoint coverage. Absence of a CI
record is still not treated as proof that no private or local testing occurred.

The agent could freely inspect a repository checkout at the PR's base commit.
Runtime-specific image and workspace identifiers are intentionally excluded
from the public package.

## 5. Taxonomy and construct definitions

The final taxonomy is
`vllm-pr-reproduction-2026-08-16-v11` (SHA-256
`84d938a781638a312d82a7b889d5d6ee1886ce37f025e5562c1edea57dd48ecd`). Every emitted
label includes a concise Chinese evidence rationale.

The four RQ1 dimensions intentionally answer different questions:

| Dimension | Mode | Construct |
| --- | --- | --- |
| `change_type` | single | The PR's dominant engineering intent: why the change exists |
| `project_scope` | multi | Roles of materially changed repository artifacts: what kind of project surface changed |
| `architecture` | multi | Production system components whose implementation or contract changed |
| `affected_platforms` | multi | Hardware backends deliberately affected by changed production, build, test, benchmark, CI or documentation artifacts |

For example, a ROCm CI fix can be `change_type=ci`, `project_scope=[ci]`,
`architecture=[support_only]`, and `affected_platforms=[amd_rocm]`. The labels are
not redundant.

### 5.1 Why these project scopes

The scope vocabulary is a partition of artifact roles observable in vLLM:

- shipped implementation and runtime data (`production_code`);
- executable correctness assets (`tests`);
- measurement assets (`benchmarks`);
- hosted automation (`ci`);
- compilation, dependencies, packaging and containers (`build`);
- human guidance (`documentation_examples`);
- developer-only tooling and governance (`developer_tooling`);
- explicit `other` and `unknown` escape values.

These categories were checked against the repository tree and all CODEOWNERS
patterns. They are about the role of changed artifacts, not directory prefixes
alone; a file is classified from patch semantics and repository context.

### 5.2 Why these architecture components

The 20 production components were derived from the vLLM source tree, runtime
control/data flow, CODEOWNERS, Buildkite test areas and nearby implementation
boundaries. The CODEOWNERS audit accounts for all 160 rules:

- 114 directly map to architecture components;
- 7 map indirectly through matched files;
- 5 require explicit path overrides;
- 34 are correctly represented only by project scope (build, CI, tests,
  documentation or tooling).

`support_only` is exclusive and means no production component changed. Thus build
and CI work remains part of the vLLM workload without being falsely assigned to
a runtime subsystem.

### 5.3 Hardware versus reproduction

`affected_platforms` records the backend whose supported behavior or support
surface changed. It does not record the runner on which validation happened.
Incidental CI hardware, generic `torch.cuda` spelling, and an unchanged comparison
backend do not create a hardware label. Backend-specific tests, CI, build and
documentation do count because maintaining support is real project work.

The separately retained reproduction platform answers “what environment should
RQ2 provision?” CPU is chosen when CPU adequately exercises the important change.
Those fields are not analyzed as RQ1 outcomes.

## 6. Tagging runtime and quality controls

- task: Vela task `348689`;
- agent harness: Codex `0.144.1`;
- `vela_model_id`: `210145`;
- original requested concurrency: 500 (platform task metadata was later observed
  with a 1,000 concurrency cap; concurrency does not change labeling semantics);
- final preflight smoke: task `348596`, 100/100 execution success, reward 1 and
  schema validity.

The frozen non-terminal task snapshot was transferred at
`2026-08-17T00:26:17+08:00`. It contains 5,649 unique rows: 5,636 valid against the
whole schema and 13 with `reward=0`. All 13 failed only a downstream consistency
or whitespace rule; the four RQ1 dimensions of every one independently pass
their closed vocabulary, structure, exclusivity and reasoning checks. Thirteen
source IDs were not yet present.

Accordingly:

- full-schema coverage is 5,636/5,662 = 99.54%;
- RQ1 core-label coverage is 5,649/5,662 = 99.77%;
- the 13 absent results remain missing, never silently imputed;
- if all 13 missing PRs belonged to one label, any reported share could move by
  at most 13/5,662 = 0.23 percentage points.

This is a structural and consistency audit, not an independent human-gold
accuracy estimate. See `LIMITATIONS.md`.

## 7. Ecosystem actor and review definitions

Contributor roles use the repository collaborator table synchronized through
May 18, 2026:

- `snapshot_write_plus`: active collaborator with push permission;
- `snapshot_nonwrite`: active collaborator with triage but no push permission;
- `external_nonbot`: non-bot author absent from that active snapshot roster;
- `bot`: known bot account.

The categories describe membership in one snapshot and are not historical role
labels.

Review-capacity counts reproduce the earlier empirical study's filter exactly:
the reviewer must have GitHub type `User`, belong to the active snapshot roster,
review an identifiable PR, and not review their own PR. The 2026 January–July
result (19,091 submitted reviews, 77 active reviewers, top-five share 35.0%, Gini
0.6645) exactly matches the remote empirical branch.

## 8. Statistical analysis

- Counts and shares are descriptive, with named denominators.
- Single- and multi-label proportions include 95% Wilson score intervals.
- Monthly means compare unequal observation windows.
- Joint tables count all observed label combinations; co-occurrence shares use PR
  count as denominator.
- Workload archetypes are deterministic combinations of dominant intent, primary
  project scope, architecture shape and affected-hardware scope. No clustering
  hyperparameters or post-hoc model are involved.
- Patch complexity reports medians and 75th percentiles because churn and file
  counts are heavy-tailed. Review-event counts are not interpreted as labor time.
- All missing-label sensitivity bounds are reported explicitly.

No causal inference is claimed. Differences across author cohorts or integration
shapes are descriptive associations.

## 9. Reproducibility

`run_all.sh` pins CPython and package versions, validates the 2.3 GiB input hash,
hashes the 5.0 GiB frozen result while streaming, writes a 37 MiB compact layer,
then regenerates all tables and figures. The scripts fail on population, ID,
hash, date-window or core-taxonomy inconsistencies.
