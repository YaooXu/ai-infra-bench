# RQ1 Empirical Study Protocol

Status: draft for team review  
Repository: `vllm-project/vllm`  
Observation cutoff: 2026-08-12 23:59:59 UTC

## 1. The decision RQ1 must support

The benchmark needs a defensible denominator. RQ1 should therefore answer:

> **How has vLLM's observable public maintenance workload evolved, what work does it contain, and how does incoming demand compare with maintainer capacity?**

This wording is intentionally narrower than “the real workload.” GitHub records public artifacts and interactions, not private debugging, meetings, chat, or actual hours worked.

RQ1 describes the whole public repository, while the benchmark can directly estimate agent coverage only for its **benchmark-eligible population**: independently verifiable, code-changing maintenance work with a reconstructible base state and a human reference solution. Results on PR-derived tasks must not be presented as coverage of all user issues or all maintainer labor.

## 2. Subquestions and estimands

### RQ1a: Intake and change volume

How did monthly issue and pull-request arrivals change?

- Treat user-filed issues and external PRs as incoming demand; report maintainer-authored PRs separately as maintainer-led change activity.
- Report monthly counts and rates; do not use cumulative GitHub totals as a trend.
- Separate human activity from bot activity.

### RQ1b: Responsiveness and backlog

Did the project keep pace with incoming work?

- Month-end open backlog and backlog older than 30, 90, and 180 days.
- Monthly opened-to-closed and opened-to-merged ratios.
- Time to first human response, first maintainer response, closure, and merge.
- Probability of a response within 2, 7, 14, and 30 days.

### RQ1c: Review burden and maintainer capacity

How much observable review and triage work was absorbed by maintainers?

- Active maintainers, active reviewer days, and maintainer actions per month.
- PRs receiving maintainer review, review rounds, change-request cycles, unique reviewers, and review span.
- Incoming artifacts per active maintainer and reviewed PRs per active reviewer.
- Concentration of review work using the top-5 share and Gini coefficient.

These are burden and capacity proxies, not engineering hours.

### RQ1d: Work composition

How did work type, subsystem, and accelerator coverage change?

- Issue intent: bug/correctness, installation/build, usage/configuration, feature/model/backend request, performance, documentation/API, design/RFC, CI/infrastructure, or other.
- PR work type: bug/correctness, feature/capability, performance/efficiency, refactor/maintainability, test/evaluation, CI/build/release, documentation/API/UX, or dependency/chore.
- Subsystem: model support, engine/scheduler, memory/KV cache, distributed serving, kernels/operators, frontend/API, platform/build, or tests/docs.
- Hardware: CPU, NVIDIA CUDA, AMD ROCm, Intel XPU, Ascend NPU, MLU, TPU, cross-backend, or hardware-independent.

Issue intent and PR work type are separate, single-label dimensions. Subsystem and hardware are shared, multi-label dimensions. Declared agent assistance is a process-origin flag, not a work type.

## 3. Study population and time

The census includes every public issue and PR created in `vllm-project/vllm` from repository launch on 2023-02-09 through the cutoff. Events after the cutoff are ignored, even when the current GitHub state includes them. Artifacts still awaiting a response, closure, review, or merge at the cutoff are right-censored rather than dropped.

Monthly time series are primary. The following windows are descriptive summaries, not assumed causal breakpoints:

1. launch through 2024;
2. calendar year 2025;
3. 2026 through the cutoff.

Partial months are excluded from month-over-month trend models or normalized by observed days and clearly marked.

### Preliminary size check

GitHub Search counts collected on 2026-08-13 provide the following extraction QA targets:

| Creation window | Issues | Pull requests |
| --- | ---: | ---: |
| 2023-02-09 through 2024 | 5,778 | 5,388 |
| 2025 | 6,928 | 12,807 |
| 2026 through 2026-08-12 | 4,519 | 15,782 |
| **Total** | **17,225** | **33,977** |

These are provisional checks, not analysis results. Final counts must come from the frozen cursor-paginated snapshot and pass reconciliation tests.

## 4. Data collection

### 4.1 Sources

Use GitHub GraphQL cursor pagination for the artifact census and REST endpoints where they expose more complete event detail. Do not use the Search API as the primary extractor.

Collect:

- issue and PR identity, author, timestamps, state transitions, labels, assignees, milestone events, and links;
- issue comments and PR conversation comments;
- submitted reviews and their state and commit SHA;
- inline review comments and threads;
- review requests and timeline events;
- PR commits, changed files, additions, deletions, draft state, and merge actor;
- historical `CODEOWNERS`, issue templates, and repository configuration from git.

Pin the API version, record the collection timestamp and query version, retain raw response checksums, and store pagination checkpoints. Fetch PR patches and commit metadata because head branches can later disappear.

### 4.2 Frozen state reconstruction

For each artifact, reconstruct its state as of the cutoff from timestamps and timeline events. Examples:

- a PR opened before the cutoff but merged afterward is open and unmerged in this study;
- an issue closed and reopened before the cutoff is open at month end;
- a comment or review submitted after the cutoff is not a response;
- labels are the labels active at the relevant time, not only the labels visible today.

### 4.3 Quality checks

- Reconcile artifact totals against independent GitHub queries by month and type.
- Assert unique repository/type/number keys and monotonic event timestamps.
- Manually compare at least 50 randomly sampled reconstructed timelines with the GitHub UI.
- Re-run 5% of extraction pages and compare content hashes.
- Publish extraction and exclusion counts at every stage.

## 5. Operational definitions

### 5.1 Human and bot activity

Exclude bots from human responsiveness and maintainer-capacity metrics. Detect bots using GitHub actor type, the `[bot]` suffix, known automation accounts, and a manually reviewed high-volume actor list. Report bot responses separately because automated first responses can make a project appear more responsive than it is.

### 5.2 Maintainers

The preferred definition is a project-provided roster with role start and end dates. An actor is an **active maintainer in month m** when the actor is on that roster during the month and performs at least one qualifying human maintenance action.

Qualifying actions include a human comment or review on another person's artifact, merge, label/assignment/milestone change, or closure decision. Report sensitivity analyses using thresholds of at least one action, at least three active days, and at least five actions in a month.

The current `CODEOWNERS` file is useful for subsystem expertise but is not a historical maintainer roster. If an authoritative roster cannot be obtained, use a documented set of observed gatekeepers derived from merge and review actions plus historical `CODEOWNERS`, manually validate it, and call the group **observed gatekeepers**, not maintainers. `author_association` may be used as a sensitivity check, not the sole definition.

### 5.3 Responses

- **First human response:** first textual issue comment, PR comment, submitted review, or inline review comment by a human other than the author.
- **First maintainer response:** first such response by a maintainer active at that time.
- **First substantive maintainer response:** first maintainer message that triages, requests actionable information, diagnoses, discusses design, reviews code, or makes a disposition decision.

Acknowledgements, reactions, assignments, labels, and bot messages are not substantive responses. Because substance requires interpretation, measure it on the manually annotated sample rather than pretending a character-count rule is valid for the whole census.

For issues, the response clock begins at creation. For a PR opened as ready, it also begins at creation; for a draft PR, it begins at the first `ready_for_review` event. Report elapsed wall-clock time in UTC, with business-day timing only as a sensitivity analysis.

**First maintainer code review** is the earliest submitted maintainer review or inline maintainer review comment after the PR becomes ready. General conversation comments count as responses but not code reviews.

### 5.4 Review rounds

A review round is a maximal group of maintainer review events against the same effective PR head, followed by a contributor update or final disposition. Use the review `commit_id` when available; otherwise assign the latest head SHA preceding the event. Count repeated comments on the same head as one round, and count a later author update followed by new review as another round.

A change-request cycle requires a `CHANGES_REQUESTED` review followed by a later contributor commit. Approvals, comments, and requested changes are reported separately.

### 5.5 Backlog

For month `m`:

```text
backlog_m = artifacts created on or before month end
            - artifacts closed or merged on or before month end
```

Issues and PRs are reported separately. Closed-without-merge PRs remain a distinct outcome. Reopened artifacts re-enter the backlog.

## 6. Workload taxonomy and content analysis

Repository labels alone are insufficient: labels change over time, are incomplete, and mix intent, subsystem, model, and process state. Use a transparent hybrid process.

1. Draft the codebook from vLLM labels, issue forms, file paths, and an open-card-sort pilot.
2. Pilot on 100 artifacts sampled across time and artifact type; revise ambiguous definitions.
3. Build separate paper-grade gold samples: 900 issues (300 per reporting window) and 1,200 PRs (400 per window). Oversample rare hardware and performance strata, retaining inclusion probabilities.
4. Double-code at least 20% of each sample and adjudicate disagreements. Report Krippendorff's alpha per dimension; target at least 0.80 before freezing the codebook.
5. Apply deterministic rules and an assisted classifier to the remaining corpus. Freeze prompts/model/rules, and report macro-F1, per-class precision/recall, and confusion matrices against held-out human labels.
6. Human-review every artifact entering the 200-PR candidate pool; automated labels are never final benchmark labels.

For PRs, classification uses only information available at the chosen source cutoff: title, body, contemporaneous labels, changed-file paths, and the pre-solution issue when linked. Reference patch content may be used for RQ1 taxonomy validation, but must not leak into released task instructions.

## 7. Analysis plan

### 7.1 Descriptive results

The main figures should be:

1. monthly issue and PR arrivals, separated by author role;
2. monthly closures/merges and month-end backlog;
3. first-human and first-maintainer response curves;
4. active maintainers/reviewers and demand-per-maintainer;
5. workload, subsystem, and hardware composition over time;
6. review-round and review-concentration distributions.

Report medians, interquartile ranges, proportions, and bootstrap 95% confidence intervals. Counts with heavy tails should not be summarized by means alone.

### 7.2 Time-to-event analysis

Use Kaplan-Meier estimates for first response, first review, and issue closure so unresolved artifacts remain in the analysis as right-censored observations. For PRs, merge and close-without-merge are competing outcomes; report cumulative incidence for each rather than treating one outcome as ordinary non-informative censoring. Report restricted mean time and event probabilities at fixed horizons when the median is not reached. Stratify by reporting window and workload type.

Regression, if used, is secondary and explanatory rather than causal. Check model assumptions and cluster uncertainty by creation month. Report effect sizes and uncertainty, not only p-values.

### 7.3 Trend claims

The study may support claims such as “incoming PR volume grew faster than observed review capacity” only when both sides are measured. It must not infer burnout, insufficient staffing, or causality from latency alone. Correlations between demand/capacity ratios and response times are exploratory.

## 8. From RQ1 to the 200-PR candidate pool

The candidate pool is a probability sample from a clearly declared eligible frame, not a hand-picked list of interesting PRs.

### Eligible frame

- created and merged by the cutoff;
- human-authored and code-changing;
- belongs to vLLM maintenance rather than an automated dependency update;
- base and solution commits can be reconstructed;
- no known security embargo or privacy restriction;
- not a pure revert, duplicate, or dependent slice that would double-count one root cause.

Eligibility for sampling does not guarantee that a PR can become a benchmark task.

### Sampling

1. Assign every eligible PR a primary work type, reporting window, subsystem, and hardware tier.
2. Collapse dependent PR series and shared root causes into clusters before sampling.
3. Allocate 200 slots proportionally by work type and time, with explicit minimums for bugs, performance, review-intensive work, and heterogeneous hardware.
4. Draw with a frozen random seed and record first-order inclusion probabilities.
5. Freeze an ordered reserve list within each stratum. Replace infeasible candidates only from the same stratum and record every exclusion reason.

The final 76 representative tasks are a second-stage sample after feasibility screening. Workload-reweighted benchmark estimates therefore need both source-selection and task-feasibility weights. Unweighted success remains useful, but it is not a population coverage estimate.

## 9. Calibrating review effort

GitHub cannot reveal time spent reading, reproducing, profiling, or discussing a PR elsewhere. Do not create an “effort score” by summing comments, churn, and elapsed time.

Instead, run a small calibration study on recent sampled PRs:

- target at least 50 PR assessments from at least 10 maintainers;
- collect ordinal active-review-time bins (`<15m`, `15–60m`, `1–4h`, `4–8h`, `>8h`);
- ask which work dominated: comprehension, reproduction, design, code reading, testing, performance validation, or coordination;
- compare the ordinal responses with observable proxies and report associations and uncertainty.

The existing memorable-task survey is valuable for task discovery but is a convenience sample of difficult work, so it cannot calibrate effort for the full workload by itself.

## 10. Validity, ethics, and reporting

- **Construct validity:** GitHub activity is an observable proxy, not total labor. Keep demand, throughput, latency, burden, and effort distinct.
- **Selection validity:** merged PR benchmark tasks omit unanswered issues, abandoned PRs, and work without a stable solution. State the target population beside every coverage number.
- **Temporal validity:** use event state at the cutoff and preserve right-censoring; do not read today's status back into history.
- **Classification validity:** publish the codebook, double-coding agreement, held-out classifier performance, and sampling probabilities.
- **Identity and ethics:** publish aggregate maintainer statistics and pseudonymous stable actor IDs where row-level linkage is necessary. Do not rank individuals or publish comment text in the analytical release.
- **Reproducibility:** publish query versions, schemas, checksums, exclusions, seeds, derived tables, and a data provenance manifest.

## 11. Deliverables and release gates

### Minimum defensible RQ1 snapshot

- frozen artifact and event manifest through the cutoff;
- reconciled monthly issue/PR arrivals, outcomes, and backlog;
- bot policy and maintainer-roster decision;
- preregistered definitions and analysis notebook outputs;
- taxonomy codebook with a manually audited pilot;
- frozen 200-PR manifest, reserve order, seed, and inclusion probabilities.

### Paper-grade extension

- 900-issue and 1,200-PR content-analysis samples with agreement reports;
- substantive-response annotation;
- review-effort calibration survey;
- full sensitivity and attrition analysis;
- anonymized derived dataset and reproducibility package.

The August 16 milestone should target the minimum defensible snapshot. The paper-grade extension should not be rushed into that deadline.

## 12. References

- GitHub, [REST API endpoints for timeline events](https://docs.github.com/en/rest/issues/timeline?apiVersion=2022-11-28).
- GitHub, [REST API endpoints for pull request reviews](https://docs.github.com/en/rest/pulls/reviews).
- GitHub, [Using pagination in the GraphQL API](https://docs.github.com/en/graphql/guides/using-pagination-in-the-graphql-api).
- CHAOSS, [Issue Response Time](https://chaoss.community/kb/metric-issue-response-time/).
- Wessel et al., [Understanding the Time to First Response in GitHub Pull Requests](https://arxiv.org/abs/2304.08426), 2023.
- Kalliamvakou et al., [The Promises and Perils of Mining GitHub](https://www.microsoft.com/en-us/research/publication/an-in-depth-study-of-the-promises-and-perils-of-mining-github/), MSR 2014.
- Chatterjee, Sharma, and Ralph, [Empirical Standards for Repository Mining](https://arxiv.org/abs/2203.15950), MSR 2022.
- Huang et al., [What Do Users Ask in Open-Source AI Repositories?](https://arxiv.org/abs/2303.09795), 2023.
- Khatoonabadi et al., [On Wasted Contributions: Understanding the Dynamics of Contributor-Abandoned Pull Requests](https://doi.org/10.1145/3530785), TOSEM 2023.
