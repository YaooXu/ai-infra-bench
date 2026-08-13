# RQ1 Empirical Study Protocol

Status: implemented empirical baseline; human-validation stages pending

Repository: `vllm-project/vllm`

Observation cutoff: 2026-07-31 23:59:59 UTC

The current results are reported in [RQ1 Findings](RQ1_FINDINGS.md).
The implemented variables are specified in the [RQ1 operational codebook](RQ1_CODEBOOK.md).

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

### RQ1e: Engineering ownership and contributor lifecycle

Who implements, integrates, and reviews each kind of work, and how much contributor onboarding does the project absorb?

- Separate PR authorship, non-author review, issue response, and merge actions; never use git committer metadata as a generic developer measure.
- Compare external humans, current triage-only collaborators, and current write-capable collaborators as snapshot-role sensitivity groups.
- Report engineering and review ownership by work type, technical topic, hardware, subsystem, and concrete changed-path area.
- Report top-one/top-five shares, contributors needed for 50% and 80% of activity, HHI, and Gini; suppress individual identities and rankings.
- Measure first-time external authors, within-period contribution frequency, PR-sequence experience, and fixed-horizon return to a second PR.
- Treat experience/outcome differences as descriptive selection patterns, not contributor-quality or maintainer-bias estimates.

## 3. Study population and time

The census includes every public issue and PR created in `vllm-project/vllm` from repository launch on 2023-02-09 through the cutoff. Events after the cutoff are ignored, even when the current GitHub state includes them. Artifacts still awaiting a response, closure, review, or merge at the cutoff are right-censored rather than dropped.

Monthly time series are primary. The following windows are descriptive summaries, not assumed causal breakpoints:

1. launch through 2024;
2. calendar year 2025;
3. 2026 through the cutoff.

Partial months are excluded from month-over-month trend models or normalized by observed days and clearly marked.

### Frozen snapshot size

| Artifact | Rows |
| --- | ---: |
| Canonical issues | 16,990 |
| Canonical pull requests | 32,935 |
| Top-level comments | 205,998 |
| Submitted reviews | 131,473 |
| Mappable inline review comments | 122,470 |

## 4. Data collection

### 4.1 Sources

The canonical source is the released [merged SQLite database](https://github.com/ai-infra-bench/ai-infra-bench/releases/tag/vllm-github-data-2026-07-31), SHA-256 `2ac86507a95f9b8785e6ce0bbf2745e3fbba67c747e37b54020a7e57ce80f8b5`. It uses Simon Mo's [maintainer-provided Fivetran snapshot](https://gist.github.com/simon-mo/2b0f4e9f872d479a08ae53edac51ecb1) as its base and supplements it with cutoff-filtered GitHub GraphQL and REST data through July 31.

Collect:

- issue and PR identity, author, timestamps, state transitions, labels, assignees, milestone events, and links;
- issue comments and PR conversation comments;
- submitted reviews and their state and commit SHA;
- inline review comments and threads;
- review requests and timeline events;
- PR commits, changed files, additions, deletions, draft state, and merge actor;
- historical `CODEOWNERS`, issue templates, and repository configuration from git.

The analysis reads metadata from the SQLite file without publishing bodies, names, email addresses, or row-level actor identities. Pin the analysis version, retain the source checksum, and record every fallback or exclusion.

### 4.2 Frozen state reconstruction

For each artifact, reconstruct its state as of the cutoff from timestamps and timeline events. Examples:

- a PR opened before the cutoff but merged afterward is open and unmerged in this study;
- an issue closed and reopened before the cutoff is open at month end;
- a comment or review submitted after the cutoff is not a response;
- event-derived states use only events observed by the cutoff. The implemented exploratory classifier uses current snapshot labels as auxiliary signals and says so explicitly; a paper-grade temporal label analysis must reconstruct label intervals from history rather than reading current labels backward.

### 4.3 Quality checks

- Treat existence in the canonical pull-request table as the PR discriminator; there are no remaining issue/PR flag conflicts.
- Fall back to materialized `closed_at` for 443 closed artifacts without close-history rows.
- Use materialized state at the snapshot boundary for two current-state/history disagreements.
- Exclude 21 inline comments that cannot be mapped to a canonical PR from PR-level analysis while retaining them in the input audit.
- Define a merged PR by cutoff-consistent `merged_at`, taking the union of observed merge events and the materialized PR merge timestamp. This resolves 279 REST-state `CLOSED` representations with a merge event and retains 168 merged PRs whose merge event is absent; merge-actor analyses leave those 168 actors unknown.
- Require all 19 database release validations to pass before analysis.
- Reconcile artifact totals against independent GitHub queries by month and type.
- Manually compare at least 50 randomly sampled reconstructed timelines with the GitHub UI.
- Publish extraction and exclusion counts at every stage.

## 5. Operational definitions

### 5.1 Human and bot activity

Exclude bots from human responsiveness and maintainer-capacity metrics. The implemented census identifies 18 bot actors from the base GitHub actor type plus conservative login patterns for delta-only actors. Before paper release, audit a privacy-preserving high-volume actor sample for automation accounts incorrectly typed as users and report the sensitivity result without publishing actor rankings. Do not classify every username containing the substring “bot” as automation; that rule has obvious false positives. Report bot responses separately because automated first responses can make a project appear more responsive than it is.

### 5.2 Maintainers

The May 18 base snapshot contains 103 collaborator rows with triage permission or higher, including 70 with write permission or higher. The public API extension cannot refresh this roster and it does not contain membership start and end dates. The implemented analysis therefore calls these users **May-18 snapshot collaborators**, not July or historical maintainers. An actor is an **active snapshot collaborator in month m** when the actor performs at least one qualifying human maintenance action.

Qualifying actions include a human comment or review on another person's artifact, merge, label/assignment/milestone change, or closure decision. Report sensitivity analyses using thresholds of at least one action, at least three active days, and at least five actions in a month.

The current `CODEOWNERS` file is useful for subsystem expertise but is not a historical maintainer roster. A project-provided interval roster remains the preferred paper-grade extension. Until then, report both any-human and snapshot-collaborator results and do not silently relabel the latter as historical maintainer response.

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

The implemented PR classifier uses title, labels observed at collection, and cutoff-observed changed paths; it does not claim that current labels were present at PR creation. Because some current text and open-PR file representations were observed after the analytical cutoff, the report also publishes a stable-input exclusion sensitivity. A paper-grade temporal classifier must reconstruct label intervals and historical text explicitly. Reference patch content may be used for RQ1 taxonomy validation, but must not leak into released task instructions.

## 7. Analysis plan

### 7.1 Descriptive results

The main figures should be:

1. monthly issue and PR arrivals, separated by author role;
2. monthly closures/merges and month-end backlog;
3. first-human and first-maintainer response curves;
4. active maintainers/reviewers and demand-per-maintainer;
5. workload, subsystem, and hardware composition over time;
6. review-round and review-concentration distributions;
7. author-role, engineering-ownership, changed-path, reviewer-specialization, and contributor-lifecycle results.

Report medians, interquartile ranges, proportions, and denominators. Use Wilson intervals for fixed-horizon response proportions. Because the snapshot is a table census rather than a random artifact sample, do not attach artificial sampling-error intervals to raw counts; use design-aware bootstrap intervals for manually coded probability samples and cluster-by-month uncertainty for fitted trend models. Counts with heavy tails should not be summarized by means alone.

### 7.2 Time-to-event analysis

Use Kaplan-Meier estimates for first response, first review, and issue closure so unresolved artifacts remain in the analysis as right-censored observations. For PRs, merge and close-without-merge are competing outcomes; report cumulative incidence for each rather than treating one outcome as ordinary non-informative censoring. Report restricted mean time and event probabilities at fixed horizons when the median is not reached. Stratify by reporting window and workload type.

Regression, if used, is secondary and explanatory rather than causal. Check model assumptions and cluster uncertainty by creation month. Report effect sizes and uncertainty, not only p-values.

### 7.3 Trend claims

The study may support claims such as “incoming PR volume grew faster than observed review capacity” only when both sides are measured. It must not infer burnout, insufficient staffing, or causality from latency alone. Correlations between demand/capacity ratios and response times are exploratory.

## 8. Interface from RQ1 to the benchmark

RQ1 is a longitudinal analysis of the full public workload, not a procedure for selecting 200 PRs. Its benchmark role is to define populations, strata, and weights for three separately reported task contracts: implementation, diagnosis/reproduction, and review.

The PR-derived implementation frame requires sources that are:

- created and merged by the cutoff;
- human-authored and code-changing;
- belongs to vLLM maintenance rather than an automated dependency update;
- base and solution commits can be reconstructed;
- no known security embargo or privacy restriction;
- not a pure revert, duplicate, or dependent slice that would double-count one root cause.

The merged database yields 19,312 merged, human-authored PRs with commit data before task-feasibility screening, including 6,613 created in 2026 Jan–Jul. Sample from this or another explicitly declared frame using frozen probabilities, cluster dependent PR series, and record feasibility attrition. Diagnosis and review tasks need their own frames; they cannot inherit implementation-task weights.

Preserve author role and contributor experience as secondary implementation strata. Community-facing bugs and features, first-time-contributor patches, core runtime/refactor work, and specialist hardware/integration work expose different repository knowledge and verifier requirements. These strata inform coverage and expert-review allocation; current roster status must not be presented as historical membership.

Any intermediate candidate count is a project-management artifact. It is not an RQ1 result and should not dominate the empirical study.

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

- frozen, checksummed artifact and event snapshot;
- reconciled monthly demand, throughput, backlog, response, and competing PR outcomes;
- human/bot separation and explicit snapshot-collaborator terminology;
- current operational queues and non-author review/action decomposition;
- workload, inference-topic, subsystem, hardware, author-role, contributor-lifecycle, engineering/review ownership, concrete path-area, and task-shape results;
- an auditable operational codebook and explicit classification provenance;
- reproducible aggregate pipeline and findings report.

### Paper-grade extension

- 900-issue and 1,200-PR content-analysis samples with agreement reports;
- substantive-response annotation;
- review-effort calibration survey;
- full sensitivity and attrition analysis;
- anonymized derived dataset and reproducibility package.

The implemented baseline satisfies the quantitative snapshot requirement. Human taxonomy validation, substantive-response annotation, historical-role recovery, and effort calibration remain paper-grade extensions.

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
