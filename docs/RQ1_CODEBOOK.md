# RQ1 Operational Codebook

Version: exploratory census, 2026-08-13

Observation cutoff: 2026-05-18

This codebook defines the variables used in the RQ1 vLLM workload census. It makes the aggregate report auditable; it is not a substitute for the planned human-coded gold sample.

## Units and populations

- **Issue:** an `issue` row without a matching `pull_request.issue_id`.
- **PR:** a `pull_request` row joined to its corresponding `issue` row.
- **Human:** an actor whose GitHub `user.type` is `User`.
- **Bot:** an actor whose GitHub `user.type` is `Bot`.
- **Snapshot collaborator:** a non-deleted `repo_collaborator` row with triage permission or higher at snapshot time.
- **External human:** a human PR author not in the snapshot collaborator roster.
- **Benchmark source frame:** merged, human-authored PRs with at least one recoverable commit/file record.

The collaborator definition is not projected backward as historical membership. Results therefore say “snapshot collaborator,” not “maintainer at event time.”

## Time and state

- Creation cohorts are launch–2024, calendar 2025, and 2026 through May 18.
- May 2026 is partial and excluded from complete-month comparisons.
- Month-end backlog reconstructs close/reopen history. Materialized state is authoritative at the final snapshot boundary.
- An issue response clock begins at issue creation.
- A ready PR clock begins at creation; a draft PR clock begins at its first `ready_for_review` event. Draft PRs never observed as ready are excluded from response-time risk sets.
- Open artifacts remain right-censored in response and outcome analyses.

## Response variables

- **Any-human response:** first issue/PR conversation comment, submitted PR review, or inline review comment by a human other than the artifact author.
- **Snapshot-collaborator response:** the same event restricted to the snapshot collaborator roster.
- Labels, reactions, assignments, bot comments, merge, and closure are not textual responses.
- A submitted review can be `COMMENTED`, `APPROVED`, or `CHANGES_REQUESTED`; existence does not imply a substantive or correct review.

Fixed-horizon response rates include every artifact old enough to have the full horizon observed. The report uses 2, 7, and 30 days and gives Wilson intervals. Kaplan–Meier curves retain later and unresolved responses.

## Issue intent

Issue intent is single-label and assigned in precedence order from the normalized leading title tag and current labels:

1. **Bug/correctness:** bug title tag or `bug` label.
2. **Feature/model/backend request:** feature/model tag or feature/new-model label.
3. **Usage/configuration:** usage/question/help tag or usage label.
4. **Design/RFC:** RFC/design/discussion tag or RFC label.
5. **Performance:** performance tag or label.
6. **CI/infrastructure:** CI tag or CI-failure label.
7. **Installation/build:** install/installation tag or label.
8. **Documentation/API:** documentation tag or label.
9. **Other/tracking:** no preceding rule matches.

Because templates and labels changed over time, apparent intent trends require validation against a time-stratified human sample.

## PR work type

PR work type is single-label. It uses title tags, current labels, and conservative title verbs in this precedence order:

1. **Documentation/API/UX**
2. **CI/build/release**
3. **Performance/efficiency**
4. **Test/evaluation** as the primary purpose
5. **Refactor/maintainability**
6. **Dependency/chore**
7. **Bug/correctness**
8. **Feature/capability**
9. **Other/unclear**

Precedence is necessary when titles contain both “add” and “fix,” but it can create classification error. `Other/unclear` remains visible rather than forcing a low-confidence assignment. Touching tests is a separate signal and never changes a bug fix into a test PR.

## Multi-label technical dimensions

Subsystem, hardware, and inference-topic signals are detected from the title, current labels, and recoverable changed paths. Multiple labels may apply to one PR, so shares do not sum to 100%.

### Subsystems

- model support;
- engine/scheduler;
- memory/KV cache;
- distributed serving;
- kernels/operators;
- frontend/API;
- platform/build/CI;
- tests/evaluation;
- documentation.

### Hardware

- NVIDIA/CUDA;
- AMD/ROCm;
- Intel/XPU;
- TPU;
- CPU;
- Ascend/NPU;
- MLU;
- cross-backend when two or more hardware signals occur;
- hardware-independent when none occurs.

“Hardware-independent” means no detected signal, not proof of portability. Path-based hardware ascertainment is weaker for open PRs because commit/file coverage is outcome-related.

### Inference engineering topics

- attention and kernels;
- distributed and parallelism;
- frontend, serving, and APIs;
- KV cache, connectors, and offload;
- LoRA and adapters;
- MoE and expert parallelism;
- model support;
- multimodal and audio;
- quantization and low precision;
- speculative decoding;
- structured output, tool use, and reasoning parsers;
- torch.compile and CUDA graphs;
- V1 engine and model runner;
- disaggregated serving.

Topic signals are intended for coverage design and hotspot discovery. They are not publication-grade semantic labels until per-topic precision and recall are measured.

## Review and queue variables

- **Submitted collaborator reviews:** review rows authored by snapshot collaborators other than the PR author.
- **Inline collaborator comments:** inline review-comment rows authored by snapshot collaborators other than the PR author.
- **Review round proxy:** distinct non-null review commit SHAs used by collaborator reviews.
- **Review span:** elapsed days between first and last submitted collaborator review.
- **Review-intensive:** at least three review-head rounds, at least ten collaborator review submissions, or a review span of at least 14 days.
- **Outstanding review request:** latest add/remove event for a PR–requested-user pair has `removed = 0`.
- **Latest approval/change request:** the last submitted state per PR–collaborator pair. It is not guaranteed to apply to the current head.
- **Assigned issue:** at least one current row in `issue_assignee`.

Review-intensive is a sampling proxy, not an effort score. No event count is converted into hours.

## Patch and verifier signals

- **Cumulative churn:** summed `changes` over unique PR–commit–file records. It can exceed final diff size because repeated edits are counted.
- **Size bins:** ≤20, 21–100, 101–500, 501–2,000, and >2,000 cumulative changed lines.
- **Large change:** more than 20 files, more than 1,000 cumulative changed lines, or more than 20 commits.
- **Test touched:** at least one recoverable path under a test directory or matching a test filename signal.
- **Benchmark/eval touched:** at least one recoverable path containing a benchmark or evaluation signal.
- **Docs only:** every recoverable changed path is documentation-like.

Test and benchmark paths indicate possible verifier assets; they do not establish sufficiency, determinism, offline executability, or absence of reference-solution leakage.

Patch variables are used only where commit data is available. They are not used to estimate merge probability because commit coverage is much higher for merged PRs than for open heads.

## Validation release gate

Before freezing benchmark weights:

1. sample issues and PRs across time, rare hardware, and rare work types with recorded inclusion probabilities;
2. double-code at least 20% and adjudicate disagreements;
3. report Krippendorff's alpha for human agreement;
4. report per-class precision, recall, macro-F1, and confusion matrices for deterministic/assisted labels;
5. manually validate every task entering the released benchmark;
6. retain both original sampling weights and feasibility-adjusted weights.

The executable rules live in [`analysis/rq1/analyze.py`](../analysis/rq1/analyze.py).
