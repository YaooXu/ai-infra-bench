# Limitations and validity threats

## Construct validity

1. **Merged PRs are integrated workload, not all work.** Semantic labels cover
   merged PRs only. Open and closed-unmerged PRs consume substantial triage and
   review activity and can have a different technical distribution.
2. **PR is a delivery unit, not effort.** One small PR and one month-long PR each
   contribute one observation. Patch churn, file count and review events are
   reported separately and are not converted into hours or difficulty.
3. **Dominant intent compresses mixed motivation.** `change_type` is deliberately
   single-label. Secondary intents remain observable through multi-label scope,
   architecture, hardware, changed files and rationale, but are not counted as a
   second intent.
4. **Hardware support is broader than executed hardware.** Backend-specific
   build, test, CI and documentation are real maintenance scope and therefore
   count as affected hardware. This should not be read as successful runtime
   execution on that backend.
5. **User growth is contributor growth.** The stargazer table is truncated at
   exactly 40,000 and March 2025. We therefore do not infer general user adoption;
   we measure observed Issue/PR authors.

## Annotation validity

1. Labels were produced by one model/harness configuration with repository and
   PR evidence. The 100-item final smoke and full structural checks validate
   format and consistency, not semantic accuracy against independent human gold.
2. No double-coded random sample or inter-rater reliability statistic is yet
   available. Claims use broad categories and are accompanied by exact examples,
   rationales and source data so a future audit can estimate classification error.
3. Thirteen PRs have no frozen task output. The primary analysis is complete-case
   at 99.77% coverage and reports the exact worst-case 0.23 percentage-point
   bound; missing labels are never imputed.
4. Thirteen additional rows failed downstream verification/reproduction rules.
   Their RQ1 fields pass an independent validator, so they enter only RQ1 core
   analysis. They must not be treated as valid reproduction records for RQ2.
5. No `unknown` label was emitted among core-valid rows. That may reflect rich
   evidence and a closed merged-PR population, but it may also indicate model
   overconfidence. The `other` tail and all rationales should be sampled in a
   future human audit.

## Temporal and external validity

1. The semantic window is six months in 2026. Monthly distributions are shown,
   but the study cannot establish that the same mix held in earlier vLLM eras or
   will persist after July 2026.
2. vLLM is one fast-moving AI-inference project. Its hardware and architecture
   mix should not be generalized to all AI-infrastructure repositories without a
   replicated taxonomy mapping.
3. The collaborator roster is synchronized through May 18, 2026. Role labels are
   snapshot categories and can misclassify historical or late-period membership.
   Cohort comparisons are descriptive only.
4. Public GitHub evidence omits private discussion, local tests and decisions.
   Absence of public evidence is not evidence that no work occurred.

## Measurement and statistical limitations

1. The Wilson intervals quantify finite-population proportion uncertainty under
   a binomial framing; they do not include model-label error.
2. Multi-label co-occurrences are not independent observations. Their shares are
   descriptive and no naive chi-square p-values are reported.
3. Monthly means compare a 12-month period with a seven-month period. Seasonality
   can remain, even though monthly normalization removes the trivial exposure
   difference.
4. Review counts exclude self-review and non-`User` GitHub accounts and use a
   fixed collaborator roster to reproduce the prior study. They measure events,
   not review depth, quality or labor time.
5. Churn is heavy-tailed and generated files can inflate it. Medians and upper
   quartiles are used, but no causal relationship between integration breadth and
   difficulty is claimed.

## Recommended next validity step

Before paper submission, draw a stratified random audit sample across rare
backends, `other`, support-only, multi-component and top archetypes. Have at least
two vLLM-knowledgeable annotators independently code the four core dimensions,
adjudicate disagreements, and report per-field agreement plus error-adjusted
sensitivity intervals. This is the largest remaining threat to strong prevalence
claims.
