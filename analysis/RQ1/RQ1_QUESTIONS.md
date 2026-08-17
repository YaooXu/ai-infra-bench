# RQ1 questions and analytical boundaries

## Main question

> vLLM 的真实工程 workload 由哪些工作组成？

“Workload” does not mean only lines of code or only accepted bug fixes. It means
the observable portfolio of engineering changes and integration work entering
and being incorporated into the project.

## Subquestions

### RQ1.1 — Workload envelope

How did the scale of Issues, incoming PRs, integrated PRs and active/new
contributors change from 2024 through July 2026? How much public work remains
open or closes without merge?

This establishes the population and prevents merged-only content analysis from
being misrepresented as all maintainer work.

### RQ1.2 — Technical composition

Among PRs that were actually integrated, what is the distribution of:

1. dominant engineering intent (`change_type`);
2. materially changed repository surfaces (`project_scope`);
3. affected vLLM production components (`architecture`); and
4. deliberately affected hardware backends (`affected_platforms`)?

### RQ1.3 — Joint structure

Are these dimensions separable in practice, or does work routinely cross
repository surfaces, architecture components and hardware backends? What joint
workload archetypes recur most often?

### RQ1.4 — Division of contribution

How does the integrated workload differ between external non-bot contributors,
snapshot write-capable collaborators, other snapshot collaborators and bots?

The collaborator roster is a May 2026 snapshot, so this is a descriptive cohort
comparison rather than a historical causal claim about membership.

### RQ1.5 — Consequence for benchmark coverage

Which workload strata and joint combinations must RQ2 retain for a benchmark to
represent vLLM engineering rather than a convenient subset of local bug fixes?

## Context, not the main RQ1 outcome

Review volume, reviewer concentration and backlog are reported as ecosystem
context. They show which lifecycle activities matter, but they do not redefine
RQ1 as a study of reviewer staffing or claim that a coding benchmark solves a
maintainer-capacity problem.

Verification evidence and reproduction hardware are not RQ1 outcome dimensions.
They are retained in the tagging output because RQ2 needs them to construct and
execute benchmark instances.

## Unit and population

The technical unit is one merged PR. The deep semantic population contains all
5,662 PRs merged from 2026-02-01 00:00:00 UTC to 2026-08-01 00:00:00 UTC. The
ecosystem context uses all canonical Issues and PRs visible at the 2026-07-31
cutoff. These two populations answer different subquestions and are never pooled.
