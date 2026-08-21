# Curation report

## Source and scope

- Source ID: `vllm__pr__35781`
- Canonical source: <https://github.com/vllm-project/vllm/pull/35781>
- Track: memorable
- Workload: performance
- Subsystems: scheduler, distributed serving, KV connector
- Clean base: `d88f28da05b12bc7d63ebe3dcedf445ecb274343`
- Base snapshot date/source cutoff: 2026-03-10

The task asks an agent to identify and eliminate repeated scheduler work for
requests blocked on asynchronous remote KV loading. It includes the complete
production scope of the source PR as one task because queue separation,
promotion, accounting, cleanup, ordering, and failure recovery form a single
behavioral invariant.

## Reproduction

The base does not normally crash. The observable defect is scheduler CPU work
and queue churn that scales with the number of blocked requests on every step,
even when no completion event has arrived. Deterministic operation-count tests
reproduce this behavior without GPUs; a same-process paired microbenchmark
provides secondary timing evidence.

The original PR also reported an end-to-end gain in a two-GPU P/D deployment.
That hardware result is not used as a verifier threshold because the benchmark
task is CPU-only and network-isolated.

## Packaging and leakage controls

- The source archive is an exported base worktree and contains no `.git` data.
- The image creates a new one-commit repository with no remotes.
- Runtime networking is disabled.
- The instruction contains no PR number, commit, target file, symbol, or data
  structure from the reference implementation.
- The model fixture contains configuration/tokenizer metadata only, no model
  weights or remote checkpoint dependency.
- Tests and solutions are Harbor verifier/oracle assets and are not exposed in
  the agent worktree.

## Verifier QA

| Gate | Status | Evidence |
| --- | --- | --- |
| Base/no-op fails | pass | 15/36 cases pass; reward 0.3176875 |
| Reference passes | pass | 36/36 cases pass; reward 1.0 |
| Public-example hard-code fails | pass | 16/36 pass; reward 0.4401875; the 96-request special case fails paired timing and broader behavior |
| Plausible wrong solution fails | pass | 17/36 pass; reward 0.5451875; queue isolation and lifecycle/recovery fail |
| Performance signal exceeds noise | pass | Seven paired runs: optimized mean 3.161 ms (CV 3.14%), legacy mean 209.220 ms (CV 12.09%); every ratio is at most 0.021 |
| Human solve | pending | Requires an independent coding-agent or human run from the released instruction |
| Maintainer approval | pending | To be completed in task review |

Detailed correctness groups, performance points, and weights are documented in
`TEST_PLAN.md`.

## Risks

- The environment reuses compiled extensions from the nearby vLLM v0.17.1 CPU
  image. The task touches Python scheduler code, but image/source ABI drift is
  still a packaging risk.
- The bundled base archive is about 30 MB. It improves offline reliability at
  the cost of repository size.
- The paired timing assertion could be affected by extreme host contention;
  deterministic callback and queue-operation assertions carry most of the
  performance reward.
