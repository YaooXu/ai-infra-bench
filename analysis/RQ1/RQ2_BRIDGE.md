# From RQ1 workload to RQ2 benchmark tasks

RQ1 and RQ2 connect through two orthogonal axes.

## Axis A: what technical work is represented?

RQ1 supplies workload strata over:

- engineering intent;
- project surface;
- architecture component;
- affected hardware backend;
- single- versus cross-surface/component/backend integration.

RQ2 should sample and report coverage over their joint distribution, not only
over four independent marginals.

## Axis B: what reasoning contract is tested?

The same technical workload can be exposed to a frontier model through different
task contracts:

| Task contract | Information given | Capability tested | Typical output |
| --- | --- | --- | --- |
| Diagnosis/localization | Symptom, failure, log or regression report | Form a system model, locate root cause, identify affected boundary | Diagnosis, relevant files/components, evidence |
| Implementation | Requirement or diagnosed defect plus repository | Design and implement a correct change | Patch and explanation |
| Patch review | Candidate patch plus PR context | Detect correctness, compatibility, performance and integration risks | Actionable review findings |
| Review-driven revision | Patch plus review feedback | Distinguish valid feedback, revise without regressions | Updated patch and disposition |
| Performance engineering | Target, workload and measurements | Diagnose bottleneck and optimize under constraints | Patch plus benchmark evidence |

Therefore the observed review/integration workload does have a direct RQ2
consequence: review and diagnosis should be first-class benchmark contracts. It
does **not** imply that RQ2 should become only a review benchmark. A representative
suite crosses task contracts with workload strata.

For example, a CUDA/ROCm cross-backend attention change can yield several
different instances: locate a backend divergence from a failing symptom, review
a proposed fix, implement the fix, or revise it after a portability review. The
technical cell stays the same while the tested reasoning capability changes.

## Why this evaluates frontier AI-engineering reasoning

The benchmark target is not “can a model emit code?” It is whether a model can
build a sufficiently complete mental model of an AI-inference system to:

1. interpret underspecified symptoms and requirements;
2. navigate architecture and hardware boundaries;
3. identify interactions and regressions;
4. make and evaluate engineering trade-offs; and
5. produce a change that survives repository-grounded verification.

Diagnosis and review are especially useful because they expose understanding and
judgment without making code volume the proxy for reasoning quality.

## Construction rule

Each RQ2 instance should carry both:

1. a workload-stratum key derived from RQ1; and
2. a task-contract key such as diagnosis, implementation, review or revision.

Coverage reports should show the resulting matrix. A suite can then state, for
example, that it covers production bug fixes across attention, model loading and
serving on both backend-agnostic and backend-specific paths, and that those cells
are exercised through more than one reasoning contract.

Verification/reproduction labels determine whether an instance is runnable and
what environment it needs. They are construction metadata, not evidence that one
workload cell is more important than another.
