# Constructing vLLM and GPU benchmark tasks

Use this playbook for ai-infra-bench tasks derived from vLLM or other GPU
infrastructure changes. It supplements the generic task-creation workflow. The
post-construction independent review remains governed by the
`ai-infra-bench-task-review` skill.

## 1. Freeze provenance before task design

Record before looking at evaluated-agent outcomes:

- repository, PR and linked issues;
- complete PR discussion and review timeline;
- exact base SHA and tree SHA;
- every PR commit and the final candidate/Oracle source;
- later fixes, regressions, reverts, or replacement PRs that change whether the
  proposed Oracle is still defensible;
- the earliest reproducible public symptom in the A/B/C/D discovery timeline;
- whether the task represents the complete PR or an explicitly named slice.

A merged PR is not automatically a valid Oracle. If later public evidence shows
that the patch regressed or was reverted, stop and establish a stable behavior
contract and reference implementation before constructing the task.

## 2. Freeze the task contract and difficulty

Write a contract matrix with at least:

| Field | Required decision |
| --- | --- |
| Surface behavior | What an operator or developer can observe |
| Starting point | Earliest reasonable A/B/C/D stage and why |
| Desired behavior | Public result that must change |
| Preserved behavior | Existing modes that must not regress |
| Negative behavior | Invalid or persistent failures that must remain visible |
| Scope | Full PR or a named slice |
| Real E2E | Production entrypoint and deepest exercised boundary |
| Substitutions | Missing model/service boundaries replaced deterministically |
| Resources | CPU/GPU type, count, memory, disk, and network |
| Time limits | Separate build, agent, verifier, and collective timeouts |

Difficulty must come from the authentic diagnosis and implementation scope. Do
not make a task difficult by omitting a necessary public contract, dependency,
model, command, or performance protocol. Do not make it easy by stating the
root cause, Golden helper, field, file list, or algorithm.

If a narrow slice is useful, keep it and label it honestly. Creating a harder
full-feature task is a new task or a material new version, not a silent rewrite
of the slice after observing a model pass.

## 3. Instruction rules

The instruction may disclose:

- the authentic observable symptom or developer trace;
- a runnable public reproduction;
- required public API names;
- hardware, shapes, dtypes, warmup, statistics, and frozen performance gates;
- categories of state or behavior that must be preserved.

It must not disclose:

- Golden-only private helpers, constructor keywords, attributes, or filenames;
- the root cause learned only at the final discovery stage;
- a checklist copied from the Golden diff;
- the internal algorithm selected by Golden;
- hidden test inputs or expected source structure.

Hidden tests may vary values, ordering, shape, lifecycle repetitions, topology,
and error cases. They may not introduce a behavior category that the
instruction did not disclose.

## 4. Environment profiles

Choose one declared profile:

### `cpu`

For tasks whose scored production boundary is CPU-only. A GPU-specific problem
must not be converted to this profile merely because mocks make it convenient.

### `cuda-triton`

For Python/Triton work that executes a real GPU kernel but does not modify a
compiled vLLM extension. Pin the CUDA/PyTorch/vLLM runtime combination and prove
that the tested tensors and kernel run on the requested GPU architecture.

### `cuda-native-extension`

For `_C`, `_moe_C`, or other compiled extensions. Declare:

- the exact-source versus donor-artifact policy;
- the focused build target and command;
- CUDA architectures and compiler limits;
- source, build-input, output-library, and cold-import digests;
- the rule excluding generated `.so`, build directories, and caches from the
  agent source patch.

The verifier must rebuild the affected target or prove that the imported
library was built from the candidate source. A source patch paired with a stale
library is not valid evidence.

### `cuda-multigpu-nccl`

For PP/TP/collective behavior. In addition to the GPU rules, declare:

- exact GPU count and topology;
- process count and rank mapping;
- rendezvous method;
- NCCL and subprocess timeouts;
- deterministic cleanup after success, failure, and timeout;
- the precise data-path interval in which object collectives or GPU-to-CPU
  synchronization are forbidden.

Do not flag initialization, logging, or final test assertions when the contract
only forbids synchronization on the hot data path.

## 5. Version and offline-fixture compatibility

Before building, construct a compatibility record for:

- vLLM base SHA and source tree;
- official/donor image digest and donor vLLM revision;
- Python, CUDA, driver requirement, PyTorch, Triton, and compiler versions;
- native extension origin;
- Oracle applicability to the base;
- tokenizer/model configuration or deterministic fixture revisions.

The verifier must be runnable under its declared network policy. If it uses a
model identifier while offline, bake the smallest required immutable metadata
or a valid local fixture into the image. Do not let Base, Oracle, or agent fail
first because a model, pytest, compiler, or package is missing.

## 6. Native and image provenance

Build from an empty context or an allow-listed context that cannot contain
tests, solution, validation patches, trajectories, credentials, or future
source. Pin all images and downloads by digest.

Record at minimum:

- task/input SHA and candidate patch SHA;
- base and source tree SHA;
- Dockerfile and lock SHA;
- build command, exit code, duration, and focused target;
- CUDA/C++ source closure SHA;
- output `.so` SHA and ELF/CUDA architecture evidence;
- actual Python import path and imported-library SHA in a fresh process;
- image ID/digest and relevant installed versions.

Build failure, correctness failure, E2E failure, and performance failure are
different outcomes and must be reported separately.

## 7. Construction acceptance checks

Map every reward-affecting test to an instruction behavior. Use observable
behavior or stable subsystem state, not Golden implementation shape.

Complete these checks while constructing the task:

1. **Base negative:** reward 0 because the target behavior is wrong. Import,
   dependency, missing-model, missing-private-symbol, or build failures do not
   satisfy this control.
2. **Oracle positive:** reward 1 at the same E2E boundary, with zero unexpected
   skips.
3. **Environment smoke:** the declared compiler, package, model fixture, GPU,
   and collective requirements are available before the target behavior runs.
4. **Production-boundary smoke:** Base and Oracle exercise the production entry
   point named in the task contract rather than only a mock or private helper.
5. **Artifact provenance:** native builds and imports match the candidate source
   and recorded hashes in a fresh process.

For performance tasks, publish hardware, shape, dtype, warmup, repetition,
statistic, and threshold. Confirm that the Oracle has adequate margin over the
threshold rather than relying on a borderline or noisy pass.

These are construction checks, not a complete independent review. Do not add
post-construction review or evaluated-agent procedures here.

## 8. Handoff to independent review

After the construction checks pass, freeze and hand off:

- instruction and declared task scope;
- base source and image identities;
- environment and dependency locks;
- Oracle patch and applicability record;
- verifier and public reproduction;
- Base/Oracle logs and artifact provenance;
- hardware requirements and time limits.

Hand these artifacts to the `ai-infra-bench-task-review` skill. Its procedures
and acceptance criteria are intentionally outside the scope of this creation
reference. If review later returns a construction defect, address it as a new
task version rather than silently editing the frozen candidate.
