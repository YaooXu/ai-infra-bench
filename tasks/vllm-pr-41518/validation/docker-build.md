# Construction and validation record

## Status

`BLOCKED-BEFORE-BUILD / INELIGIBLE-OPEN-GOLD`

No Dockerfile was constructed, no Docker build was started, and no image tag
was created. This directory is an eligibility audit, not a runnable benchmark.

## Blocking evidence

1. The PR is open and its solution continued to change after the survey
   snapshot. The snapshot head is not a stable, accepted golden solution.
2. The survey's recorded base `228bcc436b0f09cf1824f00fd44f8f9c94060940`
   is not an ancestor of snapshot head
   `7d85640f2ddb613b83449982d5d6d5a7dcf14293`.
3. GitHub compare reports `ahead_by=14`, `behind_by=1`, and merge base
   `3d3ba460a20a7e313d278c2df710a9c0b4bcb00e`.
4. Directly treating the recorded pair as Base/Oracle adds unrelated native
   source reversions, so it would measure a synthetic patch that the PR author
   did not propose.
5. Review had not converged on the import architecture; maintainers raised
   long-term lazy-import maintenance concerns and discussed a narrower
   overlapping approach.

The source hashes and Git tree identities are recorded in
`lock/README.md`. They establish the audit facts but are not approved build
inputs.

## What a future verifier must cover

After the PR is merged or an accepted immutable head is selected, rebuild the
task from the true merge base and keep Base, Oracle, and Verifier separate.
The solved head and verifier assets must not enter the Agent image.

The functional hard gate should compare deterministic, non-interactive output
from Base and Oracle for:

- `vllm --help`;
- `vllm serve --help`;
- a misspelled top-level option;
- a misspelled `vllm serve` option;
- both the installed `vllm` entry point and
  `python -m vllm.entrypoints.cli.main`.

Expected return codes, stdout, and stderr must remain equivalent after
normalizing only unavoidable executable-path text. The performance gate should
warm each command, use multiple subprocess runs, and compare medians on the
same host. The upstream target is a warm median below one second; timing must
not be accepted without the output/return-code gate. An import-graph regression
probe should additionally fail if `torch` or `transformers` is imported on the
pure CLI hot path.

All verifier runs should use `--network none`. No model or dataset is needed.
GPU availability is irrelevant to the target behavior, so an A100-only result
must not be presented as stronger evidence than a CPU run.

## Restart conditions

Construction may resume only when all of the following are true:

- the project has merged the PR, or maintainers have explicitly selected an
  immutable head as the accepted solution;
- the true merge base and accepted head form a valid ancestor pair;
- the Base-to-Oracle diff contains only the accepted change;
- source archives, Git trees, official runtime image digest, and dependency
  provenance are freshly locked;
- the output-equivalence and import-graph verifier can be kept evaluator-side.

Until then, there is intentionally no `environment/Dockerfile`, image ID,
build duration, or runtime result.

## Construction-guide feedback

For open PRs, a recorded `base_sha` is not sufficient. Before any image work,
verify that it is an ancestor of the recorded head and record the merge base.
If the pair has diverged, do not manufacture a golden patch by diffing the two
trees. Open, rebased, design-sensitive PRs should remain survey candidates
until an accepted immutable solution exists.
