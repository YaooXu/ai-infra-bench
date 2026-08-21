# vLLM PR 35781: blocked waiting-request scheduler

This directory packages source `vllm__pr__35781` as one Harbor coding-agent
task for AI Infra Bench's memorable track.

## What the agent sees

- A clean source snapshot at base commit
  `d88f28da05b12bc7d63ebe3dcedf445ecb274343`.
- `instruction.md`, which describes the observable performance problem and
  correctness constraints without naming a file or prescribing a data
  structure.
- No network access at runtime and no post-fix git history.

## What the verifier checks

- FCFS and priority ordering across FSM, remote-KV, streaming and ready work.
- Remote-prefill allocation, completion, promotion and interleaving lifecycle.
- Abort cleanup, request counts and scheduler statistics.
- Async error propagation, invalid-block correctness and load-failure recovery.
- Unchanged CPU-only scheduler regressions.
- Event-driven remote-KV update work and ordinary waiting-queue churn.
- A paired, same-process CPU microbenchmark of the blocked-request hot path.

The score is written as `/logs/verifier/reward.json`. Correctness contributes
65% and performance contributes 35%. See `TEST_PLAN.md` for the complete test
matrix, subgroup weights and performance methodology.

## Validation sequence

Install Harbor, then run from the repository root:

```bash
uv tool run --from harbor harbor run -p tasks/vllm-pr-35781 -a nop
uv tool run --from harbor harbor run -p tasks/vllm-pr-35781 -a oracle
uv tool run --from harbor harbor run -p tasks/vllm-pr-35781 -a codex -m "<model>"
```

Expected task-author checks:

- `nop` gets less than 1.0 (normally only the regression component passes).
- `oracle` gets exactly 1.0.
- `solution/solve-wrong.sh` gets less than 1.0 in the verifier QA run.
- A real agent cannot use the network or local git history to recover the PR.

The Docker image uses the official vLLM v0.17.1 CPU release image while
exposing the exact pre-PR source tree in `/app`. This avoids a full source
build, keeps verification CPU-only, and does not require a GPU runtime. The
immutable source archive and a minimal tokenizer/config fixture are bundled in
the build context, so image construction and verification do not depend on
GitHub or Hugging Face availability.

## Validation performed

- Harbor 0.21.0 parses the task as schema 1.4 and reports it as a valid task.
- The verifier contains 36 CPU-only cases: 33 correctness cases and three
  performance cases.
- Applying `solution/reference.patch` exactly reproduces the PR head version of
  `scheduler.py`.
- The exact official vLLM CPU image was pulled and the Dockerfile was built by
  the isolated DXZ Docker daemon.
- Harbor `oracle` passes all 36 cases with correctness 1.0, performance 1.0 and
  reward 1.0.
- The latest Oracle paired microbenchmark measured 1.874 ms for the optimized
  path and 133.555 ms for the explicit legacy work model, with zero versus 768 expensive
  readiness callbacks.
- Harbor `nop` passes 15 cases and fails 21, producing correctness 0.48875,
  performance 0.0 and reward 0.3176875.
- The latest NOP paired microbenchmark measured 116.926 ms for the scheduler
  path and 129.742 ms for the legacy work model, with 768 callbacks in both paths.
- A seven-run Oracle timing check measured an optimized mean of 3.161 ms
  (3.14% coefficient of variation) versus a 209.220 ms legacy mean; all paired
  ratios were between 0.014 and 0.021.
- The plausible incomplete solution scores 0.5451875, and the public-workload
  hard-coded solution scores 0.4401875; neither passes the verifier.
- The verifier timeout path was exercised and confirmed to write a numeric
  zero reward rather than hanging or omitting the reward file.

Automated packaging and verifier gates are complete. Human-solve and
maintainer-approval gates remain pending and are intentionally recorded as
such in `CURATION.md`.
