# vLLM 8-task GPU-group rebuild audit

This document records the second construction pass for the eight assigned vLLM
PRs. “GPU group” identifies the assignment batch; it does not imply that every
PR has a GPU data-plane contract. CPU memory, scheduler, plugin lifecycle, and
process-supervisor tasks intentionally request no GPU.

## Review policy

- Prompt evidence comes from the original PR/Issue and public follow-up reports.
- Instructions expose the hardest fair external symptom or user contract, not
  the target file, private helper, queue, or Oracle design.
- Verifiers judge production behavior and deterministic resource work. Exact
  private names and exact error prose are rejected as grading criteria.
- Release controls are Base=0, Oracle=1, plausible incomplete patch=0, and an
  independently structured correct patch=1. Real-agent results do not repair an
  invalid control.
- GPU claims require a real device operation. Mocked ranks are described as a
  reduced unit boundary, not multi-GPU serving evidence.

## Prompt/provenance review

| Task | Real source contract | Hard prompt exposure | Static status |
|---|---|---|---|
| `vllm-pr-34179` | PR #34179 adds DCP to GPU Model Runner V2, including CUDA graphs; roadmap #32455 supplies the rollout context. | Existing DCP configuration fails or disagrees under V2; support eager/graph and preserve non-DCP. No slot formula, constructor, file names, or public reduced test. | Prompt aligned. Real CUDA checks cover block capacity, slot mapping, replay, and graph attention metadata. A previously accepted Agent patch now correctly fails for omitting DCP-local graph metadata. TP4/DCP4 serving remains outside the reduced boundary. |
| `vllm-pr-34183` | PR #34183 / Issue #28726 report unbounded host memory in repeated Qwen2.5-VL prefix-cache serving because completed requests retain large MM state. | Workload and CPU-memory symptom only; no reference-cycle diagnosis or reduced weakref script. | Prompt aligned. CPU lifecycle task by design. Static AST `gc.collect` inspection removed; behavior now determines release. |
| `vllm-pr-34246` | PR #34246 changes multimodal masking; Issue #38257 reports Qwen3-VL multi-image OOM in the embedding merge, including eager mode. | Production OOM/synchronization/resource symptom and the observed CPU-mask trigger; no implementation recipe, target file, or public reduced test. | Prompt aligned. Verifier executes CUDA merge, sync-debug, ordering, mismatch, dtype, and peak allocation controls; diagnostic messages are graded semantically rather than verbatim. Full 235B serving is represented by a reduced resource test. |
| `vllm-pr-35781` | PR #35781 reports remote-KV waiting queue churn and a roughly 9% end-to-end improvement in the supplied NIXL benchmark. | Idle tick work scales with unchanged remote waiters; completion/failure must resume correctly. No blocked-queue design/name. | Prompt aligned. Private `skipped_waiting` assertion removed. Base measured 120 redundant callbacks; Oracle measured at most one and passed event/FCFS/accounting/abort controls. |
| `vllm-pr-39337` | PR #39337 introduces tri-state V1/V2 selection and initially defaults supported dense Qwen3 text generation to V2. | User setting semantics, eligible rollout set, unsupported fallback/error, and consistent engine-wide choice. No config-class or GPUWorker target. | Released in PR #3 after Base=0, Oracle=1, and an independently structured accepted candidate=1. |
| `vllm-pr-39832` | PR #39832 removes the warned pre-v0.12 two-argument external connector constructor. | Plugin-author lifecycle: current config arrives once; retired plugin fails early; internal exceptions are not retried. | Prompt aligned. Exact “2-argument/third” prose check relaxed to migration-relevant connector/config context. CPU control-plane task. |
| `vllm-pr-40841` | Feature issue #40814 and PR #40841 define a node-local multi-port DP supervisor with aggregate readiness and group shutdown. | A deployment operator reports that a dead rank leaves the node receiving traffic, then requests the two public CLI flags, three accepted-head probe paths, rank/port/device mapping, validation, readiness, failure, signal, and cleanup. No module or class name. | Prompt aligned. The verifier launches public `vllm serve`; Base=0, full Oracle=1, wrong-public-CLI Opus patch=0, and a module/class/helper-renamed equivalent=1. |
| `vllm-pr-42430` | PR #42430 fixes Mamba NIXL PD FULL-CG divergence by treating a prior-state one-token extend as decode. PR #43034 later proposes a revert after Hybrid SSM/HMA accuracy fell from about 0.80 to 0.75. | A serving operator reports silent NIXL/FULL-CG token divergence; no metadata diagnosis or reproduction. The later regression is recorded as Oracle risk rather than exposed as an impossible contract. | Prompt aligned to the historical PR. Real CUDA checks cover ordinary decode, prior-state one-token recomputation, first-token and speculative-prefill guards, a mixed batch, and synchronization debug. Full NIXL/model accuracy remains outside the focused boundary. |

## Dynamic evidence

| Task | Base | Oracle | Evidence |
|---|---:|---:|---|
| `vllm-pr-34183` | 0 | 1 | 11/11 requests and payloads retained on Base with cyclic GC disabled; Oracle releases them while preserving initial, append, and streaming hash updates. |
| `vllm-pr-35781` | 0 | 1 | Base performs 120 remote-KV callbacks for 24 waiters over 5 idle rounds; Oracle passes the deterministic-work and completion/lifecycle checks. |
| `vllm-pr-39337` | 0 | 1 | Base lacks automatic selection; Oracle and an independently structured selector pass tri-state resolution, eligibility, incompatibility, and production-consumer propagation. |
| `vllm-pr-39832` | 0 | 1 | Base still accepts the retired connector constructor; Oracle and an equivalent lifecycle implementation pass, while a compatibility-retry candidate is rejected. |
| `vllm-pr-34246` | 0 | 1 | Base reaches the production CPU-mask/CUDA device mismatch; Oracle passes sync, allocation, dtype, order, identity, and cardinality checks. Incomplete Opus=0 and independent CPU-index GPT patch=1. |
| `vllm-pr-40841` | 0 | 1 | Public CLI black-box test passes port/rank/device mapping, aggregate probes, child death, live-unhealthy shutdown, SIGTERM forwarding, and socket cleanup. Frozen wrong-CLI patch scores 0; renamed equivalent implementation scores 1. |
| `vllm-pr-42430` | 0 | 1 | Base leaves the prior-state row prefill-shaped; Oracle reclassifies only that row and passes first-token, wider speculative-prefill, mixed-batch, ordinary-decode, and CUDA sync controls on H20. |
| `vllm-pr-34179` | 0 | 1 | Strengthened real-CUDA verifier rejects both earlier slot-only work and the fresh Opus-5 budget-limited constructor-only patch; a private-helper-renamed Oracle remains accepted. |

## Submission map

| Task or infrastructure | Pull request |
|---|---:|
| `vllm-pr-39337` | #3 |
| `vllm-pr-34183` | #4 |
| `vllm-pr-39832` | #5 |
| `vllm-pr-35781` | #6 |
| `vllm-pr-40841` | #9 |
| `vllm-pr-34246` | #10 |
| `vllm-pr-34179` | #14 |
| `vllm-pr-42430` | #15 |
| `create-task` construction guidance | #11 |
| GPU-aware vLLM environment template | #12 |

## Known release blockers

1. #34179 needs either an affordable real TP/DCP serving fixture or an explicit
   benchmark-boundary note in task metadata and README.
2. #42430 is releasable only as the explicitly bounded historical metadata
   contract. It must not be described as proving Hybrid SSM/HMA or end-to-end
   NIXL accuracy because the later upstream regression remains material.
3. One Agent attempt is not a pass-rate estimate. The fresh Opus-5 runs recorded
   here are verifier probes; the requested ten-run pass@k campaign remains a
   separate dynamic-analysis phase.
