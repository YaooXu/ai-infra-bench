# Eligibility and build validation

Status: blocked before Agent construction. This issue is not publishable as one
benchmark task without curator scoping.

## Survey and issue audit

Survey record `vllm__issue__27433` is an open feature program titled
`Batch Invariant Feature and Performance Optimization`. The 2026-08-08 survey
snapshot records 67 comments, no source SHA, and these explicit blockers:
`open_umbrella_issue`, `not_atomic`, and `no_single_gold_patch`. The issue remains
open. The current discussion has 75 comments; all were read along with all 260
timeline events across three API pages.

The issue starts by saying that the basic framework is already supported, then
tracks a changing matrix of follow-up work:

- FlashInfer, FlashAttention, Triton MLA, and still-open FLASHINFER_MLA;
- DeepSeek V3, R1 TP8, dense and MoE models, AWQ/FP4 and other quantization;
- Torch compile/CUDA graphs, BMM/GEMM/RMSNorm/log-softmax performance;
- prefix caching, speculative decoding, DP/EP, AMD, XPU, and model validation;
- correctness across batch size/order plus device-specific performance tax.

There is no universal failing Base symptom: the tracker explicitly says the
basic framework works while different combinations remain unsupported, slow,
untested, or intermittently nondeterministic.

## Solution mapping

There is no closing PR or single mapped solution. Representative links prove
that this is a program rather than one patch:

| Item | State / scope | Why it is not an issue-level gold |
| --- | --- | --- |
| PR #25603 | merged, basic framework, 8 files | Merged before tracker creation; only step 1/n |
| PR #29345 | merged, BMM performance, 3 files | Atomic child already has its own survey task |
| PR #40408 | merged, CUTLASS FP8 performance, 11 files | Hardware/kernel child already independently surveyed |
| PR #42456 | merged, SM80 compile mode, 2 files | Enables one A100 path, not full determinism support |
| PR #48391 | merged, RMSNorm block-size fix | Maps to the later #51187 sub-bug, not the umbrella |
| Issue #47359 | open | FLASHINFER_MLA support is still unfinished |
| PR #46592 | open at survey cutoff | Prefix-cache support remains unfinished |
| PR #51287 | open at survey cutoff | Async-scheduling interaction remains unfinished |

Other merged children target DeepSeek V3 on 8xH100, R1 TP8 on Blackwell,
DeepGEMM Blackwell, NVFP4 linear/MoE, CI, docs, and individual model coverage.
Their mutually different bases cannot be combined into a canonical issue base.
Open or post-cutoff speculative decoding, ROCm, XPU, prefix-cache, and performance
work further prevents treating the issue as solved.

Issue #51187 is potentially suitable for a separate atomic survey: it names a
v0.26.0/Qwen3-0.6B behavior and later evidence strongly maps it to PR #48391.
That mapping must be curated as its own task; importing it here would shrink the
wrong problem and erase the rest of #27433.

## Hardware and topology eligibility

The requested host provides one NVIDIA A100-SXM4-40GB (SM80). That device can
run vLLM and some individual batch-invariance paths, but it is not qualified for
the full issue contract:

- Maintainers initially stated that Hopper and Blackwell were the officially
  supported/tested devices and A100 community work was not well tested.
- A100 sub-issue #32658 reports a direct negative result: with Qwen3-4B target,
  Qwen3-0.6B draft, 80 prompts, the H100 batch-invariant case matched while A100
  retained 56-62 mismatches. It closed as `not_planned` due inactivity, not as
  fixed.
- PR #42456 later enabled compile mode on SM80, and model-validation docs were
  widened to SM80+, but that does not resolve speculative decoding, MLA,
  quantized/MoE kernels, prefix caching, or the issue-wide hardware matrix.
- Core tracked demonstrations require 8xH100, TP8 Blackwell, B200/B300, H20,
  RTX/Ada, AMD, or XPU. One A100 GPU0 cannot preserve those topologies.

A real infrastructure-only probe used the required isolated daemon and GPU0:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
docker run --rm --network none --gpus device=0 \
  --entrypoint /usr/local/bin/python3 \
  vllm/vllm-openai:v0.20.2 \
  -c '<torch/vllm native capability probe>'
```

Observed:

```text
vllm 0.20.2
torch 2.11.0+cu130
cuda_available True
visible_count 1
gpu NVIDIA A100-SXM4-40GB
capability (8, 0)
allocation 16.0
target_device None
native /usr/local/lib/python3.12/site-packages/vllm/_C.abi3.so
```

This proves the daemon, native runtime, non-empty target, runtime no-network,
and GPU0 access. It does not qualify A100 for the umbrella behavior.

## Reproduction cost and external dependencies

A correct full-system reproduction is not a single test. It would require a
model/kernel/topology matrix spanning at least small Qwen, DeepSeek V3/R1,
quantized and MoE models, multiple attention backends, cold/warm prefix cache,
compile/eager modes, and batch-size/order variation. Several referenced runs
use 4 or 8 GPUs and models from 0.6B through 120B; model weights and tokenizer
artifacts must be acquired and digest-locked at build time.

Using a tiny synthetic model or one cached Qwen model would remove the exact
kernel shapes, reductions, quantization, communication, and cache interactions
that cause the tracked failures. That would shrink away the problem rather than
minimize it. Runtime model download is also incompatible with the required
offline contract.

## Agent / Base / Oracle / Verifier status

- **Agent: blocked.** The issue supplies no exact candidate SHA or canonical
  tree. No Dockerfile or task image was created, because choosing a child base
  or inferred time-based HEAD would falsely claim exactness.
- **Base: blocked.** There is no one failing issue-level behavior, model, or
  topology. Basic support already passes while separate matrix cells fail.
- **Oracle: blocked.** No closing solution exists. Multiple children remain open
  and merged children solve disjoint contracts. No solved code enters Agent
  assets.
- **Verifier: blocked.** Without a two-sided atomic contract, a public Dev would
  either test source/meta state, pass vacuously, or benchmark only one child.
  No such misleading asset is emitted.

No `docker build` was run and no `ai-infra-bench/vllm-issue-27433:base` image was
tagged. The real A100 probe above is intentionally classified as infrastructure
preflight, not an Agent/Base result.

## Unblocking options

The preferred action is to reject this umbrella item and curate one child:

1. package an existing atomic PR such as #29345 or #40408 under its own task;
2. separately survey #51187 with an explicit v0.26.0 base and validated #48391
   mapping; or
3. choose one still-open axis (for example FLASHINFER_MLA or prefix caching),
   wait for a merged solution, and record an exact source/model/topology lock.

## Survey-manual feedback

- Issue-only records must include a curator-selected `base_sha`; `as_of` and
  issue creation time are not interchangeable source cutoffs.
- Open umbrella programs with multiple child PRs are not benchmark tasks. Split
  them before environment work rather than selecting a convenient child later.
- A merged hardware-enablement PR or documentation gate does not establish
  issue-wide hardware qualification; require behavior evidence for the exact
  model, backend, and topology.
- Treat infrastructure GPU/native probes separately from Agent/Base evidence.
  A successful import/allocation cannot legitimize a missing source contract.
- Do not minimize kernel determinism tasks with synthetic weights when tensor
  shapes, quantization, communication, or cache state are causal dimensions.
- Feature-addition Verifiers need both a failing Base and a mapped positive
  Oracle. Source-string, issue-state, or missing-symbol tests are insufficient.

## Remaining uncertainty

The technical program will continue to evolve after the survey cutoff, and
some individual A100 paths may now work. That does not remove the structural
blocker: no single exact Base/Oracle pair represents issue #27433 as a whole.
