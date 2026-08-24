# Eligibility and build validation

Status: blocked before Agent construction. The issue is not publishable as one
benchmark task without curator scoping.

## Exhaustive issue audit

Survey item `vllm__issue__41286`, `Migration from Model Runner v1 to Model
Runner v2`, is an open feature roadmap. Its 2026-08-08 snapshot records eight
comments and the explicit blockers `open_umbrella_issue`, `not_atomic`, and
`no_single_gold_patch`; it provides no source SHA. The live issue remains open.
All eight comments and all 58 timeline events were read; timeline pagination
pages 2-4 were empty, so the cross-reference audit is complete.

The roadmap explicitly says migration is gradual:

- begin with Qwen3/OPT dense models;
- add Llama/Mistral and Qwen/DeepSeek MoE;
- add quantized and pooling models;
- eventually cover a popular model and switch MRV2 on by default.

The last two gates remain unchecked. The linked `Model Runner V2 Remaining
TODOs` issue #47172 is also open and still lists sequence parallelism, logits
processors, multimodal, NGram/draft speculative decoding, fast KV-sharing
prefill, Elastic EP, dual batch overlap, and other gaps.

## Solution mapping

There is no closing PR or single fix. Top-level migration PRs demonstrate
distinct source cutoffs and disjoint contracts:

| PR | Contract | Base SHA / state |
| --- | --- | --- |
| #39337 | Qwen3 dense config oracle `[1/N]` | `c7560af4...`, merged |
| #43458 | Llama/Mistral dense | `8a9eb408...`, merged |
| #42667 | Qwen/DeepSeek V2 MoE `[3/N]` | `272c1695...`, merged |
| #44443 | all dense defaults | `4787f2dd...`, merged |
| #44446 | quantized defaults `[5/N]` | `0a7bacdc...`, merged |
| #45461 | GraniteMoE default | `d467a2a7...`, merged |
| #48290 | pooling defaults | `f4b161d7...`, merged after survey cutoff |
| #43915 | Elastic EP support | closed without merge |
| #46646 | all non-pooling defaults | `da329cc3...`, still open |

The issue body links dozens of prerequisite/follow-up PRs for prompt embeds,
weight reload/offload, KV connectors, prompt logprobs, CUDA graphs, acceptance
length, reasoning parsing, distributed P/D, pooling modes, and model-specific
features. Their different bases cannot be merged into one canonical issue base.

PR #39337 is separately surveyable only under its narrow tri-state/config
oracle. Treating its positive result as completion of #41286 would ignore the
subsequent `[2/N]` through `[5/N]` migrations, remaining TODOs, and the still
open default-switch PR.

## Hardware, model, and topology eligibility

The available host provides one NVIDIA A100-SXM4-40GB (SM80). It can execute
some child paths, but it cannot preserve the roadmap's full test matrix:

- A concrete CI OOM discussed in the issue requires an NVIDIA L4; a contributor
  with A100 could not reproduce it without L4 access.
- A performance data point uses GB10/DGX Spark, aarch64 unified memory, a
  122B-A10B INT4 MoE model, and MTP speculative decoding.
- Child PRs cover multi-GPU NIXL P/D + speculative decode, Qwen and DeepSeek MoE,
  quantized models, pooling, CUDA graph/compile, connector, and offload paths.
- The roadmap names Qwen3-0.6B, OPT-125M, DeepSeek-V2-lite, and a future popular
  model; one tiny model cannot represent architecture-wide default selection.

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
gpu NVIDIA A100-SXM4-40GB
capability (8, 0)
visible_count 1
allocation 8.0
target_device None
native /usr/local/lib/python3.12/site-packages/vllm/_C.abi3.so
```

This proves the daemon, A100 allocation, native imports, non-empty target, and
runtime no-network. It is not an MRV2 migration Agent/Base result.

## External dependency and reproduction cost

A faithful issue-wide verifier would need multiple models and tokenizer/weight
digests, dense/MoE/quantized/pooling modes, compile/eager and CUDA graphs,
speculative decoding, connector/offload/distributed configurations, plus L4,
A100/Hopper/Blackwell, and GB10-like platforms. Some paths require several GPUs
and large or access-controlled models.

Using only OPT-125M, a synthetic tiny model, or a config-only property would
erase the kernels, memory pressure, distributed lifecycle, and architecture
selection that distinguish later migration stages. That is valid only for a
separately scoped child such as PR #39337, not for this issue.

## Agent / Base / Oracle / Verifier status

- **Agent: blocked.** There is no exact candidate SHA or canonical tree. An
  issue-creation or survey-date HEAD would be an ungrounded inferred base.
- **Base: blocked.** No single model/topology expresses the migration roadmap;
  many earlier stages already pass while later feature cells remain unsupported.
- **Oracle: blocked.** The issue is open, #46646 is open, #43915 did not merge,
  and #47172 lists multiple unfinished behaviors. No solved code enters Agent
  assets.
- **Verifier: blocked.** A public Dev covering one model or source/meta state
  would pass vacuously or shrink the task. No misleading test is emitted.

No `docker build` was run and no `ai-infra-bench/vllm-issue-41286:base` image
was tagged. The A100 command above is strictly an infrastructure preflight.

## Unblocking options

Reject the umbrella item and curate one child:

1. retain PR #39337 as a config-contract/environment-only task;
2. wait for #46646 or one #47172 TODO to merge, then lock its exact Base/Oracle;
3. select a model-specific crash/performance regression with the same hardware
   and topology available to the benchmark; or
4. create separate migration tasks for dense, MoE, quantized, pooling, and
   distributed paths instead of one impossible full-system oracle.

## Survey-manual feedback

- Issue-only roadmap records require a curator-selected `base_sha`; `created_at`
  and `as_of` do not define a canonical code tree.
- A checklist with many independently merged PRs is a program, not an atomic
  benchmark, even if most boxes are checked.
- Do not treat `[1/N]` infrastructure or a default-selection oracle as proof of
  full migration completion.
- Require the exact model, accelerator architecture, and distributed topology;
  a generic CUDA/native probe is environment evidence only.
- If minimizing to one model removes later runner features or hardware-specific
  failures, split the survey item instead of publishing the minimization.
- Feature migrations need a mapped closing solution and a two-sided behavioral
  Oracle; source strings, checklist state, or missing symbols are insufficient.

## Remaining uncertainty

Individual MRV2 coverage continues to improve after the survey cutoff, and more
boxes may close. The structural blocker remains until the umbrella issue closes
or a curator extracts one exact child Base/Oracle contract.
