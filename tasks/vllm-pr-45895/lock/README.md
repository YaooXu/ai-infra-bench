# Source, model, and eligibility lock

This survey item comes from merged vLLM PR
[#45895](https://github.com/vllm-project/vllm/pull/45895), “Indexer init
skip and MTP TopK share for iteration”. The source solution is stable, but the
task is **hardware- and model-blocked before environment construction** on the
assigned A100 worker. This is an audit record, not an Agent environment.

## PR eligibility and solution mapping

The PR was approved by a vLLM maintainer on 2026-06-17 after confirmation with
the model vendor, and was squash-merged on 2026-06-19. It remains closed and
merged with six contributor-branch commits and nine changed files.

The survey snapshot pair is a valid ancestor pair:

- base `e1a5fc406b248925427a84beefbd123924f0c9e5`;
- head `71c64be3160377b64630d9f206b3858caf591988`;
- GitHub compare: `ahead_by=6`, `behind_by=0`, merge base equals the survey
  base;
- scope: nine files, 69 additions, 30 deletions.

The GitHub `merge_commit_sha` is actually a squash commit with one parent. A
future benchmark should prefer the accepted main-branch mapping:

- Agent Base: squash parent
  `9ea3a4015b412d146d38ee1b697aafe92979c6ae`;
- evaluator Oracle: merged commit
  `ab666069935c1f23e8ef56038b4659ac9e8f19f8`;
- GitHub compare: `ahead_by=1`, `behind_by=0`, merge base equals the squash
  parent, with the same nine-file 69/30 patch.

Locked source identities measured during the audit:

| Role | Commit | Git tree | Archive bytes | Archive SHA-256 |
|---|---|---|---:|---|
| Survey base | `e1a5fc406b248925427a84beefbd123924f0c9e5` | `5377a28f3f7e3f4fb2a37b86fd4ad1033f6bceab` | 36,985,588 | `620ccd01bed9952d487ef245d94463916a2381177c34eb665b4c362e2098aef6` |
| Survey head | `71c64be3160377b64630d9f206b3858caf591988` | `02bc3741c856c834c246cb006b6ee4607ceda12a` | 36,990,163 | `6fd832c6faec4582500459194e16b13c92c48ed466d8b4d08668e878ffdd3bc5` |
| Accepted Base | `9ea3a4015b412d146d38ee1b697aafe92979c6ae` | `754ca81b1ff8f63be9280445ab5e8745055114d4` | 36,997,319 | `9a07a38d6214265dcd4941aadc6bf034ad188d96578826842cdd80676a72d02b` |
| Accepted Oracle | `ab666069935c1f23e8ef56038b4659ac9e8f19f8` | `53f1e85b25db3cc867219df4428637ce41b6e681` | 36,999,347 | `a47edfa208d81c01cee08652fceee8705bcf4606167119eee1d1a1205e51c5ab` |

The archives came from the immutable GitHub archive endpoint for each commit.
They were downloaded only for read-only source inspection and hashing; they
are not Docker inputs because no image was built.

## Atomicity

The accepted patch addresses two coupled GLM-5.2 speculative-decoding faults:

1. backbone layers that skip Top-K indexer construction must still consume the
   iteration-shared Top-K indices buffer; and
2. GLM/DeepSeek MTP must recycle the post-final-norm hidden state through the
   proposer tuple contract.

The shared behavioral symptom is low MTP acceptance. The patch touches the
model, proposer, MLA layer, and four sparse MLA backends to keep the same
buffer contract across NVIDIA, ROCm, and XPU paths. This is a coherent
cross-layer fix, not an unrelated patch bundle.

## Exact model dependency

The PR's reproduction command uses `zai-org/GLM-5.2-FP8` with tensor
parallelism 8. The latest model revision available when the PR was tested and
merged was:

| Item | Locked value |
|---|---|
| Hugging Face revision | `a0b55e88465d1a06afece97bc8d6b366aff39089` |
| Revision time | 2026-06-17 09:32:09 UTC |
| Repository files | 150 |
| Total repository bytes | 755,663,672,710 |
| Safetensors shards | 141 |
| Safetensors bytes | 755,632,050,320 |
| Parameter count | 753,375,793,584 |
| `config.json` SHA-256 | `d1539d36be7546a1d827fe9cf74c55874695652efb6a5aaa3e60cde1c76ba819` |

The model is public and ungated, but it is not downloaded into a Docker build
context. A future eligible worker should keep this revision in an external,
read-only cache, lock every downloaded file digest, and run the final verifier
with network disabled.

## Hardware gate

The assigned worker has eight NVIDIA A100-SXM4-40GB GPUs, compute capability
8.0, for 320 GiB aggregate nominal device memory. The locked FP8 checkpoint's
weight files alone occupy about 703.73 GiB, over 2.19 times the worker's total
device memory before allocator overhead, activations, CUDA Graphs, KV cache,
or MTP state.

The exact base CUDA platform reports native FP8 support only at compute
capability 8.9 or newer, and its FP8 KV-cache path requires native `fp8e4nv`
on SM89+. The original command explicitly selects `--kv-cache-dtype
fp8_e4m3`, while A100 is SM80. In the PR discussion, an upstream participant
answered the direct “Can A100 run GLM-5.2?” question with “No, you need H200
or higher. Ideally B200 or B300.” The PR's own test plan uses tensor
parallelism 8.

These are independent blockers: neither more host RAM nor a smaller prompt
would give A100 native FP8 support or enough aggregate device memory for the
real model.

The source and model locks above do not authorize publishing a reduced model
as a substitute. A tiny synthetic GLM configuration would expose the relevant
branches and tuple shapes directly, substantially reducing the diagnosis
search space while failing to reproduce the reported acceptance-length
symptom.
