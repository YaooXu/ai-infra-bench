# Environment lock

This eligibility environment packages survey item `vllm__pr__20087` at its
exact base revision.

## Source

- Upstream: `https://github.com/vllm-project/vllm.git`
- PR: `https://github.com/vllm-project/vllm/pull/20087`
- External dependency PR: `https://github.com/deepseek-ai/DeepGEMM/pull/112`
- Base commit: `b854321ffe50fd04c6b1ac58eecdab4caf5b4295`
- Base date: `2025-07-10T23:06:37Z`
- Base subject: `[Docs] Lazy import gguf (#20785)`
- Head inspected: `f490831d86cdbc71ce8c835e88d0d84bedc6602d`
- Base archive SHA-256:
  `fd58dd99ebcef8c7470fe87c714dd29be68b20cb78f0c1e7be2dea53d43e908d`
- Canonical forced-add tree: `d2526d84114539875b099b79ca4002f1d39c977e`
- Runtime Git: one synthetic commit, branch `benchmark-base`, no remote

The base is after v0.9.2 (2025-07-07) and before v0.10.0 (2025-07-24), so the
Dockerfile pins the official v0.9.2 amd64 manifest:
`sha256:37cd5bd18d220a0f4c70401ce1d4a0cc588fbfe03cc210579428f2c47e6eac33`.

## Base image and runtime

- Donor image ID / digest:
  `sha256:37cd5bd18d220a0f4c70401ce1d4a0cc588fbfe03cc210579428f2c47e6eac33`
- Donor image size: `10,674,731,647` bytes
- Platform: linux/amd64, Ubuntu 22.04
- Python: 3.12
- vLLM donor: 0.9.2
- PyTorch: `2.7.0+cu128`
- CUDA reported by PyTorch: 12.8
- Runtime user: `agent` (UID 1000)
- Accelerator used for probe: physical GPU 2, NVIDIA A100-SXM4-40GB,
  capability 8.0, UUID `GPU-3815a178-ad22-4b81-5669-0533760a7e6b`

Exact base Python sources replace the release Python tree. The donor supplies
only same-release generated/native files:

```text
_C.abi3.so
_flashmla_C.abi3.so
_moe_C.abi3.so
_version.py
cumem_allocator.abi3.so
vllm_flash_attn/_vllm_fa2_C.abi3.so
vllm_flash_attn/_vllm_fa3_C.abi3.so
```

Real native import and CUDA allocation pass with both source and `_C` resolving
under `/workspace/repo`. This is a same-release ABI approximation, not an
exact-SHA native compilation.

## Hardware and dependency boundary

The feature requires SM100/B200 and DeepGEMM v2. The available A100 is SM80.
This image therefore locks vLLM source/native state and performs a real CUDA
eligibility probe, but intentionally does not package a future DeepGEMM v2
checkout that cannot be faithfully executed here. This is an eligibility
artifact, not a runnable target-kernel benchmark.

The vLLM PR links DeepGEMM PR 112 but does not record the exact dependency SHA
installed in the author's environment. At the vLLM base cutoff, the newest
commit visible in that external PR was
`cc416ee4faf0533a9263c2de814e5565f56ca1cc`; three more commits, including
the merge with upstream, landed on 2025-07-18 after vLLM PR 20087 merged. A
future SM100 verifier must confirm and lock the compatible external revision
instead of silently using today's PR head.

No model, tokenizer, or dataset is included. Runtime uses `--network none`, a
non-root writable user, and does not set `VLLM_TARGET_DEVICE`.
