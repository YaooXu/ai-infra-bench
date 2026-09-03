# Environment lock

This environment packages the survey base state for `vllm__pr__34183`.

## Source

- Upstream repository: `https://github.com/vllm-project/vllm.git`
- Survey base commit: `e94ec597334d9a3e9b0d04bc17152e2747c83d51`
- Commit date: `2026-02-10T01:18:42Z`
- Commit subject: `[LMCache] Token Base IPC API (#34175)`
- Acquisition: depth-one fetch of the exact commit during image build
- Runtime Git state: exported source plus one synthetic commit named
  `Synthetic benchmark base`, branch `benchmark-base`, no upstream remote

The original issue reported vLLM 0.11.0, PyTorch 2.8.0, and CUDA 12.8. The
required survey base is later and was committed between v0.15.1 and v0.16.0;
its own build files had already moved to PyTorch 2.10.0 and CUDA 12.9.1. The
official v0.15.1 image is the latest release that existed at the base commit's
cutoff and predates the target fix. A later release image is not used because
its installed source could disclose post-cutoff implementation changes.

## Base image and runtime

- Image: `vllm/vllm-openai:v0.15.1`
- Multi-platform manifest digest:
  `sha256:8c9aaddfa6011b9651d06834d2fb90bdb9ab6ced4b420ec76925024eb12b22d0`
- Linux/amd64 image digest:
  `sha256:06f9f0d5c7cb079504615c51dab70cd18abbf609d1358b940172181ac0a92efa`
- Linux/amd64 compressed size reported by Docker Hub: `9,125,854,482`
  bytes
- Platform: `linux/amd64`
- Python: 3.12.12
- vLLM package: 0.15.1
- PyTorch/CUDA: 2.9.1+cu129 / CUDA 12.9
- Ubuntu packages added at build time:
  - `ca-certificates=20260601~22.04.1`
  - `git=1:2.34.1-1ubuntu1.17`
  - `git-man=1:2.34.1-1ubuntu1.17`
  - `liberror-perl=0.17029-1`
- Apt sources: the Ubuntu Jammy, Jammy updates/security/backports, NVIDIA CUDA,
  and Deadsnakes repositories already configured by the official base image
- Accelerator used for validation: NVIDIA A100-SXM4-40GB, GPU 0
- Runtime network: disabled with `--network none`
- `VLLM_TARGET_DEVICE`: not overridden

The official image's CUDA compatibility directory contains
`libcuda.so.575.57.08`. On the validation host with driver 580.126.20, that
compatibility library returns CUDA error 803. The image therefore gives
`/lib/x86_64-linux-gnu` priority via `LD_LIBRARY_PATH`, selecting the NVIDIA
Container Toolkit's host-driver injection. A real CUDA tensor allocation is
part of validation.

The build needs network access only for Ubuntu apt metadata/packages and the
exact Git commit fetch. On the validation host, apt uses its directly reachable
official repositories while GitHub uses the supplied build proxy. Runtime
reproduction has no external data, model, tokenizer, or package dependency.

## Reproduction assets

`environment/public_dev/reproduce_retention.py` uses only Python's standard
library and the vLLM runtime already present in the image. It creates a small,
deterministic set of completed requests with request-associated multimodal
payloads and reports whether normal owner release promptly reclaims them.
