# Environment lock

This environment packages the survey Base for `vllm__pr__34246`.

## Source cutoff

- Upstream repository: `https://github.com/vllm-project/vllm.git`
- Base commit: `31a719bcd37a195107711dc8b498288e49ef8576`
- Canonical root tree: `a45811fe928b168245e519e4205bbd99c6fa3f57`
- Commit date: `2026-03-31T23:22:23Z`
- Commit subject: `[ROCm][perf] fix Aiter sparse MLA with MTP>1 (#37887)`
- Archive SHA-256: `f248739d9eb5591e5282be31b323a499a3643426221aa1dd8d58caba3baf7f67`
- Runtime Git: one synthetic commit, branch `benchmark-base`, no remote

The closest official release before the source cutoff is v0.18.1, published
2026-03-31 at 00:53Z. Both candidate and release use PyTorch 2.10; the official
image reports `2.10.0+cu129`.

## Base image and native binding

- Image: `vllm/vllm-openai:v0.18.1`
- Repository digest:
  `sha256:228113d30448941e7a845f57ef0b3d3ea74ffda81be72ded4f8d6dfab0124fe6`
- Local image size: `9,578,701,871` bytes
- Accelerator probe: NVIDIA A100-SXM4-40GB, GPU 0
- `VLLM_TARGET_DEVICE`: not overridden

Exact candidate Python is first on `PYTHONPATH`. The v0.18.1 image supplies
seven regular ELF shared objects and generated `_version.py`. Exact Base also
imports `_C_stable_libtorch`, which the earlier same-day release does not
package. One additional regular ELF object is therefore copied by explicit path
from the same-PyTorch v0.19.0 image:

- Donor: `vllm/vllm-openai:v0.19.0`
- Donor digest:
  `sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504`
- Donor created: `2026-04-03T00:07:37.341665339Z`
- Whitelist: `vllm/_C_stable_libtorch.abi3.so` only

`base-native.sha256` locks the eight v0.18.1 artifacts before copying;
`native.sha256` locks the final eight `.so` files plus `_version.py`. The
Dockerfile rejects missing, symlinked, non-ELF, additional, or hash-mismatched
objects. All generated/native files are added after the canonical synthetic
commit, and no release/donor Python or staging directory is added to
`PYTHONPATH`.

This is a source-exact but native-approximate environment, not an exact source
build. Before overlay, v0.18.1 passed `vllm._C`, `vllm._custom_ops`, Torch CUDA
allocation, and the A100 device probe. The narrowly selected donor is
post-cutoff by about two days, although both images use Torch 2.10.0+cu129; its
ABI compatibility is verified by import/probe rather than guaranteed by source
identity. Candidate production import succeeds, but loading Base's regular
extension together with the donor stable-libtorch extension emits four
duplicate-operator registration diagnostics. Untouched v0.19.0 does not emit
them, so this remains a known mixed-build limitation, not an upstream packaging
property. The target merge behavior is Python/PyTorch indexing, executes no
vLLM native kernel, and passes the isolated Oracle despite those diagnostics.

Build networking is used only for apt and the exact SHA-checked source archive.
Runtime is offline and requires no model, tokenizer, image, dataset, or service.

## Public reproduction

The public Dev dynamically executes the production
`_merge_multimodal_embeddings` function with CUDA embeddings. It checks the
legacy GPU-mask path, the intended CPU-mask path under CUDA sync-debug error
mode, numerical placement, temporary allocator growth, and strict rejection of
excess multimodal embeddings. It does not inspect source text.

The primitive scope mirrors the upstream CUDA unit test and preserves the
causal operation. It does not claim to reproduce the full 235B/8xH100 serving
topology; that end-to-end issue was separately confirmed by upstream reporters.
