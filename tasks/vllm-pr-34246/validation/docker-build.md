# Docker build and validation

Status: environment-ready / primitive-scoped. Real A100 Base fails and the
isolated exact-diff Oracle passes. Full 235B serving remains out of scope.

## Contract and atomicity

PR #34246 has five commits, nine changed files, and `+54/-51`. Two commits merge
main into the feature branch, but the effective patch remains one coherent
contract: keep the multimodal position mask on CPU and replace
`masked_scatter_` with direct indexed assignment, updating the few model/runner
consumers that had assumed a GPU mask.

The behavior boundary is:

- CPU boolean masks merge multimodal embeddings into exact positions.
- The merge introduces no CUDA synchronization with PyTorch >=2.9.
- Temporary CUDA allocation remains bounded instead of copying the large target.
- Excess multimodal embeddings are rejected rather than silently truncated.

The public Dev exercises the real production function and allocator; it does
not inspect a signature or source string.

## Solution and issue mapping

Issue #38257 reports a 100% OOM on Qwen3-VL-235B-A22B-Instruct, 8xH100, TP8,
DCP2, EP, data-parallel multimodal encoder, three large images, and roughly 120K
text tokens. Eager mode and removal of DCP did not fix it. A maintainer identified
`masked_scatter_` as copying large image-embedding data; the reporter then
verified that this PR removed the OOM and allowed ChartQA to run concurrently.
The issue was closed with `Fixed by #34246`.

ChartQA accuracy stayed within reported noise (`0.8740` vs `0.8716` anywhere
accuracy). The PR author also profiled Qwen3-VL-2B on one L40S and added a CUDA
unit test that rejects synchronization. This is a unique solution mapping.

## Hardware and dependency closure

The exact 235B reproduction requires 8xH100 and external model/images, so it is
not available on one A100 40GB. The production merge primitive is independent
of model weights and executes the same CUDA tensor assignment. A synthetic
embedding tensor preserves the causal mask/device/allocation behavior without
shrinking away the operation; it is the same boundary selected by upstream's
new unit test.

The fix depends on PyTorch PR #156384, present since PyTorch 2.9, which removed
the CPU-mask indexing synchronization. Candidate and release use PyTorch 2.10.
No model or dataset dependency is needed for the scoped Verifier.

## Docker daemon

All Docker commands use:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
```

The daemon data root is
`/data/yaoyaoyao/pr34183-cuda-build/docker-data`; the default daemon is not used
and no pruning or deletion is performed.

## Build and validation evidence

### Image selection and closure diagnostic

The untouched pinned v0.18.1 image reported vLLM 0.18.1, Torch
`2.10.0+cu129`, successful A100 allocation, and successful `vllm._C` and
`vllm._custom_ops` imports. Its image identity was:

```text
sha256:228113d30448941e7a845f57ef0b3d3ea74ffda81be72ded4f8d6dfab0124fe6
size=9578701871
created=2026-03-30T23:21:32.922351086Z
```

An initial source-exact build exposed a real closure gap rather than the target
failure: Base Python imports `vllm._C_stable_libtorch`, but the closest earlier
same-day release does not package that object. That first image (`ff2af09383e3`,
514.24 seconds) stopped at `ModuleNotFoundError`; it is not the accepted image.

Official v0.19.0 is the earliest available same-Torch donor found to contain the
missing object. It was probed independently with Torch `2.10.0+cu129`, A100 CUDA
allocation, and successful `_C`, `_C_stable_libtorch`, and `_custom_ops`
imports. Only this explicitly hashed relative path is copied from it:

```text
vllm/vllm-openai:v0.19.0
digest=sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504
size=9577342091
created=2026-04-03T00:07:37.341665339Z
9fbcc65fb822786590c2e7bf73a471c44e7aa1244dae23ee03c61ea9a7d6d329  _C_stable_libtorch.abi3.so
```

No donor Python directory, other donor `.so`, or staging directory reaches the
final image. `base-native.sha256` validates the eight v0.18.1 artifacts, and
`native.sha256` validates all nine final artifacts (eight `.so` plus generated
`_version.py`). Every shared object is a regular ELF and the final count is
asserted to be exactly eight.

### Final build

The final cached build used the isolated daemon and pinned local image digests:

```bash
source /data/akg_kernel_bench_lite/A100_proxy.sh
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
cd /data/ai-infra-bench/survey-builds/vllm-pr-34246
/usr/bin/time -p docker build \
  --network host --pull=false \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  -t ai-infra-bench/vllm-pr-34246:base \
  -f context/environment/Dockerfile context
```

```text
Successfully built d0d035ae792d
real 391.35
user 0.05
sys 0.04
image=sha256:d0d035ae792df56be9c47dde2f114f84be4935206349b8ca7ccc6b030a139336
size=10576787251
```

The build revalidated archive SHA-256, canonical tree, one synthetic commit,
zero remotes, both native manifests, regular ELF status, exact `.so` count, and
a clean Git worktree.

### Agent and integrity probes

All runtime commands used `--network none`; GPU probes additionally used
`--gpus device=0`. The accepted image produced:

```text
UID=1000
GID=1000
COMMITS=1
TREE=a45811fe928b168245e519e4205bbd99c6fa3f57
REMOTES=0
DIRTY=0
SO_COUNT=8
TORCH=2.10.0+cu129
CUDA=True
DEVICE=NVIDIA A100-SXM4-40GB
VLLM_SOURCE=/workspace/repo/vllm/__init__.py
C_SOURCE=/workspace/repo/vllm/_C.abi3.so
STABLE_SOURCE=/workspace/repo/vllm/_C_stable_libtorch.abi3.so
CUSTOM_OPS_SOURCE=/workspace/repo/vllm/_custom_ops.py
TARGET_DEVICE=None
NETWORK_OFFLINE=1
CUDA_SUM=1024.0
```

The non-root user can write `/workspace/repo`. A fresh container has zero
candidate `.pyc`, no `/tmp/native-stage`, no HTTP(S) proxy environment, no IPv4
route, and only `lo,tunl0` interfaces. Image configuration/history contains no
proxy credential or proxy value.

Candidate production import succeeds. It emits four duplicate-registration
diagnostics while loading the v0.18.1 regular extension and v0.19.0 stable
extension together. The same production import on untouched pinned v0.19.0
exits zero with no duplicate diagnostics. This proves the warning is a known
mixed-build risk. The scoped merge executes PyTorch tensor indexing rather than
a vLLM native operator, so this is recorded rather than hidden.

### Base

Command:

```bash
docker run --rm --network none --gpus device=0 \
  ai-infra-bench/vllm-pr-34246:base \
  bash /workspace/public_dev/run.sh
```

Result (exit 1, expected):

```text
gpu_mask_peak_ratio=5.009
FAIL: CPU mask merge raised ValueError: Error during masked scatter operation
BASE_RC=1
```

This is the old production contract, not an import failure: same-device
`masked_scatter_` executes and exhibits the large temporary allocation, while
the intended CPU mask is rejected.

### Isolated Oracle and Verifier

The exact base-to-head compare diff was downloaded only to the remote validation
directory, never copied into Docker context or image:

```text
base=31a719bcd37a195107711dc8b498288e49ef8576
head=beae1ecede055847be8980528b2b7fc2d9e2fab9
bytes=12885
sha256=aa5d75dbb254f00fdfbdfc75d128abbd1098e53cf6b530ebd1ba7a161990d210
changed_files=9
```

It passed `git apply --check` and was applied only inside an ephemeral offline
container. The unchanged public Verifier then produced exit 0:

```text
gpu_mask_peak_ratio=0.504
cpu_mask_peak_ratio=0.504
PASS: CPU mask merge is correct, asynchronous, bounded, and strict
ORACLE_RC=0
```

The Oracle therefore establishes both sides of the behavior boundary: direct
indexed assignment avoids the old target-sized copy, accepts a CPU mask without
detected synchronization, preserves positions, and precisely rejects excess
multimodal embeddings.

## Remaining risks

- The synthetic primitive test does not reproduce full 235B serving, TP/DCP/EP,
  or end-to-end peak VRAM; those remain external upstream evidence.
- Native artifacts are from the closest pre-cutoff same-Torch release rather
  than an exact SHA build. The one necessary post-cutoff stable-libtorch donor
  object creates duplicate-registration diagnostics with Base's regular
  extension. Imports and the scoped Oracle pass, but unrelated native operators
  are not certified by this environment.

## Survey-manual feedback

- A huge-model OOM may be minimized to the causal framework primitive only when
  upstream tests and issue experiments independently establish that boundary.
- Memory tasks should record allocator deltas and correctness/synchronization,
  not only catch OOM, because available GPU capacity changes the symptom.
- Merge commits in a feature branch require effective-diff analysis; commit
  count alone does not determine atomicity.
- Keep full-topology evidence separate from the publishable primitive contract.
- When an exact source falls between official releases, first expose missing
  native closure with a source-binding import. If a donor is unavoidable, copy
  only explicit hash-locked ELF paths, prove future Python is invisible, and
  report mixed-registration warnings even when the target path is unaffected.
- An Oracle must apply solved code outside Agent assets and rerun the unchanged
  behavioral Verifier; a Base-only failure cannot distinguish a valid contract
  from an impossible test.
