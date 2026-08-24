# Docker build and hardware-eligibility validation

## Status

`HARDWARE-BLOCKED` on the assigned A100. This task is not environment-ready
and is not verifier-ready. The image described below is an eligibility and
source-structure probe only; it must not be released as a runnable benchmark.

## Target

- Remote host: `bm-baai-dx-zone1-d-a100-40g-2-106`
- Requested physical GPU: index 2, NVIDIA A100-SXM4-40GB
- Work directory: `/data/ai-infra-bench/survey-builds/vllm-pr-40408`
- Image tag: `ai-infra-bench/vllm-pr-40408:base`
- Docker endpoint: `unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock`
- Docker data root: `/data/yaoyaoyao/pr34183-cuda-build/docker-data`

All Docker commands explicitly export the isolated endpoint. Runtime commands
use `--network none`.

## Eligibility result

The upstream test utility at the exact base says FP8 requires Ada or Hopper,
compute capability 8.9 or newer. The PR's native batch-invariant dispatch
changes cover SM89, SM90, SM100 and SM120; there is no corresponding SM80 FP8
path. The real A100 probe reported:

```json
{"compute_capability": [8, 0], "container_device_index": 0, "cuda_available": true, "hardware_blocked": true, "minimum_required_compute_capability": [8, 9], "name": "NVIDIA A100-SXM4-40GB", "torch": "2.11.0+cu130", "torch_cuda": "13.0", "uuid": "3815a178-ad22-4b81-5669-0533760a7e6b", "vllm_supports_fp8": false}
```

Host physical GPU index 2 reported UUID
`GPU-3815a178-ad22-4b81-5669-0533760a7e6b`, matching the container UUID.
The container renumbers the sole visible device to index 0.

The PR's determinism test has bf16 input/output, but it forces an FP8 Cutlass
linear kernel. A bf16 smoke test on A100 would therefore not reproduce the
target path and is intentionally omitted.

## Source and base image acquisition

The exact source archive measured:

```text
base commit: ea0e501bb18c12b80acc05ff8c7f013db515ba80
Git tree: f3a7898db0c5cf56d04ae58070633342bfae3856
archive size: 34314430 bytes
archive sha256: 28f46bea58c2a2ac64ea5beaddc58c26a9f3fe05121e65afdbd125a4e65ce665
```

The digest-pinned v0.20.0 linux/amd64 image was absent from the isolated daemon.
Cold pull was kept separate from build time:

```text
digest: sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc
cold pull elapsed: 1550.22 seconds (25 minutes 50.22 seconds)
image inspect Size: 8138369228 bytes
vLLM: 0.20.0
PyTorch: 2.11.0+cu130
CUDA runtime/toolkit: 13.0 / nvcc 13.0
```

The release image contains `nvcc`, GCC and Ninja but not CMake or Git. Since
this PR changes native C++/CUDA, copying the release `.so` files into the base
commit source would be a false provenance claim. The probe image deliberately
keeps exact source and release extensions separate.

## Probe-image build

The already verified archive is served by a temporary HTTP server bound only
to A100 loopback. The Dockerfile verifies the same SHA-256 again. No proxy
credential is used as a build argument or stored in a layer.

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
docker build --network host --no-cache \
  --build-arg VLLM_SOURCE_URL=http://127.0.0.1:40408/vllm-codeload.tar.gz \
  -f environment/Dockerfile \
  -t ai-infra-bench/vllm-pr-40408:base \
  environment
```

Build result:

```text
elapsed: 358.29 seconds (5 minutes 58.29 seconds)
image ID: sha256:b8b4d263a06d2ed50cb4a653ef55f2b15f46c7be93897d0aa98b094414cb6919
docker inspect Size: 8216962225 bytes
published registry digest: not available (local validation image only)
image role label: eligibility-and-structural-probe-only
status label: hardware-blocked-a100-sm80
```

The temporary loopback server was checked before the build and stopped after
validation. Build-time structural output was:

```json
{"base_sha": "ea0e501bb18c12b80acc05ff8c7f013db515ba80", "exact_file_hashes": true, "git_metadata_absent": true, "minimum_fp8_compute_capability": [8, 9], "new_regression_test_absent": true, "pre_pr_batch_invariant_dispatch_absent": true, "structural_base_confirmed": true}
```

## Offline runtime provenance

The built image was run with physical GPU 2 and `--network none`:

```json
{"candidate_native_exact_base_bound": false, "candidate_source": "/app", "compute_capability": [8, 0], "container_device_index": 0, "cuda_available": true, "hardware_blocked": true, "minimum_required_compute_capability": [8, 9], "name": "NVIDIA A100-SXM4-40GB", "native_extension": "/usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so", "native_extension_role": "release-image hardware probe only; not candidate code", "network_connect_ex": 101, "python_source_role": "release-image hardware probe only; not candidate code", "release_python_source": "/usr/local/lib/python3.12/dist-packages/vllm/__init__.py", "torch": "2.11.0+cu130", "torch_cuda": "13.0", "uuid": "3815a178-ad22-4b81-5669-0533760a7e6b", "vllm": "0.20.0", "vllm_supports_fp8": false}
```

`connect_ex=101` confirms that the runtime had no route to the public network.
Both release Python and native paths are outside `/app`; this is intentional
and proves the image is not relabeling release extensions as exact candidate
code. The source gate was also rerun in the final image and produced the same
JSON as the build-time gate.

## Structural baseline

Build-time source checks lock five PR-relevant files by SHA-256 and confirm:

- the exact source has no Git metadata;
- upstream's FP8 test gate requires compute capability 8.9+;
- the pre-PR source lacks `test_cutlass_batch_invariance.py`;
- the pre-PR SM89/SM90 dispatch sources lack the new batch-invariant branch.

This is stable structural evidence, not an executable reproduction of output
invariance. A genuine baseline requires an eligible Ada/Hopper GPU and an exact
native build of the base commit.

## Required follow-up

Move this task to an FP8-capable worker (minimum SM89; SM90/H100 preferred),
build the base commit's native extensions from source with CUDA 13.0.2 and
PyTorch 2.11.0, prove `vllm._C` resolves from the candidate tree, and run the
real Cutlass batch-invariance regression repeatedly under `--network none`.

## Construction-guide feedback

Hardware eligibility must be checked before large image, model, or dataset
downloads. For native tasks, derive the minimum architecture from both upstream
test markers and changed dispatch files, then compare it with the exact worker
GPU capability. Cross-compilation capability and runtime capability must be
recorded separately. Mixed bf16 inputs/outputs do not make an FP8 kernel test
A100-compatible. When hardware is ineligible, publish a truthful blocked audit,
not a substitute smoke test or a release extension relabeled as candidate code.
