# Docker build and baseline validation

## Scope assessment

The upstream change is not atomic enough to publish as a single ordinary
bug-fix task: it comprises 64 commits, modifies 16 files, and adds about 2,152
lines while spanning the public asynchronous API, request lifecycle/state,
scheduler, output processing, GPU runner, and tests. The environment is
usable, and its public Dev targets the smallest coherent end-to-end contract,
but the benchmark must retain a `project-scale / needs-scoping` risk. A safer
publication plan would split API plumbing from scheduler/session execution.

The minimal public entry point is `AsyncLLM.generate` receiving an async
generator of `StreamingInput` chunks. No external checkpoint is required: a
deterministic tiny Llama checkpoint and tokenizer are generated at build time.
The base fails before model initialization because the public API is absent;
after a solution supplies the API, the same Dev proceeds through real engine
execution using the local model.

## Docker daemon and host

All Docker commands used the dedicated isolated daemon:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
```

Its data root is `/data/yaoyaoyao/pr34183-cuda-build/docker-data`. The default
daemon was never used. No image pruning or deletion was performed.

Validation host: NVIDIA A100-SXM4-40GB GPU 0, driver 580.126.20. Validation
date: 2026-08-25 (Asia/Shanghai). Runtime commands used `--network none`.

## Source and image selection

The survey base is `0118cdcc02ae16a137645e2289bf41f5e3da9d80`, committed
after v0.14.0 and before v0.14.1. The final image therefore remains based on
the digest-pinned official v0.14.0 image. The exact source is downloaded from
the commit-specific codeload endpoint and checked against SHA-256
`75b2632ec1ea5f92539b9c5f6a7e3cd3357874f04cfd6953c9ae851f1b992957`.

The first build used depth-one Git fetch. Apt succeeded, but GitHub disconnected
after a long transfer:

```text
error: RPC failed; curl 56 GnuTLS recv error (-54)
fatal: early EOF
fatal: fetch-pack: invalid index-pack output
real 796.55
```

The codeload archive is only 18.9 MiB, naturally has no upstream `.git`, and
supports bounded retries. The build extracts it, creates one synthetic commit,
and asserts one revision, no remote, and a clean worktree.

## Native ABI discovery and anti-leak decision

The unmodified official v0.14.0 image is internally healthy:

```text
official_vllm 0.14.0 /usr/local/lib/python3.12/dist-packages/vllm/__init__.py
official_C /usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so
official_custom_ops /usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py
cuda True tensor([0.], device='cuda:0')
custom_ops_import=PASS
```

After overlaying the Jan-23 candidate Python source on the v0.14.0 native
extension, importing candidate `_custom_ops.py` failed:

```text
RuntimeError: operator _C::marlin_gemm does not exist
```

This is an overlay binding mismatch, not an upstream v0.14.0 packaging defect.
It also blocks real engine use: initializing the local Llama follows the model
registry into candidate `_custom_ops.py` and fails at the same registration.

The adjacent v0.14.1 image was pulled and tested first. Pull time was
`1867.52s`, repository digest/local image ID was
`sha256:6bf34e50e2387dc46dc87a9d6a945fdd616a022bccfddd949052f54063ebcb8c`,
and the image used Torch 2.9.1+cu129. Its own `_custom_ops` was healthy, but its
native extension still did not define the operator expected by the Jan-23
candidate. A v0.14.1 native-only build therefore failed the same strict probe
and was rejected.

The already-cached v0.15.1 native extension explicitly exposed
`torch.ops._C.marlin_gemm`. The final Dockerfile uses v0.15.1 only as a
multi-stage donor. The donor stage copies exactly seven files matching
`*.so` and asserts that no other file type exists. The final stage starts from
v0.14.0, copies those native files into `/workspace/repo/vllm`, and removes the
staging directory. No v0.15.1 Python file or site-packages tree is copied.

Anti-leak evidence from the final image:

```text
candidate_vllm /workspace/repo/vllm/__init__.py
native_C /workspace/repo/vllm/_C.abi3.so
custom_ops /workspace/repo/vllm/_custom_ops.py
native_staging_exists False
underlay_vllm 0.14.0 /usr/local/lib/python3.12/dist-packages/vllm/__init__.py
```

The underlay result was obtained with `PYTHONPATH` cleared and the workdir set
to `/tmp`, proving the final lower layer is v0.14.0 rather than donor Python.
The seven candidate-tree native files were `_C`, `_moe_C`, `cumem_allocator`,
two FlashMLA modules, and the FA2/FA3 modules.

This donor is a post-cutoff ABI approximation, not an exact source build. It
is a material publication risk and must remain visible. Exact compilation of
the base SHA's native extensions would be the stronger but substantially more
expensive alternative.

Dependency boundary: the public baseline itself exits at the missing
`StreamingInput` import and does not load any native operator. A correct
streaming implementation continues into real AsyncLLM/Llama execution, whose
model-registration path unconditionally imports `_custom_ops`; therefore ABI
completeness is required even though this tiny unquantized model does not
actually execute the Marlin GEMM kernel.

## Build

The successful build form was:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
source /data/akg_kernel_bench_lite/A100_proxy.sh
cd /data/ai-infra-bench/survey-builds/vllm-pr-28973
/usr/bin/time -p docker build \
  --network host \
  --pull=false \
  --build-arg HTTP_PROXY \
  --build-arg HTTPS_PROXY \
  --build-arg NO_PROXY \
  -t ai-infra-bench/vllm-pr-28973:base \
  -f context/environment/Dockerfile \
  context
```

Build network is used only for apt and the exact source archive. Runtime is
offline. Proxy arguments are Docker's predefined proxy args and are not
declared in the Dockerfile or retained in image config/history.

Relevant build iterations:

- v0.14.0 native overlay/codeload build: `real 455.78s`.
- rejected v0.14.1 native-only build: `real 670.54s`.
- v0.15.1 native-only build: `real 665.96s`.
- final cached rebuild after raising tiny-model head dimension: `real 252.63s`.
- Final image ID:
  `sha256:5989a6dfb0acf49b27ff233d62a159d674c33c5b374c3cc3a5e942902bd389a8`.
- Final inspect size: `10,622,871,229` bytes.
- Displayed local size: `34.6GB`.
- Platform: `linux/amd64`.
- `docker history --no-trunc` proxy/credential-pattern scan: pass.

## Runtime validation

All final probes used GPU 0 and `--network none`. Candidate/native, GPU, Git,
offline, and isolation checks passed:

```text
candidate_vllm /workspace/repo/vllm/__init__.py
native_C /workspace/repo/vllm/_C.abi3.so
custom_ops /workspace/repo/vllm/_custom_ops.py
versions 0.14.0 2.9.1+cu129 12.9
marlin_gemm _C.marlin_gemm
cuda_available True count 1
gpu NVIDIA A100-SXM4-40GB tensor tensor([0.], device='cuda:0')
git_subject Synthetic benchmark base
git_revisions 1
git_remotes ''
git_status ''
VLLM_TARGET_DEVICE unset
offline 1 1
public_dev_pyc []
runtime_route_rows=0
```

The deterministically generated tiny Llama then completed actual model loading
and GPU inference offline:

```text
tiny_llama_token_ids [3]
```

The first tiny model used head dimension 8 and reached model execution before
FlexAttention rejected dimensions below 16. The final model uses hidden size
64 and four attention heads (head dimension 16); no test was removed or
weakened.

## Public baseline

Command:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
docker run --rm --network none --gpus 'device=0' \
  ai-infra-bench/vllm-pr-28973:base \
  bash /workspace/public_dev/run.sh
```

Observed exit code: `1` (expected baseline failure).

```text
FAIL: session-based streaming input API is unavailable
```

The Dev does not print internal request/scheduler state or disclose a repair.
It checks only the public streaming-session behavior. A solved-tree pass was
not available during environment construction, so the precise post-solution
API compatibility remains a residual validation risk.

## Manual feedback

- Project-scale PRs need a mandatory atomicity gate before environment work;
  a working container does not make a 64-commit cross-layer task publishable.
- Native checks must compare the untouched official image with the candidate
  overlay before classifying a failure as upstream packaging versus binding.
- A future native donor must be digest-pinned, multi-stage, file-type
  whitelisted, stripped of Python source, and labeled as a cutoff/ABI risk.
- Validation must distinguish what the baseline executes from what a repaired
  end-to-end path needs; global import completeness is not proof that a kernel
  is exercised by the target workload.
- Tiny offline models should satisfy backend minimum head dimensions and must
  complete a real token-generation smoke test, not only config/model loading.
