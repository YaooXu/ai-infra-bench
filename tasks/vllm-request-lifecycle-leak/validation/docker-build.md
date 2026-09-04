# Docker build and baseline validation

## 2026-09-03 behavior-only review controls

The current verifier removes the earlier source/AST rejection for
`gc.collect`; it now grades only externally observable lifetime and hashing
behavior with cyclic GC disabled. This avoids prescribing weak references or
any other implementation technique.

Four fresh, offline CPU containers against the locked base produced:

```text
Base:                                      0
Oracle (unbound hasher + update method):   1
Opus-5 scheduler-only cleanup patch:       0
Independent weak-callback object patch:    1
```

The rejected Opus-5 patch only cleared multimodal fields after scheduler
completion, leaving all 11 Requests and payloads retained by the earlier
self-cycle. Its frozen patch SHA-256 is
`5f20c4b63802f0bff4832c644fdc226201d7ab481e08390280fedce10a3e62c1`.
The independent positive control stores a callable object containing a weak
request reference, dereferences it, and passes the real live Request to the
unchanged block hasher. It preserves initial, append-token, and streaming
hashing and has patch SHA-256
`dac2e534c1009026a4dc0311aba6a4ee37cf6bb92f6546fc8afe2fedd2a3220d`.

These controls demonstrate both directions without binding the verifier to
the Oracle's field or helper names.

## v1.1 verifier and Agent revalidation (2026-08-27)

This section supersedes the original image and verifier details below. The
scored behavior is CPU request-lifecycle logic, so version 1.1 uses the locked
`vllm/vllm-openai-cpu:v0.17.1-x86_64` donor and overlays the same exact pre-fix
source. This removes an unrelated CUDA image download and initialization from
every Agent attempt.

The revised verifier does not require `weakref`, a closure name, or any other
Oracle implementation shape. With cyclic GC disabled it checks prompt release
for eleven held-out requests and multimodal payloads. It separately proves
that initial, append-token, and streaming-session block hashing still runs and
that live multimodal data is not discarded. A narrow anti-cheat check rejects
forcing `gc.collect()` from the request module.

- Base: reward `0`; all reproduced requests and payloads remained retained.
- Accepted Oracle: reward `1`.
- Deliberately incomplete solution (incremental hashing disabled): reward `0`.
- Claude Opus 4.8 (`claude-opus-4-8`, Claude Code 2.1.220): reward `1` after
  running `agent-test` and replacing the self-retaining callback with a weak
  reference. The attempt used 51 trajectory turns, approximately 303 seconds
  of model time (347 seconds end to end), and reported cost `$4.63658075`.

The successful attempt is recorded outside the task payload at
`/data0/dxz-ai-infra-eval-opus48-retry/runs/vllm-pr-34183/attempt-1`.

## Docker daemon

All Docker commands use the dedicated isolated daemon:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
```

Its data root is `/data/yaoyaoyao/pr34183-cuda-build/docker-data`. The default
daemon is not used, and no image pruning or deletion is performed.

## Build

Validation host: NVIDIA A100-SXM4-40GB GPU 0, driver 580.126.20. Build date:
2026-08-25 (Asia/Shanghai).

The base was first pulled and inspected through the isolated daemon:

```text
vllm/vllm-openai:v0.15.1
manifest digest: sha256:8c9aaddfa6011b9651d06834d2fb90bdb9ab6ced4b420ec76925024eb12b22d0
linux/amd64 digest: sha256:06f9f0d5c7cb079504615c51dab70cd18abbf609d1358b940172181ac0a92efa
```

The successful full overlay build used:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
source /data/akg_kernel_bench_lite/A100_proxy.sh
cd /data/ai-infra-bench/survey-builds/vllm-pr-34183
/usr/bin/time -p docker build \
  --network host \
  --pull=false \
  --build-arg HTTP_PROXY \
  --build-arg HTTPS_PROXY \
  --build-arg NO_PROXY \
  -t ai-infra-bench/vllm-pr-34183:base \
  -f context/environment/Dockerfile \
  context
```

The build network is intentionally distinct from runtime network. The
isolated dockerd has no bridge, so `--network host` is required only while
installing build dependencies and fetching the exact Git commit. Proxy values
are inherited from the named script and are not embedded in this document or
the image history.

Build evidence:

- Full successful overlay build: `real 739.55s`.
- Final cached rebuild after adding the host-driver library preference:
  `real 92.68s`.
- Final image ID:
  `sha256:d6e5c71a38099fd430cbed0efcbd8a331fb014ab7d6bcc940e34e44d70436d7c`.
- Final image size from `docker image inspect`: `10,276,009,661` bytes.
- Displayed local size from `docker images`: `33.4GB`.
- Platform: `linux/amd64`.
- `docker history --no-trunc` secret scan: no proxy host, username, or
  password was present.

Two environment-discovery failures were retained rather than hidden:

1. `--progress=plain` is unsupported by the host's legacy builder; the command
   exited before build in `0.03s`.
2. The first default-network build had no route because this dedicated daemon
   uses `--bridge=none` (`224.57s`). A host-network retry reached the supplied
   proxy, but the proxy returned 500/502 for apt repositories (`7.98s`). The
   final Dockerfile lets apt directly reach its configured repositories while
   GitHub continues to use the supplied build proxy.

During the successful build, the log showed and asserted:

```text
HEAD is now at e94ec59 [LMCache] Token Base IPC API (#34175)
e94ec597334d9a3e9b0d04bc17152e2747c83d51 =
e94ec597334d9a3e9b0d04bc17152e2747c83d51
remote list: empty
revision count: 1
working tree: clean
```

## Runtime validation

All runtime checks used the final image, GPU 0, and `--network none`:

```bash
docker run --rm --network none --gpus 'device=0' \
  ai-infra-bench/vllm-pr-34183:base ...
```

Candidate/source, native ABI, GPU, Git, and isolation checks passed:

```text
candidate_vllm=/workspace/repo/vllm/__init__.py
native_C=/workspace/repo/vllm/_C.abi3.so
custom_ops=/workspace/repo/vllm/_custom_ops.py
cuda_available=True gpu=NVIDIA A100-SXM4-40GB tensor=tensor([0.], device='cuda:0')
git_commit=e949ea65ff694d19fc0f095a1a7e42d7aba90261 subject=Synthetic benchmark base
VLLM_TARGET_DEVICE=unset
runtime_default_route=no
```

The native probe matters because the exact source base already pins PyTorch
2.10 while the latest pre-fix release image contains PyTorch 2.9.1. Imports of
`vllm._C` and `_custom_ops` both succeeded with the candidate worktree taking
precedence. No later official vLLM image is used because post-fix installed
Python source would be solution leakage.

### CUDA 803 evidence and resolution

Before the final `LD_LIBRARY_PATH` setting, the official image selected
`/usr/local/cuda-12.9/compat/libcuda.so.575.57.08` ahead of the toolkit-injected
host library. On driver 580.126.20 the real Torch probe produced:

```text
count=1 available=False
CUDA initialization ... Error 803: system has unsupported display driver /
cuda driver combination
```

Giving `/lib/x86_64-linux-gnu` priority selected the host-injected driver. The
final image then reported CUDA available, named the A100, and allocated a CUDA
tensor as shown above.

### Baseline reproduction

Command:

```bash
docker run --rm --network none --gpus 'device=0' \
  ai-infra-bench/vllm-pr-34183:base \
  bash /workspace/public_dev/run.sh
```

Observed exit code: `1` (expected baseline failure).

```text
candidate source: /workspace/repo/vllm/__init__.py
completed requests still retained: 16/16
multimodal payloads still retained: 16/16
FAIL: completed request state was not released promptly
```

The workload is local and deterministic. It does not fetch a model or dataset,
does not force global collection before measurement, and does not inspect or
print the retaining reference chain.
