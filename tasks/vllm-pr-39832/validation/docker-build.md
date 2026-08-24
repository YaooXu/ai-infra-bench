# Docker build and baseline validation

Status: environment-ready for the constructor-contract scope. The real base
image reproduces the old compatibility behavior. A solved-tree pass was not
available in this survey run.

## Atomicity and contract

The PR has one coherent contract: remove pre-v0.12 two-argument external KV
connector construction and require `KVCacheConfig` as the third argument. Its
18 commits and 22 changed files are broader than an ideal atomic patch, but
275 of 341 deletions remove the dedicated compatibility test and most other
changes mechanically update constructor call sites. The task is publishable
only when scoped to this constructor contract; unrelated connector behavior
must remain outside the oracle.

The behavioral boundary is:

- A current three-argument connector is constructed once and receives the
  exact `KVCacheConfig` supplied by the factory.
- An external legacy two-argument connector is rejected before construction.
- Direct legacy use of `KVConnectorBase_V1` without the third argument fails.

The public Dev exercises actual constructors and does not search source text.

This PR contains 18 commits, changes 22 files, and has `+119/-341`. It is not
commit-atomic in the strict sense. It is behavior-atomic only under the narrow
factory/base constructor contract above; a benchmark must not claim coverage
of the connector implementations that were mechanically migrated in the same
PR.

## Docker daemon

All Docker commands use:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
```

The daemon data root is
`/data/yaoyaoyao/pr34183-cuda-build/docker-data`. The default daemon is not
used, and no pruning or deletion is performed.

The daemon inherited `/data/akg_kernel_bench_lite/A100_proxy.sh` for build-time
networking. Every run below uses `--network none`. The final image config has
zero `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY` variables, and its history has
no proxy value or credential. Public upstream URLs inherited from the official
base image and literal `Acquire::*::Proxy=false` do appear in history.

## Official release image selection

The exact base commit is dated `2026-05-09T21:46:52Z`. v0.20.1 was published
on 2026-05-04 and v0.20.2 on 2026-05-10, so v0.20.1 is the latest official
release before the source cutoff. It was pulled by digest and probed before any
source overlay:

```text
image/digest sha256:9eff9734a30b6713a8566217d36f8277630fd2d31cec7f0a0292835901a23aa4
size 8230603218 bytes
versions 0.20.1 2.11.0+cu130 13.0
python_source /usr/local/lib/python3.12/site-packages/vllm/__init__.py
native_C /usr/local/lib/python3.12/site-packages/vllm/_C.abi3.so
custom_ops /usr/local/lib/python3.12/site-packages/vllm/_custom_ops.py
cuda True 1
gpu NVIDIA A100-SXM4-40GB tensor([0.], device='cuda:0')
```

The official image's native import and allocation pass establishes that the
same-release `.so` files are viable before binding them to the exact candidate
Python tree.

## Build

Remote context:
`/data/ai-infra-bench/survey-builds/vllm-pr-39832/context`

Command:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
source /data/akg_kernel_bench_lite/A100_proxy.sh
cd /data/ai-infra-bench/survey-builds/vllm-pr-39832
/usr/bin/time -p docker build \
  --network host \
  --pull=false \
  --build-arg HTTP_PROXY \
  --build-arg HTTPS_PROXY \
  --build-arg NO_PROXY \
  -t ai-infra-bench/vllm-pr-39832:base \
  -f context/environment/Dockerfile context
```

Result:

```text
Successfully built 116c9220c7c9
Successfully tagged ai-infra-bench/vllm-pr-39832:base
real 407.37
user 0.06
sys 0.03
```

- Image ID:
  `sha256:116c9220c7c93305d8d156137fdbcc87e41818273d5089308d2dfb768443fb66`
- Image size: `8,862,491,446` bytes
- Runtime user: `agent` (UID 1000)
- Upstream archive SHA-256 check: passed
- Synthetic Git: one commit (`60216caa535944ead2f8dc9aea638dd9392d1278`),
  branch `benchmark-base`, no remote, clean worktree

## Public baseline

The public workload is deliberately CPU-only. It imports and executes the real
factory and base constructor lifecycle, but needs no model, tokenizer, or GPU.

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
docker run --rm --network none \
  ai-infra-bench/vllm-pr-39832:base \
  bash /workspace/public_dev/run.sh
```

Exit status: `1` (expected baseline failure).

```text
INFO ... Creating v1 connector with name: CurrentConnector ...
WARNING ... Connector LegacyConnector uses deprecated signature with 2 required arguments ...
INFO ... Creating v1 connector with name: LegacyConnector ...
WARNING ... KVConnectorBase_V1 initialized without kv_cache_config ...
FAIL: legacy constructor remained accepted (factory_rejected=False, base_rejected=False, constructed=2)
```

This is the intended behavioral symptom: the current constructor succeeds and
preserves the exact supplied KV-cache configuration, while both the factory
compat path and direct base compat path still accept the legacy signature.
The target boundary is the inverse: current construction still passes, legacy
factory construction is rejected before instantiation, and direct omission of
the third base argument raises `TypeError`.

## GPU, native, source-binding, and offline probes

The integrity probe used GPU 0 even though the constructor workload itself is
CPU-only:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
docker run --rm --network none --gpus device=0 \
  ai-infra-bench/vllm-pr-39832:base python3 -c '<import/path/CUDA assertions>'
```

Observed:

```text
uid 1000
versions 0.20.1 2.11.0+cu130 13.0
source /workspace/repo/vllm/__init__.py
native /workspace/repo/vllm/_C.abi3.so
custom_ops /workspace/repo/vllm/_custom_ops.py
factory /workspace/repo/vllm/distributed/kv_transfer/kv_connector/factory.py
base /workspace/repo/vllm/distributed/kv_transfer/kv_connector/v1/base.py
target_device None
offline 1 1
cuda 1 NVIDIA A100-SXM4-40GB tensor([0.], device='cuda:0')
```

An independent no-network/sanitization audit observed:

```text
git_count 1
git_remote_rows 0
git_status_rows 0
route_rows 0
pyc_count 0
```

The candidate tree contains only the same-release native/generated whitelist
recorded in `environment/lock/README.md`. No future Python package directory is
added to `PYTHONPATH`; Python and native paths both resolve under
`/workspace/repo`.

## Remaining risks

- No solved-tree image was built, so the expected target pass is defined by the
  explicit two-sided oracle rather than demonstrated here.
- The PR is multi-commit and mechanically broad. The public Dev intentionally
  covers only external connector factory/base construction, not every migrated
  connector implementation.
- Native extensions come from the nearest pre-cutoff official release rather
  than a compilation of the exact SHA. This is a same-version ABI
  approximation; it passed real import and GPU allocation, and the CPU-only
  constructor reproduction does not execute native kernels.

## Survey-manual feedback

- For deprecation-removal tasks, require a dual oracle: the old signature must
  be rejected before construction and the new signature must still succeed
  with argument identity preserved.
- Reject source-string-only tests when a lightweight real constructor lifecycle
  can express the contract.
- Select release donors from the actual base commit time, not the PR creation
  time, and state the source-cutoff relationship explicitly.
- Record CPU-only behavioral workloads separately from mandatory GPU/native
  environment probes.
- A mechanically broad PR may be publishable as contract-atomic only when the
  benchmark's narrower scope and untested call sites are explicit.
