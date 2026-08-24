# Docker build and validation

Status: **environment-ready for focused rank-local GPU behavior**. The full
four-GPU model verifier remains unavailable in this task environment.

## Atomicity and solution mapping

PR 34179 is behavior-atomic: 3 commits, 5 files, `+117/-3`. It adds DCP-aware
local sequence lengths, virtual/local block translation in the production
Triton slot-mapping kernel, persistent model-runner buffers, metadata wiring,
and CUDA-graph preparation. Review requested deduplicating the local-sequence
logic; the final patch implements that shared helper in `attn_utils.py`.

The focused verifier covers actual DCP slot mapping and CUDA graph replay on
one A100 while emulating only the rank metadata. Full multi-rank collectives,
model accuracy, and performance remain outside this environment.

## Docker daemon

All Docker commands use:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
```

Its data root is `/data/yaoyaoyao/pr34183-cuda-build/docker-data`; runtime uses
`--network none`. No pruning or deletion is performed.

## Build and validation evidence

Remote working directory:

```text
/data/ai-infra-bench/survey-builds/vllm-pr-34179
```

The locked source archives were served only over A100 loopback during build.
The representative build command was:

```bash
source /data/akg_kernel_bench_lite/A100_proxy.sh
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
docker build --network host --pull=false \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  --build-arg VLLM_SOURCE_URL=http://127.0.0.1:18084/source/vllm-base.tar.gz \
  -t ai-infra-bench/vllm-pr-34179:base \
  -f context/environment/Dockerfile context
```

- Cold pull of the selected `v0.17.0` amd64 digest: `390.63 s`.
- First full build: `530.94 s`; its runtime probe exposed that GitHub's
  source archive contains only `.gitkeep` in `vllm/vllm_flash_attn`.
- Full rebuild after explicitly locking the two required donor Python shims:
  `385.91 s`.
- Final cached rebuild after correcting the public CUDA-graph boundary:
  `159.75 s`.
- Final image:
  `sha256:f15cf8b9a654e2c8030c6e1d03a6d0d06a3fc7e84764ba4e4b54128916d9f92b`.
- Final image size: `10,264,014,475` bytes.

The initial Oracle probe showed that `prepare_dcp_local_seq_lens` creates its
rank-offset tensor and therefore cannot itself execute inside CUDA stream
capture. The public Dev was corrected to match the production boundary:
prepare DCP local sequence metadata outside capture, while capturing and
replaying the production Triton slot-mapping kernel.

### Base result

Executed on physical GPU 2 with `--network none`:

```text
FAIL: base GPU model runner v2 has no DCP local-sequence helper; the real CUDA slot-mapping contract cannot be prepared
DETAIL: cannot import name 'prepare_dcp_local_seq_lens' from 'vllm.v1.worker.gpu.attn_utils' (/workspace/repo/vllm/v1/worker/gpu/attn_utils.py)
RC=1
```

This is a behavior failure at the intended missing production API, not a
source-string assertion or an environment import failure.

### Ephemeral Oracle result

Only the PR head versions of `attn_utils.py` and `block_table.py` were mounted
read-only for this check. Neither is present in the Agent image:

```text
PASS: exact DCP rank-1 virtual-block/interleave mapping ran through the production Triton kernel and replayed from a CUDA graph
gpu=NVIDIA A100-SXM4-40GB capability=8.0 uuid=3815a178-ad22-4b81-5669-0533760a7e6b slots=[47, 46, -1, -1, 45, 44, -1, -1, 43, 42, -1, -1, 41, 40, -1, -1]
RC=0
```

The verifier uses real `BlockTables`, CUDA tensors, the production Triton
kernel, DCP world size 2/rank 1 metadata, virtual-block interleaving, PAD slot
semantics, and CUDA graph replay. It does not claim NCCL or model-level
coverage.

### Integrity probes

- Runtime user: `uid=1000(agent)`, with a successful create/delete probe in
  `/workspace`.
- Candidate source: `/workspace/repo/vllm/__init__.py`.
- Candidate native binding: `/workspace/repo/vllm/_C.abi3.so`.
- Runtime: Torch `2.10.0+cu129`, CUDA `12.9`; `VLLM_TARGET_DEVICE` is absent.
- GPU 2: `NVIDIA A100-SXM4-40GB`, capability `8.0`, UUID
  `3815a178-ad22-4b81-5669-0533760a7e6b`; CUDA tensor operation passed.
- Git: branch `benchmark-base`, one commit, zero remotes, clean status, tree
  `fa64310667f4c1849399eedea2e4e05c57936453`.
- Runtime `--network none`: `/proc/net/route` data rows `0`.
- Image history proxy credential/address matches: `0`; final image config has
  no HTTP/HTTPS proxy variables.
- Docker build verified exactly eight native artifacts and exactly two donor
  Python shims against `native-donor.json`; target/native intersection is
  empty.

## State separation

- Agent image: exact-base Python/Triton tree plus scoped donor runtime; no PR
  head files or solved code included.
- Base: deterministically fails because model-runner-v2 lacks the DCP local
  sequence helper.
- Oracle: passes only through ephemeral read-only mounts of two PR head files.
- Full verifier: requires four GPUs, model assets, TP4/DCP4, CUDA graph,
  accuracy, and paired performance.

## Survey-manual feedback

- For distributed features, separate rank-local GPU behavior from collective
  and end-to-end topology claims.
- A single-GPU test may emulate rank metadata only when it still executes the
  production GPU kernel and explicitly leaves collectives unclaimed.
- Reject a native donor whose Torch ABI conflicts with the exact source, even
  if it is the latest pre-cutoff release.
- Record copied native artifacts and their intersection with target files in a
  machine-readable manifest, especially when the donor is post-cutoff.
- A GitHub source archive may omit generated Python wrappers that accompany
  native extensions. Candidate-path probes must import the relevant runtime
  module, not merely locate the top-level `_C` binding; any required wrapper
  copied from a donor must be individually locked and included in the donor
  risk manifest.
- CUDA-graph verifiers must mirror production capture boundaries. A helper
  that allocates a tensor during each call should be tested outside capture,
  while the persistent-buffer kernel can still be captured and replayed.
