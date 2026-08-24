# Docker build and baseline validation

## Target

- Remote host: `bm-baai-dx-zone1-d-a100-40g-2-106`
- Physical GPU: index 2, NVIDIA A100-SXM4-40GB
- Work directory: `/data/ai-infra-bench/survey-builds/vllm-pr-30475`
- Image tag: `ai-infra-bench/vllm-pr-30475:base`
- Docker endpoint: `unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock`
- Docker data root: `/data/yaoyaoyao/pr34183-cuda-build/docker-data`

Every Docker command below must set the endpoint explicitly:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
```

## Commands

```bash
docker run --rm --network none --gpus 'device=2' \
  ai-infra-bench/vllm-pr-30475:base

docker run --rm --network none --gpus 'device=2' \
  ai-infra-bench/vllm-pr-30475:base \
  python3 -I /opt/bench/reproduce_baseline.py
```

On the validation host the isolated daemon has no bridge, so the real build
uses `--network host` and an audited loopback mirror of the same archive:

```bash
docker build --network host --no-cache \
  --build-arg VLLM_SOURCE_URL=http://127.0.0.1:30475/vllm-source.tar.gz \
  -f environment/Dockerfile \
  -t ai-infra-bench/vllm-pr-30475:base \
  environment
```

The mirror file must first pass the same
`ac774fb6...30240624` SHA-256 check; the Dockerfile checks it again inside the
build. It carries no credentials and is stopped after the build.

## Results

Validated on 2026-08-25.

### Base-image acquisition

The isolated daemon was explicitly selected for every Docker command. Its
`docker info` reported data root
`/data/yaoyaoyao/pr34183-cuda-build/docker-data` and a daemon-level HTTP/HTTPS
proxy. A shell proxy alone had previously been insufficient for `docker pull`.

The first digest pull was interrupted after approximately 22 minutes 39
seconds while diagnosing shared-proxy throughput. The resumed pull reused all
completed layers and succeeded in 30 minutes 02.26 seconds. Cold acquisition
wait was therefore approximately 52 minutes 41 seconds in this shared,
concurrently loaded environment. It is not included in Dockerfile build time.

### Source mirror used by this host

Because this isolated daemon runs with `--bridge=none`, a build using
`--network default` could not resolve `codeload.github.com`. The host downloaded
the public commit archive through the configured shell proxy and verified:

```text
archive bytes: 17769007
sha256: ac774fb6bbf75b083d997dcb70b5cdf8746c6e6bfa8d95b3c2415a6c30240624
```

A temporary Python HTTP server bound only to `127.0.0.1:30475`; the build used
`--network host` and `VLLM_SOURCE_URL` to read that file. The Dockerfile
rechecked the SHA-256 inside the build. No proxy credential entered the build
arguments, image, source tree, or logs. The server PID and command were checked
before it was stopped after validation.

### Successful build

Final clean command:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
docker build --network host --no-cache \
  --build-arg VLLM_SOURCE_URL=http://127.0.0.1:30475/vllm-source.tar.gz \
  -f environment/Dockerfile \
  -t ai-infra-bench/vllm-pr-30475:base \
  environment
```

Result:

```text
elapsed: 714.50 seconds (11 minutes 54.50 seconds)
image ID: sha256:439026ed115db4ce60b916771cf1205900ce1623d6afb6512201553b67dc32be
docker inspect Size: 15688758352 bytes
docker images virtual size: 49GB
published registry digest: not available (local validation image only)
```

A warm-cache successful build took 297.64 seconds. It is retained as a
diagnostic result, not substituted for the clean timing above.

### Runtime provenance and GPU smoke

Executed with physical GPU index 2 and `--network none`:

```json
{"candidate_source": "/app/vllm/__init__.py", "cuda_runtime": "12.9", "gpu": "NVIDIA A100-SXM4-40GB", "native_extension": "/app/vllm/_C.abi3.so", "torch": "2.9.0+cu129"}
```

The candidate tree also passed these assertions:

- one synthetic Git commit;
- Git tree `aafa39e6544cbeaf83b72985f83aaefd4e9e3456`;
- no remote, tag, or dirty candidate files;
- source and `vllm._C` both resolve under `/app`;
- outbound TCP probe failed with `connect_ex=101`.

### Baseline reproduction

The minimal reproducer uses production vLLM classes, not a reimplementation.
It creates a sparse placeholder with `P=100` decoder positions and `E=8`
embedding rows, then supplies an encoder-cache capacity of 8. The base should
accept it under embedding-unit semantics, but instead rejects it:

```json
{"baseline_bug_reproduced": true, "cache_capacity": 8, "can_allocate": false, "embedding_rows_E": 8, "expected_fixed_semantics": true, "placeholder_positions_P": 100}
```

This is a stable structural reproduction of the PR's resource-accounting bug;
it requires neither a model nor a dataset. It does not by itself validate the
full eventual task verifier (storage, gather/merge, profiling, connector,
preemption, and regression behavior still require task tests).

### Residual risk

Native extensions are copied from the nearest digest-pinned official v0.11.2
release rather than rebuilt from base SHA `676db55e`. The PR's implementation
surface is Python-only and the final image passes source/native import and CUDA
smoke checks, so this is a deliberate simplification for Dockerfile synthesis.
A task that changes native code must instead rebuild extensions from the exact
base source and lock the native build inputs.

### Failed attempts retained for audit

- `251.61s`: build default network could not resolve GitHub because the daemon
  has no bridge.
- `241.13s`: source SHA passed, but synthetic tree omitted three
  upstream-tracked files that also match `.gitignore`; fixed with
  `git add -f -A` and canonical tree assertion.
- `294.95s`: provenance check assumed CUDA 12.8; the pinned official v0.11.2
  image actually supplies PyTorch `2.9.0+cu129` and CUDA 12.9.
- `292.74s`: `.dockerignore` was incorrectly copied as a normal context file;
  Docker consumes it but excludes it from `COPY`, so the unnecessary COPY was
  removed.

Logs and raw outputs are retained under
`/data/ai-infra-bench/survey-builds/vllm-pr-30475/` on the validation host.
