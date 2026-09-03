# Docker build and baseline validation

> This task is **CPU** (`environment_profile = "cpu"`, `gpus = 0`,
> `accelerator = "CPU"`). No GPU is needed to build or to reproduce the
> baseline. Any GPU-specific commands in older revisions of this file were
> incorrect for this task and have been removed.

## Target

- Image tag (versioned): `ai-infra-bench/vllm-pr-30475:v1.2.1-20260903T161811Z`
- Canonical candidate tag:
  `ghcr.io/ouycc/ai-infra-bench-task-envs:vllm-pr-30475-ea49c23e150d3c530749d845b4ad298694b05a2788648d54d921082a27d950c3`
  where the suffix is the `<env-key>` from
  `task_ci.py env-key --task vllm-pr-30475 --platform linux/amd64`.
- Image ID: `sha256:3136c80d465ea0b7ad8f5c05238892e2ce1709ab2193a7b1ccee64ac9ddeccc9`
- Base image: `vllm/vllm-openai@sha256:47a9896f86818fea323b2d38082758c62d9a0155d6fe6c4dbd7d735c556f680a`

Set the Docker endpoint explicitly for every command (the build host uses an
isolated daemon):

```bash
export DOCKER_HOST=unix:///data/docker-bench/docker.sock
export DOCKER_CONFIG=/data/yinchen/docker-config
```

## Commands

The baseline reproducer and runtime smoke script are **not** baked into the
image. The evaluator mounts them read-only at run time; the image only carries
`public_tests/` under `/opt/bench`. Reproduce the baseline like this:

```bash
docker run --rm --network none \
  -v "$PWD/environment/reproduce_baseline.py:/tmp/reproduce_baseline.py:ro" \
  ai-infra-bench/vllm-pr-30475:v1.2.1-20260903T161811Z \
  python3 -I /tmp/reproduce_baseline.py
```

## Build

The `environment/` Dockerfile builds the image. Its first stage fetches the
exact Base commit and its real parent history from GitHub, so the build needs
network egress that can pull a full git history. On this host `github.com` is
not reachable directly, so the build uses `--network host` plus an egress proxy
passed only as predefined build-args (`http_proxy`/`https_proxy`). Docker does
not persist those build-args into the image or `docker history`, and no
credential is written to the image, git tree, or evidence.

```bash
docker build --network host --no-cache \
  --build-arg http_proxy="$EGRESS_PROXY" \
  --build-arg https_proxy="$EGRESS_PROXY" \
  --build-arg HTTP_PROXY="$EGRESS_PROXY" \
  --build-arg HTTPS_PROXY="$EGRESS_PROXY" \
  --build-arg no_proxy=localhost,127.0.0.1 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  -f environment/Dockerfile \
  -t ai-infra-bench/vllm-pr-30475:v1.2.1-20260903T161811Z \
  environment
```

The pytest wheels and the vLLM source archive are SHA-256-checked inside the
Dockerfile (`ac774fb6...30240624` for the source tarball). The source archive
is pulled from `codeload.github.com`, which is reachable directly; only the
git-history stage requires the proxy.

## Results

Rebuilt and validated on 2026-09-03 (task v1.2.1, CPU).

### Build result

```text
elapsed: 342 seconds (--no-cache, CPU)
image ID: sha256:3136c80d465ea0b7ad8f5c05238892e2ce1709ab2193a7b1ccee64ac9ddeccc9
docker inspect Size: 15924285519 bytes
base image: vllm/vllm-openai@sha256:47a9896f86818fea323b2d38082758c62d9a0155d6fe6c4dbd7d735c556f680a
```

### Git provenance (verified in-image by `image-check`)

The `environment/Dockerfile` fetches the exact Base commit and its real parent
history from GitHub (there is **no** synthetic single commit). `image-check`
confirmed inside the image:

- `HEAD` == Base commit `676db55eecf8b6d9ec38ea243cf6f35ea8378ec6`;
- Git tree `aafa39e6544cbeaf83b72985f83aaefd4e9e3456`;
- real parent history (`git rev-list --count HEAD` = 12285, > 1);
- no remotes, no tags, no reflog, no `FETCH_HEAD`/`ORIG_HEAD`/`shallow`;
- working tree clean; `git fsck` clean; Oracle/future commit
  `f5f51e5931ffd99afe69696b60765b88d3eb13f2` absent;
- no `/tests`, `/solution`, `/validation` in the image;
- `vllm` and `vllm._C` both resolve under `/app`.

### Baseline reproduction

The minimal reproducer uses production vLLM classes, not a reimplementation.
It creates a sparse placeholder with `P=100` decoder positions and `E=8`
embedding rows, then supplies an encoder-cache capacity of 8. The base should
accept it under embedding-unit semantics, but instead rejects it. Run with the
reproducer mounted read-only (it is not baked into the image):

```json
{"baseline_bug_reproduced": true, "cache_capacity": 8, "can_allocate": false, "embedding_rows_E": 8, "expected_fixed_semantics": true, "placeholder_positions_P": 100}
```

This is a stable structural reproduction of the PR's resource-accounting bug;
it requires neither a model nor a dataset. It does not by itself validate the
full task verifier (the Harbor control cases and stability rounds do that).

### Residual risk

Native extensions (`*.so`, `_version.py`) are overlaid from the nearest
digest-pinned official v0.11.2 release rather than rebuilt from base SHA
`676db55e`. The PR's implementation surface is Python-only and the final image
passes source/native import checks, so this is a deliberate simplification for
Dockerfile synthesis. A task that changes native code must instead rebuild
extensions from the exact base source and lock the native build inputs.

### Network note

`github.com` (git smart-HTTP) is not reachable directly from the build host, so
the git-history stage requires an egress proxy (passed only as predefined
build-args, never persisted in the image, git tree, or evidence). The vLLM
source archive and pytest wheels come from `codeload.github.com` /
`files.pythonhosted.org`, which are reachable directly and SHA-256-checked
inside the Dockerfile.
