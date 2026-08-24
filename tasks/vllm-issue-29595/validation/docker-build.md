# Docker build and A100 validation

## Outcome

- Environment status: **built and runtime-validated**.
- Original symptom status: **not positively reproduced on A100**.
- Release readiness: **reproduction-blocked / solution-mapping-blocked**.

The final environment was subsequently hardened to run as non-root and expose
a clean one-commit synthetic Git repository. That rebuilt image passed
source/native/GPU/permissions/Git smoke. Per audit scope, the large model
inference was not repeated; the grounding evidence below came from the prior
runtime-equivalent root/no-Git image and is retained as historical pipeline
evidence.

This distinction is important. The final image runs a real Qwen3-VL model,
real image inputs, torch.compile, and CUDA Graphs with no runtime network. It
does not prove that the original 235B/H20 grounding regression can be reduced
to a 4B/A100 workload.

## Host and isolated Docker daemon

- Host: `bm-baai-dx-zone1-d-a100-40g-2-106`
- Runtime GPU: physical GPU 1, NVIDIA A100-SXM4-40GB
- Docker daemon:
  `unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock`
- Docker data root:
  `/data/yaoyaoyao/pr34183-cuda-build/docker-data`
- Remote task directory:
  `/data/ai-infra-bench/survey-builds/vllm-issue-29595`

Every Docker pull/build/run/inspect command used this daemon. No image prune
or removal of another task's image was performed. The probe container was
removed after validation and GPU 1 returned to 40,438 MiB free.

## Immutable inputs

| Input | Value |
|---|---|
| Official base | `vllm/vllm-openai:v0.11.1` |
| Base digest | `sha256:d5b12dfb74d605615f8b29ebafaa52294c118bcac7bc9e941785c4108fdb913a` |
| Base image ID | same as the digest above |
| Git helper | `alpine/git:2.49.1@sha256:c0280cf9572316299b08544065d3bf35db65043d5e3963982ec50647d2746e26` |
| vLLM source commit | `439368496db48d8f992ba8c606a0c0b1eebbfa69` |
| Source archive SHA-256 | `b36c716c28bfd08359e566db92775d3621bcf31a1fd18505f78cb94128a22972` |
| Source archive bytes | 17,209,494 |
| Model | Qwen3-VL-4B-Instruct, 43 files |
| Model bytes | 8,887,294,190 |
| Model directory SHA-256 | `7ca8d99e4466b09409d92de5daaba11c047bf8b363076a37756d89e899e8f141` |

The model directory digest is the SHA-256 of the sorted stream of per-file
SHA-256 records, computed from relative paths inside the model directory.

## Source cache preparation

The first implementation fetched Git inside Docker build. Although the commit
was pinned, the shared proxy produced both connection failures and a later
`GnuTLS ... early EOF`. The final implementation moves this unreliable network
operation outside Docker build and verifies the downloaded archive twice:
once before build and once inside the Dockerfile.

```bash
source /data/akg_kernel_bench_lite/A100_proxy.sh
curl --retry 10 --retry-all-errors --retry-delay 3 -fL \
  -o vllm-source.tar.gz \
  https://codeload.github.com/vllm-project/vllm/tar.gz/439368496db48d8f992ba8c606a0c0b1eebbfa69
sha256sum vllm-source.tar.gz
```

Observed archive digest:

```text
b36c716c28bfd08359e566db92775d3621bcf31a1fd18505f78cb94128a22972
```

The proxy was never placed in the Dockerfile or retained image. Final image
inspection reported `proxy_env_names=[]`.

## Final cold build

Command:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
DOCKER_BUILDKIT=0 docker build \
  --no-cache \
  --network none \
  -t ai-infra-bench/vllm-issue-29595:base \
  .
```

Result:

| Field | Value |
|---|---|
| Exit code | 0 |
| Elapsed time | 575 seconds (9m35s) |
| Image ID | `sha256:841bce5d3b4edaf8ed005c26f46eb30ef1aa9194bef5b26f36d2a968705120c4` |
| Inspect size | 14,452,315,932 bytes |
| Configured user | `agent` (UID/GID 1000) |
| Build network | none |
| Image tag | `ai-infra-bench/vllm-issue-29595:base` |

Complete hardened build log and timing summary on A100:
`/data/ai-infra-bench/survey-builds/vllm-issue-29595/docker-build-nonroot-git.log`
and `docker-build-nonroot-git-summary.txt`. The earlier 315-second log is
retained as `docker-build-cold.log` but no longer identifies the final image.

## Source/runtime binding checks

Checks were executed with GPU 1 and `--network none`:

```text
cuda= True NVIDIA A100-SXM4-40GB
vllm= 0.11.1 /workspace/repo/vllm/__init__.py
native= /workspace/repo/vllm/_C.abi3.so
user= uid=1000(agent) gid=1000(agent)
git= 2.49.1, one commit, zero remotes, clean status
```

The source archive is authoritative. Files that are generated or bundled only
in the official wheel (native extensions, `_version.py`, and packaged
`vllm_flash_attn` wrappers) are copied with `--no-clobber`, so they cannot
replace upstream source files. The build failed during development until this
wheel-only runtime surface was included; a Python source tree plus top-level
`.so` files alone was insufficient to start Qwen3-VL.

The source archive contributes no upstream Git metadata. After the complete
runtime tree is assembled, the Dockerfile creates exactly one synthetic commit
on `benchmark-base`, with no remote. The non-root agent successfully wrote a
new file in `/workspace/repo`, and `git status` reported it as untracked. The
Git helper is digest-pinned and copied in a multi-stage build; no online apt
operation occurs in the final `--network none` build. Final image history again
contained no proxy variable or proxy host match.

## Runtime configuration and optimization probes

The model was mounted read-only, the container used `--network none`, and the
service was started without `--enforce-eager`:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 8000 \
  --model /models/Qwen3-VL-4B-Instruct \
  --served-model-name qwen3-vl \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.80 \
  --generation-config vllm
```

The server log proves the optimized path was active:

```text
enforce_eager=False
mode=<CompilationMode.VLLM_COMPILE: 3>
cudagraph_mode=<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>
torch.compile takes 20.19 s in total
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%
Capturing CUDA graphs (decode, FULL): 100%
Application startup complete.
```

The earlier 4096-token probe independently recorded 61.05 seconds of
torch.compile and the same two successful CUDA Graph capture phases. The final
server log is retained at:
`/data/ai-infra-bench/survey-builds/vllm-issue-29595/vllm-server-8192.log`.

## Real-data grounding probes

### Wikipedia screenshot

- Image bytes: 986,579
- Image SHA-256:
  `b8aa2779b53e5f88e068633fa57c666a08c1da8eefdc32762f03ed3bfad4eb72`
- Prompt: `Locate the button that is used to log in`
- Processed prompt length: 4,470 tokens
- Qwen3-VL-4B result: `[]`

A first service configured with `max_model_len=4096` correctly rejected the
request because the multimodal prompt occupied 4,470 tokens. With 8192 the
request completed in 6.66 seconds but the substituted 4B model did not detect
the small target. This is a model-capability limitation, not evidence of the
original coordinate-shift symptom.

### Smaller sign-in screenshot

- Image bytes: 328,512
- Image SHA-256:
  `06257c37c50086cbf7efe2f8b6fe5264666218e37e09e4016be4dc6f3c3b7c0f`
- Prompt: `Locate the sign in button`
- Processed prompt length: 2,054 tokens
- Reference bbox: `[796, 19, 879, 71]`
- Optimized-path bbox: `[796, 18, 880, 72]`
- Maximum coordinate error: 1px, below the existing 3px threshold
- First request: 0.68 seconds
- Repeat request: 0.39 seconds, identical bbox

This verifies a real image/model/service request and stable CUDA Graph-path
output on A100. It is a **negative reproduction** consistent with the upstream
comment that A100 retained grounding accuracy while Hopper systems degraded.

Raw artifacts are retained in the remote task directory as
`grounding-query.log`, `grounding-details.json`, `signin-query.log`,
`signin-details.json`, `signin-query-repeat.log`, and
`signin-details-repeat.json`.

## Why the original symptom remains blocked

1. The original report used Qwen3-VL-235B-A22B-Instruct(-FP8), TP=8 on eight
   H20 GPUs. The assigned validation resource is one 40GB A100.
2. An upstream reproducer explicitly reported that A100 did not show the
   grounding degradation and H100 did.
3. The survey row already classifies this item as `needs_solution_mapping`
   with blockers `issue_has_no_surveyed_fix_pr` and
   `requires_large_multigpu_environment`.
4. The issue was closed by vLLM PR #30525, a PyTorch 2.10 dependency upgrade,
   not a small vLLM source patch. A legal repair boundary therefore needs to
   decide whether dependency upgrades are allowed and how they are made
   available in an otherwise offline Agent environment.

The 4B model is a valid minimum *pipeline* probe, but no evidence establishes
that it is symptom-equivalent to the original 235B model.

## Construction-guide feedback

The following should become explicit preflight rules:

1. Separate `environment-buildable`, `pipeline-smoke-passed`,
   `symptom-reproduced`, and `solution-mapped`; do not collapse them into a
   single “Docker works” state.
2. Before task construction, verify that the available accelerator generation
   and count can reproduce the symptom. A100 success must not stand in for a
   Hopper-only regression.
3. Do not silently substitute a smaller model. A replacement is only a smoke
   fixture until baseline-fail/oracle-pass equivalence is demonstrated.
4. Pin and cache source archives outside Docker build. A pinned Git fetch is
   still operationally unstable; Docker should verify the archive digest and
   build without network.
5. For official-wheel/native reuse, test the full packaged runtime surface.
   Generated Python wrappers may be absent from the Git source archive even
   when all `.so` files are present. Fill only missing wheel files and never
   overwrite candidate source.
6. Build-time CUDA checks must use module discovery because `libcuda.so.1` is
   normally unavailable during Docker build. Actual native import belongs in
   a GPU runtime check.
7. Budget multimodal inputs by processed token count, not only file bytes.
   The 986KB image expanded to 4,470 prompt tokens and invalidated a 4096-token
   server configuration.
8. For dependency-regression issues, map the legal solution before release:
   source patch, dependency version change, or both. An offline image that
   exposes only vLLM source may make the real fix impossible.
9. A sanitized source archive alone is not a usable coding environment. The
   final image should run as a non-root user who owns a clean, writable,
   one-commit synthetic Git repository with no remote. Record this as a
   separate smoke check.
