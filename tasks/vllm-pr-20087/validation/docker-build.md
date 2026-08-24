# Docker build and hardware eligibility validation

Status: hardware-blocked eligibility audit. The exact-base Agent image is
built and sanitized, but Base, Oracle, and target Verifier cannot execute the
SM100 kernel on the available A100/SM80. Do not count this as
`environment-ready` for benchmark execution.

## Atomicity and solution mapping

PR 20087 has 43 review commits and a final 16-file `+397/-114` patch. Its
benchmarkable core is coherent: add a DeepGEMM old/new API compatibility
wrapper, detect SM100, cast/requantize block-FP8 scales to UE8M0, and route MoE
call sites through the wrapper. The one-line `causal_conv1d` Triton import fix
is excluded from the solution/oracle boundary.

The full target verifier must run on SM100/B200 with a pinned DeepGEMM v2
revision. It must exercise real per-token/per-block casts, UE8M0 weight
requantization, and grouped FP8 GEMM, compare correctness against a reference,
and report paired base/oracle timing from the same GPU. Static symbol searches,
an SM90 old-API run, or a Triton fallback are not substitutes.

The external dependency is not reproducibly pinned by the original PR. At the
2025-07-10 vLLM cutoff, DeepGEMM PR 112's latest commit was
`cc416ee4faf0533a9263c2de814e5565f56ca1cc`; its later merge head
`4c4ff2e4ffcdbd8ddac3fb1f9c8caccec27f932a` was created on 2025-07-18.
The author environment's installed SHA is not stated, so a future B200 run must
first resolve this dependency lock rather than assuming the post-cutoff head.

## Docker daemon

All Docker commands use:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
```

The daemon data root is
`/data/yaoyaoyao/pr34183-cuda-build/docker-data`. Runtime probes use
`--network none`; no pruning or deletion is performed.

The daemon inherited `/data/akg_kernel_bench_lite/A100_proxy.sh` for the cold
Docker Hub pull. The image build needed no external package network: it read
the checksummed base archive from a temporary A100 loopback server, which was
stopped afterward. Final config, runtime environment, and full image history
contain no proxy values or credentials.

## Release selection and pull

The base commit is dated 2025-07-10. v0.9.2 was published on 2025-07-07 and
v0.10.0 on 2025-07-24, making v0.9.2 the closest pre-cutoff official image.

```bash
source /data/akg_kernel_bench_lite/A100_proxy.sh
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
/usr/bin/time -p docker pull \
  vllm/vllm-openai@sha256:37cd5bd18d220a0f4c70401ce1d4a0cc588fbfe03cc210579428f2c47e6eac33
```

Observed:

```text
real 399.96
image ID sha256:37cd5bd18d220a0f4c70401ce1d4a0cc588fbfe03cc210579428f2c47e6eac33
size 10674731647 bytes
vLLM 0.9.2, torch 2.7.0+cu128, CUDA 12.8
native import and A100 CUDA allocation passed
deep_gemm None
```

## Build and probe evidence

Remote context:
`/data/ai-infra-bench/survey-builds/vllm-pr-20087/context`

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
cd /data/ai-infra-bench/survey-builds/vllm-pr-20087
/usr/bin/time -p docker build \
  --network host \
  --pull=false \
  --build-arg VLLM_SOURCE_URL=http://127.0.0.1:18082/vllm-base.tar.gz \
  -t ai-infra-bench/vllm-pr-20087:base \
  -f context/environment/Dockerfile context
```

Result:

```text
Successfully built 39b985437269
Successfully tagged ai-infra-bench/vllm-pr-20087:base
real 418.37
```

- Image ID:
  `sha256:39b98543726950715c5ec54372d19128d72e08d23b82a97d889e00873a00942f`
- Image size: `11,480,327,981` bytes
- Exact archive SHA-256 and canonical tree assertions: passed
- Synthetic commit: `40b93fbf21435bd6bac63b49dc7dc49db2f0d83c`
- Git: one commit, tree `d2526d84114539875b099b79ca4002f1d39c977e`,
  no remote, clean worktree
- Runtime: UID 1000, candidate tree writable

The real hardware gate was run with physical A100 GPU 2:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
docker run --rm --network none --gpus device=2 \
  ai-infra-bench/vllm-pr-20087:base \
  bash /workspace/public_dev/run.sh
```

Exit status: 2, intentional hardware-blocked result.

```text
device=NVIDIA A100-SXM4-40GB
capability=8.0
uuid=3815a178-ad22-4b81-5669-0533760a7e6b
torch=2.7.0+cu128 cuda=12.8
candidate_source=/workspace/repo/vllm/__init__.py
candidate_native=/workspace/repo/vllm/_C.abi3.so
deep_gemm_available=False
uid=1000 target_device=None
BLOCKED: observed a real CUDA device, but PR 20087 targets DeepGEMM v2 block-FP8 kernels on SM100/B200; executing an SM90 or fallback kernel would not validate the requested feature.
```

This is executable device/native behavior, not a source-string check. An
additional integrity probe successfully allocated a CUDA tensor, observed zero
network route rows under `--network none`, and confirmed:

```text
git_count 1
git_remote_rows 0
git_status_rows 0
route_rows 0
proxy_env_rows 0
writable True uid 1000
source /workspace/repo/vllm/__init__.py
native /workspace/repo/vllm/_C.abi3.so
target_device None
offline 1 1
gpu NVIDIA A100-SXM4-40GB (8, 0) 3815a178-ad22-4b81-5669-0533760a7e6b
tensor tensor([0.], device='cuda:0')
```

Image config and image history had zero proxy-address/credential matches.
`VLLM_TARGET_DEVICE=empty` was never used.

## State separation

- Agent image: built, exact base source with same-release native donor; suitable
  for inspection but not target-kernel execution.
- Base behavior: hardware eligibility only; target kernel cannot run on SM80.
- Oracle behavior: hardware-blocked; head source alone cannot create SM100.
- Verifier: requires an SM100/B200 runner and pinned DeepGEMM v2 dependency.

No solved PR code is placed in Agent-visible assets.

## Survey-manual feedback

- Run architecture eligibility before expensive external-kernel builds.
- Do not accept fallback kernels as evidence for hardware-enable PRs.
- Lock external kernel source independently of the vLLM source cutoff.
- A linked, unmerged external PR is not a dependency lock. Record the exact
  source SHA or wheel hash from the author's environment; if it is absent,
  mark dependency provenance unresolved.
- Record Agent, Base, Oracle, and Verifier readiness separately; a successfully
  built inspection image does not make a hardware-blocked benchmark runnable.
