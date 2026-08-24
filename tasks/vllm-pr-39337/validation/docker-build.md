# Docker build and baseline validation

Status: environment-ready for the config-contract scope. Base failure and an
isolated Oracle positive pass were both observed. This remains environment-only:
consumer integration is explicitly out of scope.

## Scope and atomicity

The upstream PR has 55 commits, changes 6 files, and is one step in the open
multi-PR Model Runner V2 migration. It is not commit-atomic. The publishable
scope is the narrow runner-selection contract: tri-state environment override,
Qwen3 dense default selection, and unsupported-feature fallback/validation.

The full patch also changes Distributed FlashInfer consumption of the runner
flag and was immediately linked to a four-GPU NIXL/FlashInfer P/D nightly block
count mismatch. That distributed behavior is excluded from this benchmark
oracle; treating all six files as a solved atomic gold would be unsafe.

The PR head is `0579be818c0d2b438cd41b76d8d09f9338ac1fd8`, with
`+208/-20`. Review requested the final fallback matrix (including custom logits
processors and no-Triton platforms) rather than blindly setting V2 in tests.
The author explicitly limited this first migration step to Qwen dense models;
the umbrella issue describes later dense, MoE, quantized, MLA/DSA, multimodal,
and mamba steps. PR #39353 was the stated prerequisite and had merged before
this PR. These facts support a narrow config contract, not a complete model
runner migration oracle.

The two-sided boundary is:

- Explicit `VLLM_USE_V2_MODEL_RUNNER=0/1` remains a hard override.
- An unset variable is represented by `None`, allowing configuration selection.
- Only dense, unquantized Qwen3 generation defaults to V2 in this step.
- Unsupported features automatically fall back to V1; forced V2 rejects them.

The production `VllmConfig` methods are executed dynamically. No test searches
source text. Scheduler, GPUWorker, and FlashInfer consumer integration is not
verified or claimed: lightweight construction does not exist, and the latter
would couple the benchmark to the known distributed regression. Publication
status must therefore remain `config-contract / environment-only`.

## External dependency preflight

The original end-to-end scenario names `Qwen/Qwen3-0.6B`, which would require
external Hugging Face artifacts. The selection behavior is entirely determined
by `VllmConfig`, so the public Dev uses local synthetic model configuration
objects and executes production selection methods. It needs no model, tokenizer,
dataset, network, connector, or service. A separate integrity probe uses A100
GPU 0 because the behavioral workload is intentionally host-side.

## Docker daemon

All Docker commands use:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
```

The daemon data root is
`/data/yaoyaoyao/pr34183-cuda-build/docker-data`. The default daemon is not
used, and no pruning or deletion is performed.

## Build, baseline, and integrity evidence

### Official-image pre-overlay probe

The official v0.20.2 image is the newest release before the base commit cutoff:

```text
repository digest sha256:70a098d90dbab428a001d9e852fc0fc8d67da5beb03e7851a22247653bf35923
image size 8231364540 bytes
created 2026-05-08T20:26:26.98773105Z
vllm 0.20.2
torch 2.11.0+cu130
cuda NVIDIA A100-SXM4-40GB
allocation 128 elements
vllm._C /usr/local/lib/python3.12/site-packages/vllm/_C.abi3.so
vllm._custom_ops import passed
```

The native manifest contains exactly nine regular ELF `.so` files plus
generated `_version.py`. All ten hashes passed both before and after copying;
the final candidate tree contains exactly nine `.so` files.

### Build

Remote context:
`/data/ai-infra-bench/survey-builds/vllm-pr-39337/context`

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
source /data/akg_kernel_bench_lite/A100_proxy.sh
cd /data/ai-infra-bench/survey-builds/vllm-pr-39337
/usr/bin/time -p docker build \
  --network host \
  --pull=false \
  --build-arg HTTP_PROXY \
  --build-arg HTTPS_PROXY \
  --build-arg NO_PROXY \
  -t ai-infra-bench/vllm-pr-39337:base \
  -f context/environment/Dockerfile context
```

Initial real build result before the manifest-whitespace-only cached rebuild:

```text
Successfully built 50f83080da46
Successfully tagged ai-infra-bench/vllm-pr-39337:base
real 514.59
user 0.10
sys 0.03
```

Warning-free final rebuild result:

```text
Successfully built 289e072dfaa7
Successfully tagged ai-infra-bench/vllm-pr-39337:base
real 335.07
user 0.06
sys 0.03
```

- Final image ID:
  `sha256:289e072dfaa7adc8f5a291adc56c4242580673b4122cafe89731d1e746f9833e`
- Final image size: `8,853,930,570` bytes
- Runtime user: `agent` (UID 1000)
- Synthetic commit: `f9e756b26bed94e7cede482d8de4b4dfda224319`
- Branch: `benchmark-base`
- Root tree: `00dedbf38a187d7be24007ab92ce0158c646d024`
- Image proxy environment rows: `0`
- Image history rows containing proxy assignments: `0`

The manifest-whitespace fix removed the earlier `sha256sum` formatting warning;
all ten artifact checks completed with only `OK` results in the final build.

### Public Base baseline

```bash
docker run --rm --network none \
  ai-infra-bench/vllm-pr-39337:base \
  bash /workspace/public_dev/run.sh
```

Exit status: `1` (expected).

```text
FAIL: model-runner selection contract is incomplete
 - env None: expected None, got False
 - default model selector is unavailable
 - runner oracle property is unavailable
```

This is a feature-addition Base failure with preserved legacy explicit `0/1`
behavior. It is not accepted alone; the Oracle pass below proves that the
public behavior distinguishes the intended solution.

### Isolated Oracle positive pass

The official cumulative PR diff was downloaded only to the remote validation
directory, never copied into the Agent context or image:

```text
/data/ai-infra-bench/survey-builds/vllm-pr-39337/oracle.diff
SHA-256 2db98aeab32bfc2a66abfbaa4a851c8e8171050f198896c79808857820090ed7
size 13987 bytes
```

It was bind-mounted read-only and applied inside a disposable offline
container based on the Base image:

```bash
docker run --rm --network none --gpus device=0 \
  -v /data/ai-infra-bench/survey-builds/vllm-pr-39337/oracle.diff:/tmp/oracle.diff:ro \
  ai-infra-bench/vllm-pr-39337:base \
  bash -lc 'git apply --check /tmp/oracle.diff && \
    git apply /tmp/oracle.diff && bash /workspace/public_dev/run.sh'
```

The diff changed the expected six upstream files and the public Dev exited 0:

```text
tests/test_config.py
vllm/config/vllm.py
vllm/envs.py
vllm/v1/attention/backends/flashinfer.py
vllm/v1/core/sched/scheduler.py
vllm/v1/worker/gpu_worker.py
PASS: tri-state overrides, default selection, and fallback all work
```

This demonstrates the positive boundary without shipping solved code.
The warning immediately before PASS is expected: it is production behavior for
the synthetic unsupported-feature fallback case, not a test or environment
warning.

### GPU, source binding, offline, and sanitization

All runs used `--network none`; the integrity probe additionally used GPU 0:

```text
uid 1000
vllm 0.20.2
torch 2.11.0+cu130
source /workspace/repo/vllm/__init__.py
native /workspace/repo/vllm/_C.abi3.so
custom_ops /workspace/repo/vllm/_custom_ops.py
gpu NVIDIA A100-SXM4-40GB 8.0
target_device None
offline 1 1
git_count 1
git_remote_rows 0
git_status_rows 0
pyc_count 0
so_count 9
route_file_lines 1
user 1000
```

Both `/workspace/repo` and `/workspace/public_dev` are writable by `agent`.

## Remaining risks

- Nearest-release native artifacts are not a compilation of the exact SHA,
  though they share Torch/CUDA pins and are not invoked by the target workload.
- Distributed FlashInfer/NIXL and full Qwen inference are deliberately outside
  the scoped contract.
- Scheduler and GPUWorker consumer propagation is not covered. The environment
  must not be described as validating the complete six-file PR.

## Survey-manual feedback

- A long-running migration PR should be publishable only after extracting a
  two-sided behavior contract; file count alone does not establish atomicity.
- Require explicit distinction between host-side selection tests and mandatory
  GPU/native environment-integrity probes.
- An end-to-end model named by a PR should not force weight downloads when the
  changed behavior has a deterministic local configuration boundary.
- Immediate follow-up/revert evidence must be used to exclude unsafe consumers
  from a supposedly atomic oracle.
- Feature-addition tasks need an isolated Oracle positive pass; Base failing on
  a missing property is insufficient by itself.
