# Docker build and baseline validation

Status: environment-ready for the focused node-local supervisor lifecycle.
The exact base deterministically fails because the supervisor is absent; an
ephemeral PR-head module injection makes the same behavioral harness pass. It
is not a Kubernetes end-to-end environment.

## Atomicity and external testbed gate

PR 40841 has 60 commits, 7 final changed files, and `+1168/-17`. The commit
count reflects review iteration, while the final patch implements one coherent
node-local protocol: derive rank/port arguments, launch one API child per local
DP rank, aggregate health/readiness, propagate child failure, and terminate the
remaining process tree. The benchmark is behavior-atomic at that protocol
boundary.

Kubernetes is a downstream consumer of the admin probe and rank ports, not an
implementation dependency. The public Dev therefore tests real node-local
subprocess, HTTP, port, readiness, and failure-propagation semantics. It does
not claim Kubernetes Service/probe correctness, multi-node routing, or real
model throughput.

## Docker daemon

All commands use the isolated daemon:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
```

Its data root is `/data/yaoyaoyao/pr34183-cuda-build/docker-data`. Runtime
commands use `--network none`; no image pruning or deletion is performed.

The daemon inherited `/data/akg_kernel_bench_lite/A100_proxy.sh` for the cold
Docker Hub pull. The successful build passed Docker's predefined
`HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` args for apt only; the Dockerfile does not
redeclare them, and final config/history checks found no proxy values or
credentials. The checked source tarball was provided by a temporary server
bound to A100 `127.0.0.1` and the server was stopped after the build.

## Official release selection and cold pull

The exact base commit is dated 2026-05-21. v0.21.0 was published on 2026-05-15
and v0.22.0 on 2026-05-29, so v0.21.0 is the latest official release image at
the source cutoff. The selected amd64 manifest is:

```text
vllm/vllm-openai:v0.21.0@sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b
```

Cold-pull command:

```bash
source /data/akg_kernel_bench_lite/A100_proxy.sh
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
/usr/bin/time -p docker pull \
  vllm/vllm-openai@sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b
```

Result:

```text
real 565.98
donor image ID sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b
donor size 8669305249 bytes
vLLM 0.21.0, torch 2.11.0+cu130, CUDA 13.0
```

Before source overlay, the donor imported its native `_C`, allocated a CUDA
tensor on physical GPU 2, and reported A100 UUID
`GPU-3815a178-ad22-4b81-5669-0533760a7e6b`.

## Build

Remote context:
`/data/ai-infra-bench/survey-builds/vllm-pr-40841/context`

The first attempt stopped after 132.69 seconds because the official runtime
image intentionally does not include Git. It did not produce or validate a
candidate image. The Dockerfile was then corrected to install exact Git
package versions before extracting the source.

Successful command:

```bash
source /data/akg_kernel_bench_lite/A100_proxy.sh
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
cd /data/ai-infra-bench/survey-builds/vllm-pr-40841
/usr/bin/time -p docker build \
  --network host \
  --pull=false \
  --build-arg HTTP_PROXY \
  --build-arg HTTPS_PROXY \
  --build-arg NO_PROXY \
  --build-arg VLLM_SOURCE_URL=http://127.0.0.1:18081/vllm-base.tar.gz \
  -t ai-infra-bench/vllm-pr-40841:base \
  -f context/environment/Dockerfile context
```

Result:

```text
Successfully built 1b05f963c00c
Successfully tagged ai-infra-bench/vllm-pr-40841:base
real 300.93
```

- Image ID:
  `sha256:1b05f963c00cf03a95f019117f061a7cb169cf573ad5e2542649168232bfd4bf`
- Image size: `9,309,035,545` bytes
- Runtime user: `agent` (UID 1000), candidate tree writable
- Exact source SHA-256: passed
- Forced-add canonical tree: `94c86336cf2ea962766d00bb389d43a4d6aaf697`
- Synthetic commit: `d7f5214d44d9d0e2e38c84ecbbab6c718090e31e`
- Git: one commit, branch `benchmark-base`, no remote, clean worktree

## Public baseline

Command:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
docker run --rm --network none \
  ai-infra-bench/vllm-pr-40841:base \
  bash /workspace/public_dev/run.sh
```

Exit status: 1, expected base failure.

```text
FAIL: base revision has no node-local DP supervisor; the required real subprocess/health/port lifecycle cannot start
DETAIL: No module named 'vllm.entrypoints.openai.dp_supervisor'
```

This is a feature-addition baseline, so the old tree cannot start the new
lifecycle. The check is executable import/behavior setup, not a search for
source strings.

To validate that the harness reaches its intended behavioral pass, the final
PR-head `dp_supervisor.py` was bind-mounted read-only into an ephemeral
container; it was never copied into the base image or task assets. The same
script then ran with physical GPU 2 exposed and `--network none`:

```text
oracle_rc=0
Started DPSupervisor on 127.0.0.1:23002
DPSupervisor found 1 exited DP Servers.
DPSupervisor forwarding SIGTERM to DP Servers.
PASS: real DP supervisor aggregated two child health endpoints, published readiness, propagated child failure, terminated its sibling, and released all three ports
```

The harness uses the production `DPSupervisor` and its multiprocessing spawn,
two real HTTP child processes, and three real loopback ports. It observes
aggregate health `503 -> 200`, kills one child with SIGKILL, waits for sibling
termination, and proves all ports are released.

## GPU, source/native, non-root, and offline probes

The lifecycle workload is intentionally GPU-light: it replaces model servers,
not supervisor logic. A separate integrity probe used physical A100 GPU 2 and
loaded the actual native extension from the candidate tree:

```text
uid 1000
versions 0.21.0 2.11.0+cu130 13.0
source /workspace/repo/vllm/__init__.py True
native /workspace/repo/vllm/_C.abi3.so True
target_device None
offline 1 1
ifaces [(1, 'lo'), (2, 'tunl0')]
writable ok
gpu 1 NVIDIA A100-SXM4-40GB 3815a178-ad22-4b81-5669-0533760a7e6b
tensor tensor([0.], device='cuda:0')
```

`docker run --network none` yielded zero `/proc/net/route` rows. The final
image config and runtime both had zero `HTTP_PROXY`, `HTTPS_PROXY`, or
`ALL_PROXY` variables; full image history had zero proxy-address/credential
matches. `VLLM_TARGET_DEVICE=empty` was never used.

## Remaining risks

- Full Kubernetes and actual two-rank model serving are deliberately outside
  this focused public Dev.
- Native extensions are donated by the nearest official pre-cutoff release
  rather than compiled from the exact SHA; this is appropriate only because
  the scoped PR is Python-only. Real `_C` import, CUDA allocation, and exact
  source/native path probes pass, but this remains a same-release ABI donor.
- The public baseline fails at the missing feature boundary. Behavioral pass
  was proven by an ephemeral oracle-module mount, not by building a solved
  image; benchmark publication should keep that oracle out of agent-visible
  assets.

## Survey-manual feedback

- Large review-iteration commit counts should not alone reject a task; audit
  the final behavior boundary and changed-file cohesion.
- Separate node-local supervisor contracts from Kubernetes deployment claims.
  A cluster is required only for the latter.
- Supervisor tests must use real child processes, loopback sockets, health
  transitions, and failure cleanup; source-string checks are not sufficient.
- A lightweight child server can replace model loading when the contract is
  orchestration rather than inference correctness, but GPU/native/path probes
  must be reported separately.
- For feature-addition baselines, require a solved-tree or ephemeral oracle
  check showing that the same behavioral test progresses beyond the expected
  missing-symbol failure; otherwise a typo in the test can masquerade as a
  valid baseline failure.
