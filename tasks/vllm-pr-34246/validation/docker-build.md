# Harbor build and validation record

## Outcome

- Task status: **Harbor-ready**.
- Environment: built from the standard `task/environment` context.
- Final strengthened A100 verifier: **Base reward 0, accepted Oracle reward
  1**.
- Full 235B/8xH100 serving: deliberately out of scope; the verifier exercises
  the exact production merge primitive independently validated by the PR test
  and issue reporter.

The final verifier imports both `_C` and `_C_stable_libtorch`, and its shell
wrapper rejects duplicate-registration diagnostics before awarding reward.

## Accepted mapping and task boundary

PR #34246 was squash-merged. The Agent receives the squash parent
`36d7f19897843c9cbdb701ba88d0f2c29954fe44`; the evaluator patch reconstructs
accepted commit `4f6eed3bd4a92c6bd513460ee85b917d6df88a17`. Applying the locked patch to a
fresh Base archive produced a byte-identical Oracle source tree. The solution
patch is evaluator-only and is not part of the environment context or image.

The hidden verifier calls the real production
`vllm.model_executor.models.utils._merge_multimodal_embeddings` with CUDA
target/source embeddings. It checks:

- CPU masks work without synchronization;
- no explicit CPU-to-CUDA mask materialization occurs, using Torch dispatch;
- nested embeddings preserve flattening order and dtype conversion;
- the input is updated in place and non-placeholder rows remain unchanged;
- FP16, BF16, and FP32 paths have bounded temporary allocator growth;
- GPU-mask behavior remains compatible, but is not incorrectly held to the
  CPU-mask asynchronous contract;
- both excess and missing multimodal embeddings raise precise cardinality
  errors;
- the candidate source and native paths are inside `/workspace/repo`;
- actual loading of `_C` plus `_C_stable_libtorch` emits no duplicate
  registration diagnostic.

## Host, daemon, and immutable image

- Host: `bm-baai-dx-zone1-d-a100-40g-2-106`;
- GPU: physical GPU 1, NVIDIA A100-SXM4-40GB;
- Docker host:
  `unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock`;
- Docker data root:
  `/data/yaoyaoyao/pr34183-cuda-build/docker-data`;
- remote task directory:
  `/data/ai-infra-bench/survey-builds/vllm-pr-34246-harbor`;
- image tag: `ai-infra-bench/vllm-pr-34246:harbor-base`;
- image ID:
  `sha256:816a69704cc034e4014496a7bdf92e56ab8099af9b1dba3b8bbba00fc003218c`;
- inspect size: `10,075,782,140` bytes;
- configured user: `agent`.

Final evaluator asset identities:

```text
test.sh sha256=5c22c88d62c92a77ff413cf6a6a528f9e7dd67cdae9c7330f77f2fd3eb8deaf6
verify_multimodal_merge.py sha256=1e2a9f0754fa691d6f317fe3c5bfc2e7561a26b4b242d992ba04d60e054ce3ea
fix.patch sha256=6f4c23941397f5ca178724efd6a002a20ebe5b4075e0319265ebc50ad2d29256
```

No default Docker daemon, pruning, or deletion of another task's images was
used.

## Standard-context build

The accepted build context contains only `environment/Dockerfile` and
`environment/lock/`. `tests/`, `solution/`, `instruction.md`, and `task.toml`
are siblings outside that context.

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
docker build \
  --network host \
  --pull=false \
  -t ai-infra-bench/vllm-pr-34246:harbor-base \
  /data/ai-infra-bench/survey-builds/vllm-pr-34246-harbor/context/environment
```

The successful build took approximately 264 seconds wall time, including the
31 MB immutable source download and large-layer commits. It proved:

```text
source archive sha256: OK
canonical tree: 647523aac911ae44a1dcaab79ef66d58290b478e
synthetic commits: 1
remotes: 0
native manifest: all 9 records OK
regular shared objects: 8
candidate source: /workspace/repo/vllm/__init__.py
configured user: agent
```

Docker builds do not expose `libcuda.so.1`; the build-time production import
therefore logged the expected CUDA-extension warning and continued. GPU/native
loading belongs to runtime validation.

## Base = 0

The verifier and logs were mounted from outside the image. Runtime networking
was disabled and GPU 1 was selected:

```bash
docker run --rm --network none --gpus device=1 --user root \
  -v .../context/tests:/tests:ro \
  -v .../logs/base-final:/logs/verifier \
  ai-infra-bench/vllm-pr-34246:harbor-base \
  bash /tests/test.sh
```

The exact Base reached the production function and failed for the intended
reason, not import or environment setup:

```text
candidate_source=/workspace/repo/vllm/__init__.py
gpu=NVIDIA A100-SXM4-40GB
RuntimeError: Expected all tensors to be on the same device, but got mask is
on cpu, different from other tensors on cuda:0
ValueError: Error during masked scatter operation
reward=0
```

## Accepted Oracle = 1

The accepted patch was mounted read-only and applied only inside a disposable
offline container. The unchanged hidden verifier was rerun:

```bash
docker run --rm --network none --gpus device=1 --user root \
  -v .../context/tests:/tests:ro \
  -v .../context/solution:/solution:ro \
  -v .../logs/oracle-final:/logs/verifier \
  ai-infra-bench/vllm-pr-34246:harbor-base \
  bash -lc 'git apply /solution/fix.patch && bash /tests/test.sh'
```

Observed final strengthened result:

```text
peak_ratio[cpu_torch.float16]=1.003
peak_ratio[cpu_torch.bfloat16]=1.003
peak_ratio[cpu_torch.float32]=0.500
peak_ratio[cuda_bfloat16]=1.003
PASS: production merge is ordered, async, bounded, strict, and CPU-mask native
reward=1
```

The same final run actually imported both native modules from the candidate
tree. The wrapper found zero duplicate-registration matches. A separate
default-user integrity container recorded:

```text
uid=1000(agent) gid=1000(agent)
COMMITS=1
REMOTES=0
DIRTY=0
GPU=NVIDIA A100-SXM4-40GB
_C=/workspace/repo/vllm/_C.abi3.so
_C_stable_libtorch=/workspace/repo/vllm/_C_stable_libtorch.abi3.so
```

An earlier verifier revision incorrectly placed the GPU-mask compatibility
control under the no-sync guard. The accepted Oracle correctly exposed that
GPU boolean indexing may query dynamic cardinality. The final verifier scopes
the no-sync hard gate only to CPU masks, exactly matching the PR contract; the
GPU-mask case remains a correctness and allocation control.

## Native risk disposition

The old survey image is retired because it mixed v0.18.1 native objects with a
single v0.19.0 stable-libtorch object and emitted four duplicate-registration
diagnostics. The Harbor image instead uses all eight extensions from one
digest-pinned v0.19.0 image and validates their hashes before copying.

This removes the mixed-native failure mode. It remains a scoped donor design,
not an exact Base native build. That limitation is acceptable only because the
scored primitive is implemented entirely in candidate Python and PyTorch; it
does not dispatch a vLLM native operator. The verifier imports the native
objects and rejects warnings so general import health is still a hard gate.

## Remaining scope limit

The benchmark does not claim end-to-end 235B TP/DCP/EP coverage. Upstream
provides that evidence: the issue reporter confirmed the PR eliminated the
8xH100 OOM and preserved ChartQA behavior. The local fixture is credible
because the upstream PR itself isolates the bug to this production primitive
and adds a CUDA no-sync test at the same boundary. The verifier adds allocator,
ordering, dtype, nesting, and strictness coverage without substituting a mock.
