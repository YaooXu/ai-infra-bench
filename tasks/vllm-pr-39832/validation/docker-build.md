# Harbor build and validation

Status: Harbor-ready for the constructor-lifecycle contract. The upstream PR
contains 18 commits and changes 22 files, so the task is behavior-atomic only
under the narrow external v1 KV connector compatibility-removal mapping.

## Contract and solution mapping

The hidden verifier executes real production consumers and error boundaries:

- `ensure_kv_transfer_initialized` constructs a current three-argument
  connector exactly once, in WORKER role, with the identical `KVCacheConfig`;
- a `TypeError` raised inside a current connector propagates unchanged and the
  constructor is not retried;
- an external legacy two-argument connector is rejected before construction;
- direct omission of the third base-constructor argument raises `TypeError`.

This goes beyond constructor source/signature inspection. The base proves the
current consumer and internal-error boundary already work, then fails because
the factory and base still accept the deprecated lifecycle. The accepted
solution maps three production files from oracle commit
`825c3600113fae323da7a1520dcd9e276818904c`: factory dispatch, v1 base
constructor, and KV-transfer initialization.

The lifecycle is CPU-only and creates no model or CUDA tensor. `task.toml`
therefore declares `gpus = 0` and a correctness grader. A100 is used only for
independent native/source-binding integrity.

## Locked environment

- Survey base: `f80aa53c9dc2273a19a6855092069db7e1306fff`
- Canonical tree: `2af517bd7880077a9fed9a39dc0e8b1e244a48b1`
- Source archive SHA-256:
  `a2923ff0ff39b1c32b18ba6eb6255c646e2fc49280552d0637d522e9434baebe`
- Official base: `vllm/vllm-openai:v0.20.1` at digest
  `sha256:9eff9734a30b6713a8566217d36f8277630fd2d31cec7f0a0292835901a23aa4`
- Python/PyTorch/CUDA metadata: 3.12.13 / 2.11.0+cu130 / 13.0

The base SHA falls after v0.20.1 and before v0.20.2. The final candidate tree
uses only the v0.20.1 image's machine-readable ten-path native/generated
whitelist: nine regular ELF `.so` files plus `_version.py`. No future donor or
Python site-packages tree is added. The source archive is checked against its
canonical tree, committed once with `git add -f -A`, and has no remote; ignored
native/generated artifacts are installed after that canonical commit.

## Build evidence

All commands used the dedicated isolated daemon:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
source /data/akg_kernel_bench_lite/A100_proxy.sh
cd /data/ai-infra-bench/survey-builds/vllm-pr-39832/harbor-context
/usr/bin/time -p docker build \
  --network host --pull=false \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  -t ai-infra-bench/vllm-pr-39832:harbor \
  -f environment/Dockerfile environment
```

The standard Harbor context was 10.75 kB, excluding the solution, tests,
instruction, and validation documents.

```text
Successfully built f19cf5a1c619
real 659.82
image sha256:f19cf5a1c6192dd2d3c1455ebde650a8beda72f814b2e3cc34ddbd983e69a78b
size 8862497297 bytes
USER agent
WORKDIR /workspace/repo
```

The build passed exact archive/canonical-tree, one-commit/no-remote/clean-Git,
ten-path whitelist, nine-SO count, regular-file, non-symlink, and ELF checks.
Image config/history scans found no proxy environment or credential-shaped URL.

## Runtime integrity

A fresh no-GPU container with `--network none` passed:

```text
uid=1000 writable=PASS assets_absent=PASS
git_count=1 canonical_tree=PASS network_routes=0
source=/workspace/repo/vllm/__init__.py
consumer=vllm.distributed.kv_transfer.kv_transfer_state
```

`/workspace/public_dev`, `/tests`, and `/solution` were absent; the repository
was writable by `agent`, had one revision, no remote, and no dirty paths.

Independent no-network A100 GPU7 integrity also passed:

```text
gpu=NVIDIA A100-SXM4-40GB cuda_count=1 tensor=tensor([0.], device='cuda:0')
source=/workspace/repo/vllm/__init__.py
native=/workspace/repo/vllm/_C.abi3.so
factory=vllm.distributed.kv_transfer.kv_connector.factory
network_routes=0
```

## Base and Oracle

Base, with only `tests/` mounted read-only:

```text
contract_device=cpu
current_consumer_path=PASS
internal_type_error_boundary=PASS
FAIL: legacy lifecycle remained accepted factory_rejected=False
legacy_constructed=1 base_rejected=False
TEST_SH_RC=0 REWARD=0
```

Isolated Oracle, after applying `solution/fix.patch` as `agent`, modified exactly
the three target files and ran the same verifier:

```text
SOLVE_RC=0 GIT_STATUS_LINES=3
contract_device=cpu
current_consumer_path=PASS
internal_type_error_boundary=PASS
PASS: current consumer works and legacy lifecycle is rejected
TEST_SH_RC=0 REWARD=1
```

The Oracle logs the intended pre-construction rejection. The connector-internal
sentinel `TypeError` remains unchanged, proving the repair does not hide real
constructor errors behind compatibility handling.

## Remaining risks and manual feedback

- The PR is a broad migration; the benchmark claims only the factory/base/
  initialization contract, not coverage of every mechanically updated built-in
  connector.
- Native objects are a same-release pre-cutoff approximation rather than an
  exact-SHA compilation. The target contract is CPU-only, while separate A100
  import/allocation evidence establishes environment viability.
- Deprecation-removal tasks need both negative and positive boundaries: legacy
  calls must stop before construction, current consumers must preserve argument
  identity, and constructor-internal errors must not be caught or retried.
- Grader GPU requirements must reflect target execution, not an unrelated
  package-integrity probe.
