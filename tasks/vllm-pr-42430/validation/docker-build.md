# Docker build and validation

Status: **environment-ready for focused Mamba FULL-CG metadata behavior**.
The original distributed accuracy verifier remains a separate, blocked layer.

## Eligibility and atomicity

PR 42430 is behavior-atomic despite six commits and two branch merges. Its
final diff has four files (`+120/-21`); the sole production change is 22 lines
in `vllm/v1/attention/backends/mamba_attn.py`. The other changes add/refactor
unit-test support. Review moved NIXL-specific logic out of the model runner and
requested edge-case tests; one post-merge concern notes interaction risk with
speculative decoding.

## Solution mapping

Uniform one-token batches select a full decode CUDA graph by shape. In the
failing NIXL D-side case, however, the row is still scheduler-labelled prefill
even though it has prior Mamba state. Base therefore builds prefill metadata
for a decode graph without throwing an exception. The fix reclassifies only
one-token prefill rows whose sequence length proves prior state; first-token
prompts remain prefills.

## Original and focused scopes

- Full reproduction: NIXL P/D processes and transport, a Mamba model, full
  CUDA graphs, GSM8K inputs, and paired accuracy measurement.
- Focused public Dev: real CUDA metadata, real Mamba production builder, and
  the production `build_for_cudagraph_capture` path on one A100. It does not
  claim NIXL transport, a model forward, graph replay, or accuracy coverage.

## Docker daemon

Every Docker command uses:

```bash
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
```

The daemon data root is
`/data/yaoyaoyao/pr34183-cuda-build/docker-data`. Runtime validation uses
`--network none`; no pruning or deletion is performed.

## Evidence

Remote working directory:

```text
/data/ai-infra-bench/survey-builds/vllm-pr-42430
```

The locked base archive was served only over A100 loopback during build:

```bash
source /data/akg_kernel_bench_lite/A100_proxy.sh
export DOCKER_HOST=unix:///data/yaoyaoyao/pr34183-cuda-build/docker.sock
docker build --network host --pull=false \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  --build-arg VLLM_SOURCE_URL=http://127.0.0.1:18085/source/vllm-base.tar.gz \
  -t ai-infra-bench/vllm-pr-42430:base \
  -f context/environment/Dockerfile context
```

- The selected official donor digest was already resident in the shared
  isolated daemon before this audit, so no cold-pull duration is claimed.
- Donor image ID/digest:
  `sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b`.
- Donor image size: `8,669,305,249` bytes.
- Exact-source image build: `522.21 s`.
- Final image:
  `sha256:ffd0e8ba4660bb7d48af94497ea7a6071687cba0ce374b5bdf6cbc279f81e710`.
- Final image size: `9,306,071,141` bytes.

### Base behavior

Executed with physical GPU 2 and `--network none`:

```text
observed prior_state=(decodes=0,prefills=1) first_token=(decodes=0,prefills=1)
FAIL: a one-token Mamba row with prior state stayed prefill while the production FULL-CG metadata path is decode-shaped
RC=1
```

This is the intended silent semantic failure: the control decode path succeeds,
but a scheduler-labelled single-token prefill with prior state remains Mamba
prefill metadata despite the decode-shaped full-CG dispatch.

### Isolated Oracle behavior

Only the head revision of
`vllm/v1/attention/backends/mamba_attn.py` was mounted read-only. It is not in
the Agent image:

```text
observed prior_state=(decodes=1,prefills=0) first_token=(decodes=0,prefills=1)
PASS: production Mamba FULL-CG metadata classifies prior-state single-token rows as decode and preserves first-token prefill
gpu=NVIDIA A100-SXM4-40GB capability=8.0 uuid=3815a178-ad22-4b81-5669-0533760a7e6b
RC=0
```

The first-token guard demonstrates that the Oracle does not simply reclassify
every single-token prompt.

### Integrity and hardware probes

- Runtime user: `uid=1000(agent)`; create/delete probe in `/workspace` passed.
- Candidate source: `/workspace/repo/vllm/__init__.py`.
- Candidate native binding: `/workspace/repo/vllm/_C.abi3.so`.
- Torch `2.11.0+cu130`, CUDA `13.0`; `VLLM_TARGET_DEVICE` is absent.
- GPU 2: `NVIDIA A100-SXM4-40GB`, capability `8.0`, UUID
  `3815a178-ad22-4b81-5669-0533760a7e6b`; CUDA tensor operation passed.
- Git: branch `benchmark-base`, one commit, zero remotes, clean status, tree
  `f1ee5dc01c843ebb52d20e1714a93f96ec07cb96`.
- `--network none`: `/proc/net/route` data rows `0`.
- Image history proxy matches: `0`; final image config proxy variables: `0`.
- Build verified exactly 14 donor artifacts against `native-donor.json`; the
  target/native intersection is empty and the manifest asserts no head files
  in the Agent image.
- Target-file anti-leak check: base and Agent image both hash to
  `c47d0952bf33b0d106a48eeea1f24d99f8d87b7eb87904d853723952e4426f4e`;
  head hashes to the distinct
  `69fbca4fd159a4b30b5b529f5bbd275ccc6024cdd55bd095cc82ce8d0b2d761a`.

## State separation

- Agent image: exact base source plus scoped pre-cutoff donor binaries; no PR
  head file or solved code.
- Base: deterministic semantic failure in prior-state single-token
  classification.
- Oracle: PR head `mamba_attn.py` mounted read-only only during the isolated
  check; it passes the target behavior and first-token guard.
- Full verifier: NIXL P/D + Mamba model + FULL CG + paired GSM8K accuracy.

## Survey-manual feedback

- Silent accuracy bugs need a behavior-level intermediate invariant when the
  original accuracy topology is too costly; do not replace that invariant with
  source-string checks.
- A focused CUDA metadata test must state that it does not cover model kernels,
  actual graph replay, transport, or end-to-end accuracy.
- Post-merge objections should be retained as Oracle/generalization risk even
  when the merged head passes its added unit tests.
- A warm donor in a shared daemon must be recorded as warm-resident; do not
  invent or conflate a cold-pull duration with build time.

## Remaining risk

The official donor is ABI-compatible and pre-cutoff, but it is a release build
three days older than the exact base rather than an exact-SHA native build.
The target production file is pure Python and no target file intersects the
14 copied donor artifacts, which bounds but does not eliminate this risk.

The merged Oracle also received a post-merge objection about mixing
speculative and non-speculative decode rows. This focused verifier covers the
reported non-speculative one-token contract and first-token guard only; a full
generalization verifier should add speculative-decoding batch mixtures.
