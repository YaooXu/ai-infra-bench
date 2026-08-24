# Source, toolchain, and eligibility lock

This survey item comes from merged FlashInfer PR
[#3803](https://github.com/flashinfer-ai/flashinfer/pull/3803), “Fix 0/0
NaN in GLM52 routing renorm on sigmoid underflow”. The accepted solution is
stable, but its native target is Blackwell-only and cannot run on the assigned
A100. This directory is a **hardware-blocked audit**, not an Agent environment.

## PR state, review, and atomicity

The PR was approved three times by a FlashInfer maintainer and squash-merged
on 2026-07-04. There is no linked public issue. The PR description is the only
public symptom report: vLLM/GLM-5.2 can select Top-K experts whose unbiased
BF16 sigmoid scores all underflow to zero, after which routing renormalization
computes `0 / (0 + 0)` and poisons the token output row with NaNs.

The patch is atomic: one assignment plus six explanatory comment lines in
`csrc/trtllm_fused_moe_runner.cu`. It sets the DeepSeek-V3 no-groups routing
path's `mSumEpsilon` to `1e-20f`, matching the sibling MiniMax2 branch and the
referenced DeepSeek implementation. No test file was added despite the PR
checklist saying tests were updated; a benchmark verifier must therefore be
constructed independently.

## Correct solution mapping

The survey recorded base `b869bc2d05234aa6b6ba3879d2cce236a411e3ab`
and contributor head `f8024b0f2e8cbf41cd891342803132d1571c1d70`.
That pair is not a valid Base/Oracle mapping: GitHub compare reports
`ahead_by=2`, `behind_by=2`, and merge base
`651d877827a825d73bd304e7f54bff0649a69904`.

The GitHub `merge_commit_sha` is a one-parent squash commit. The authoritative
accepted mapping is:

- Agent Base: `b5ac097e7c96faf756ceed9a94f6cc1697b920fc`;
- evaluator Oracle: `c53229efded466e5424fe82a769a2fa81c8aa6bd`;
- compare result: `ahead_by=1`, `behind_by=0`, merge base equals the Agent
  Base, one file changed, seven additions, zero deletions.

Locked source identities measured during this audit:

| Role | Commit | Git tree | Archive bytes | Archive SHA-256 |
|---|---|---|---:|---|
| Survey base | `b869bc2d05234aa6b6ba3879d2cce236a411e3ab` | `6d46785dd1003b96487592f28189bb0756161b1c` | 5,147,771 | `fc031d38f6935ba912b433f06cbd4524b993bb63caf81fc3b90d39f34bc98c92` |
| Survey head | `f8024b0f2e8cbf41cd891342803132d1571c1d70` | `4ae112840a490716f307fc936d254dd7ba9a1162` | 5,109,156 | `14da730635eab18749b8e9f071b4e65f56aafae9b55699a85eb2358cfc1ec09b` |
| Accepted Base | `b5ac097e7c96faf756ceed9a94f6cc1697b920fc` | `34b998a80adf90a0c50b5846d3a49c2518e5ca55` | 5,199,021 | `00e426d3cf69c65191f9df1ee2cac6472e9defc9a8b3a1e9b46ba67da88b349a` |
| Accepted Oracle | `c53229efded466e5424fe82a769a2fa81c8aa6bd` | `dcaa9fdc5cb97f3f525f23d192ce646a63c090cc` | 5,199,172 | `e97931ea3340125bd69747bb7b6d470cb0257fab604a853c96e42eb47798c361` |

The archives came from immutable GitHub commit archive endpoints and were
used only for read-only inspection. They are not Docker build inputs because
no image was constructed.

## Native target and architecture gate

The changed file enters exactly one registered module through
`flashinfer/jit/fused_moe.py`:

```text
generator: gen_trtllm_gen_fused_moe_sm100_module()
module:    fused_moe_trtllm_sm100
source:    csrc/trtllm_fused_moe_runner.cu
arch filter passed to NVCC flag generation: [10, 12]
upstream note: currently only support Blackwell
```

The upstream routed-MoE tests independently check the runtime device and skip
unless the compute-capability major version is 10, stating that they are only
guaranteed on SM100 and SM103. The assigned NVIDIA A100-SXM4-40GB is SM80.

FlashInfer does support other modules on A100, but none of those modules
compile or load the affected file. Compiling a generic CUTLASS MoE target for
SM80 would therefore be a false native-path claim and cannot reproduce the
NaN or validate the patch.

## Complete future target dependencies

At the accepted Base, the upstream CUDA 13.0 development environment starts
from `nvidia/cuda:13.0.0-devel-ubuntu24.04`. The immutable Docker Hub index
digest observed during this audit is
`sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d`;
its linux/amd64 child is
`sha256:435220c0fef35cbf712e11999f8670a83835ef3cdd18564e5e8122f83078c88c`.
This is a prospective eligible-worker toolchain lock, not a built or validated
Agent image.

The exact target requires CUDA/NVCC, a C++17 compiler, Python 3.12, PyTorch,
Ninja, `apache-tvm-ffi`, and the source-locked FlashInfer package. Its common
JIT include path consumes these Git submodules from the accepted Base:

| Dependency | Gitlink commit | Target relevance |
|---|---|---|
| NVIDIA CCCL | `876867684f7fac130e0f5911236e0a92a970d4fd` | common CUDA headers |
| NVIDIA CUTLASS | `b46b16d003484063bca4ed365e44095c4c6ed633` | common CUTLASS headers |
| spdlog | `c3aed4b68373955e1cc94307683d44dca1515d2b` | common logging headers |

The repository also contains NIXL and NCCL gitlinks, but they are not on this
single-GPU fused-MoE target path and must not be mislabeled as required native
inputs.

`gen_trtllm_gen_fused_moe_sm100_module()` additionally consumes TensorRT-LLM
batched-GEMM metadata/export headers and runtime cubins from FlashInfer's
public artifact store:

```text
artifact directory:
481dce07c89a216cbfd18cf39de49a82d40739a8/batched_gemm-dd6d23e-721ae60/
checksums.txt bytes: 509789
checksums.txt SHA-256:
aa19cf2a37eed029eee5b3f96b37e069e4ab40f419b25ed7a3fd9526d8833bfb
```

A future offline build must mirror the exact required files from that manifest
and verify every digest before setting `FLASHINFER_NO_DOWNLOAD=1`. GitHub
source archives omit gitlink contents, so the three relevant submodules must
also be supplied as separate digest-verified archives. Python wheels must be
locked by filename and SHA-256 rather than reinstalling the unbounded ranges
in upstream `requirements.txt`.
