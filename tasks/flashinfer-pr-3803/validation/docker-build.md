# Construction and validation record

## Status

`HARDWARE-BLOCKED-BEFORE-BUILD`

The accepted source patch is stable, but the normal target registration on the
assigned A100 does not compile the affected native module, and SM80 cannot
execute it. A CUDA 13 toolchain could cross-compile Blackwell code if forced,
but that would still provide no runtime validation. No Dockerfile was
constructed, no image was pulled or built, and no image tag was created. This
directory must not be published as a runnable benchmark.

## Why an A100 probe would be invalid

The only changed source file, `csrc/trtllm_fused_moe_runner.cu`, is registered
only in `gen_trtllm_gen_fused_moe_sm100_module()`. That generator produces
`fused_moe_trtllm_sm100`, marks the path Blackwell-only, and requests only
compute-capability families 10 and 12. Upstream's corresponding routed fused
MoE tests skip devices whose compute-capability major is not 10.

The assigned worker exposes NVIDIA A100-SXM4-40GB devices, which are SM80.
A generic FlashInfer import test, source assertion, CPU reproduction of
`0 / 0`, or compilation of a different SM80 MoE extension would not prove
that the changed file entered the running CUDA path. Per the native-path hard
gate, no fallback image or structural probe was made.

## Future exact-native construction

Resume only on an eligible SM100/SM103 Blackwell worker, such as B200 or GB200.
Use accepted squash parent `b5ac097e7c96faf756ceed9a94f6cc1697b920fc`
for the Agent Base and merged commit
`c53229efded466e5424fe82a769a2fa81c8aa6bd` only for an evaluator-side
Oracle. Both must use the same digest-pinned CUDA 13 toolchain, exact Python
wheel lock, exact submodule archives, and mirrored TensorRT-LLM BMM artifacts.

Build only the complete affected target through
`gen_trtllm_gen_fused_moe_sm100_module()`; do not compile a single `.cu` file
or borrow a release `.so`. The resulting module must expose:

- exact Base or Oracle source identity;
- `fused_moe_trtllm_sm100` as exact-source native scope;
- SM100/SM103 cubin inventory from `cuobjdump`;
- the changed runner file in generated compile commands;
- no public network access during build or runtime.

Only the Base image may be delivered to the coding Agent. It must run as a
non-root user with a writable source tree, one synthetic Git commit, no remote,
and a clean initial worktree. The Oracle source, Oracle image, verifier, and
artifact manifest must remain evaluator-side.

## Future correctness verifier

Correctness is the hard gate. Exercise the full public fused-MoE entry point
with `RoutingMethodType.DeepSeekV3`, the no-groups routing path, BF16 routing
scores, a nonzero routing bias, and Top-K settings that select experts whose
unbiased logits are sufficiently negative for all selected sigmoid scores to
underflow to exact zero.

Required paired outcomes:

1. Exact Base reproduces a NaN in the degenerate token's output row.
2. Exact Oracle produces a finite output and agrees with a reference using
   denominator `sum + 1e-20`.
3. Non-degenerate rows are bit-identical between Base and Oracle, matching the
   PR's stated non-regression property.
4. Multiple token counts and Top-K values cover the relevant routing kernel
   tiers rather than a single launch shape.
5. The loaded `.so`, compile commands, and cubin inventory prove the affected
   source and SM100 target are active.

No GLM-5.2 checkpoint is necessary once the native symptom is exercised
through the real public kernel with genuine BF16/CUDA tensors. This reduces
data volume without replacing the affected native path or revealing the
one-line fix inside the Agent image.

## Performance layer

The PR makes no speed claim. Performance is therefore a non-regression layer
run only after correctness passes: measure paired same-process median latency
and throughput for normal routing rows on identical shapes, with CUDA events,
warm-up, synchronization, and repeated batches. Report the distribution; do
not invent a speedup threshold. Degenerate-row finiteness and normal-row
bitwise equality remain the acceptance criteria.

## Construction-guide feedback

- For CUDA PRs, map the changed source file to its exact registered module
  before building. Repository-wide minimum architecture is insufficient when
  the affected module has a narrower Blackwell-only gate.
- A successful build of another backend is not a valid fallback if the changed
  translation unit is absent from its source list.
- For squash merges, use squash parent to merged commit as the authoritative
  Base/Oracle pair; survey base/head may have diverged even when the PR is
  merged and stable.
- JIT artifact manifests and export headers are build dependencies. Mirror and
  digest-lock them before enforcing offline construction.
- Numerical correctness PRs need an adversarial native input, a finite
  reference, and normal-input bitwise non-regression before any timing result.
