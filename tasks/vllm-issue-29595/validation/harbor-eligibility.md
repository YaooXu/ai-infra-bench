# Harbor eligibility re-audit (2026-08-25)

## Decision

`BLOCKED-BEFORE-HARBOR-PACKAGING`

This directory retains a useful A100 environment and historical production
pipeline evidence, but it must not receive `task.toml`, `instruction.md`, a
hidden pass/fail verifier, or an Oracle solution. The available A100 is a known
negative platform for the reported regression, so manufacturing Base=0 and
Oracle=1 from a smaller model, a source assertion, or a static compiler probe
would change the task.

## Rechecked production behavior

The issue is specifically a grounding-accuracy regression under compiled vLLM
execution. The public reports establish all of the following:

- vLLM 0.11.1/0.11.2 with Qwen3-VL loses grounding accuracy on H20, H100, and
  H200; eager execution restores it.
- The reporter who compared architectures found that the same workload kept
  its grounding capability on A100. A later user explicitly confirmed no
  degradation on A100, A800, or Jetson.
- The defect was narrowed from CUDA Graph capture to an Inductor/Triton kernel
  generated through a PTXAS optimization. Rebuilding Triton 3.5 with
  triton-lang/triton PR #9035 fixes it on affected Hopper hardware.
- Much later reports show related grounding shifts on Blackwell, while no
  public evidence establishes an Ampere-positive case.

The existing A100 run is a real production-path attempt, not a static smoke:
vLLM 0.11.1 served Qwen3-VL-4B, processed real screenshots, ran
`torch.compile`, captured both FULL and PIECEWISE CUDA Graphs, and answered the
same grounding-style request offline. Its small sign-in target was localized
within one pixel and repeated identically. That is the expected A100 negative
control. The larger Wikipedia target returned no detection because of the 4B
model's capability, so it cannot be scored as the reported coordinate-shift
symptom. Detailed commands, image/model digests, logs, and timings remain in
`docker-build.md`.

Substituting 4B or 8B weights does not overcome the architecture gate: the
public issue already contains a 30B single-H100 reproduction and an A100
non-reproduction. Conversely, a CPU-level or isolated Triton code-generation
test would not measure grounding behavior through the real model/service path.

## Accepted-solution mapping is dependency-level

vLLM PR #30525 says `FIX #29595`, but it is a 17-file project-wide PyTorch 2.10
upgrade rather than an isolated vLLM repair. The issue was later reopened in
practice when vLLM 0.16 rolled that upgrade back for CUDA compatibility.

The narrow, independently confirmed solution is Triton PR #9035,
“Force disable slp-copyable-elements”:

- Triton base: `6213a0e8a7c3d4f2e2983b5afdf89abd2004d585`;
- contributor head: `b5ff928b8fe295814b99bcbd75b7026837bce20d`;
- merged commit: `9844da955a9db14ec69c9aac828ee9803085e288`;
- scope: one file, `python/src/llvm.cc`, `+9/-1`;
- mechanism: disable an LLVM vectorization path that exposes a PTXAS bug.

That mapping is stable enough for a dependency-repair benchmark on affected
hardware, but it does not create an A100 behavioral oracle. On A100 the
released Triton baseline and fixed Triton are both expected to preserve
grounding, so the required Base=0/Oracle=1 separation is absent.

## Hard blocker and restart condition

The task may be restarted only with an affected accelerator and a model/service
fixture that fails deterministically before the dependency patch. The minimum
credible gate is:

1. Hopper H20/H100/H200 (or a separately confirmed affected Blackwell) with
   enough memory for a publicly reproducible Qwen3-VL model;
2. identical offline model, image set, prompts, compilation configuration, and
   inference parameters for Base and Oracle;
3. exact Triton baseline and accepted fixed source/native builds, with compiler
   and loaded-artifact provenance;
4. a quantitative grounding metric over several examples, not a single
   hand-selected coordinate;
5. Base below a fixed accuracy/IoU threshold and Oracle above it, while eager
   mode acts only as a diagnostic control.

Until those conditions are available, the current environment is correctly
classified as `environment/runtime validated, reproduction blocked` and not a
Harbor task.
