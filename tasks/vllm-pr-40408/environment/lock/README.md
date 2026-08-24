# Locked sources and hardware eligibility

This task corresponds to vLLM PR 40408 at pre-PR base commit
`ea0e501bb18c12b80acc05ff8c7f013db515ba80` (Git tree
`f3a7898db0c5cf56d04ae58070633342bfae3856`). The exact GitHub codeload
archive is locked by byte size and SHA-256 in `environment.json` and verified
again during every image build. A codeload archive has no `.git` directory, so
no upstream history, tags, branches, or remotes enter the image.

The base source pins PyTorch 2.11.0 and CUDA 13.0.2. The nearest official
release image is vLLM v0.20.0, published one week after the base commit. Its
linux/amd64 manifest, vLLM, PyTorch and CUDA identities are all locked. The
release image is used only as a CUDA/toolchain and hardware-probe base.

This PR changes native Cutlass FP8 dispatch. The official v0.20.0 shared
objects are therefore not copied into `/app` and are not described as candidate
extensions. They remain under site-packages solely so the hardware probe can
ask the installed vLLM platform layer whether the visible GPU supports FP8.
The probe reports both paths and explicitly records that exact-base native
binding is false.

Upstream's own `requires_fp8` gate requires Ada or Hopper, compute capability
8.9 or newer. The assigned A100 is SM80 (8.0), so it cannot execute this
regression. The built artifact is only a source-integrity and hardware-
eligibility probe, not an agent environment or a verifier-ready benchmark.
Exact native compilation and the real batch-invariance baseline must be
performed on eligible hardware, preferably SM90/H100 because the PR benchmark
and most new Cutlass 3.x dispatch are Hopper-oriented.
