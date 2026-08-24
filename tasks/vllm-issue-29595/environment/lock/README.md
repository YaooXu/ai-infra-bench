# Environment lock for vllm-project/vllm#29595

## Upstream evidence

- Issue: <https://github.com/vllm-project/vllm/issues/29595>
- Reported baseline: vLLM 0.11.1, Python 3.12.12, torch 2.9.0+cu129,
  transformers 4.57.1, triton 3.5.0, flashinfer-python 0.5.2.
- Reported hardware: 8 x NVIDIA H20. A follow-up in the issue reports that the
  same Qwen3-VL workload retained grounding accuracy on A100 and degraded on
  H100. Therefore A100 is a valid environment/build smoke target, but not a
  positive reproduction platform for the original symptom.
- The issue was closed via vLLM PR #30525, which upgrades PyTorch to 2.10.
  This indicates an upstream PyTorch-version dependency rather than a known
  small vLLM source-only patch.

## Immutable inputs

- Official base image: `vllm/vllm-openai:v0.11.1`
- Base manifest digest:
  `sha256:d5b12dfb74d605615f8b29ebafaa52294c118bcac7bc9e941785c4108fdb913a`
- Git helper image: `alpine/git:2.49.1`
- Git helper manifest digest:
  `sha256:c0280cf9572316299b08544065d3bf35db65043d5e3963982ec50647d2746e26`
- vLLM release source commit:
  `439368496db48d8f992ba8c606a0c0b1eebbfa69`
- GitHub source archive SHA-256:
  `b36c716c28bfd08359e566db92775d3621bcf31a1fd18505f78cb94128a22972`
- Validated A100 cache:
  `/data/ai-infra-bench/survey-builds/vllm-issue-29595/vllm-source.tar.gz`
- Runtime model cache: `/data/yinchen/vllm-mm-cache-case/canary/assets/Qwen3-VL-4B-Instruct`
- Grounding image cache:
  `/data/yinchen/ai-infra-bench-upstream-review-qwen3vl/tasks/qwen3vl-grounding-cudagraph/environment/grounding/test_image.png`

The model and image are mounted read-only at runtime and are not copied into
the image. Their measured byte sizes and content digests are recorded in
`validation/docker-build.md` after validation.

## Preparing the source cache

Before building, place this pinned archive beside the Dockerfile and verify
it against `lock/vllm-source.sha256`:

```bash
curl --retry 10 --retry-all-errors -fL \
  -o environment/vllm-source.tar.gz \
  https://codeload.github.com/vllm-project/vllm/tar.gz/439368496db48d8f992ba8c606a0c0b1eebbfa69
cd environment
sha256sum -c lock/vllm-source.sha256
```

The archive is a generated build input and is not intended to be committed.
The Dockerfile verifies the same digest before extraction.

## Network boundary

After the base image and source archive have been prepared, Docker build does
not require network access. The registry/download proxy is host-owned and is
not present in the Dockerfile or image history. Runtime is executed with
`--network none`; Hugging Face, datasets, pip, and telemetry offline flags are
also set in the image.

## Source/runtime binding

The official wheel supplies the release-matching compiled CUDA extensions.
The pinned Python source is placed at `/workspace/repo`; a `.pth` file
prepends that directory, and `cp --no-clobber` fills only wheel-generated or
wheel-only runtime files such as native extensions, `_version.py`, and the
packaged `vllm_flash_attn` wrappers. The build fails unless `vllm.__file__`
resolves underneath the source tree and the `vllm._C` module is discoverable;
actual native loading is verified later with a GPU because Docker build does
not expose `libcuda.so.1`.

The GitHub archive contains no Git metadata or remotes. After the full runtime
tree is assembled, the image creates exactly one synthetic commit on
`benchmark-base`, with a clean status and zero remotes. The non-root `agent`
(UID 1000) owns the worktree and repository. Git itself comes from a
digest-pinned helper stage because the official runtime omits it; the final
runtime base remains the official vLLM image.
