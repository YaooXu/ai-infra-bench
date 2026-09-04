# vLLM Harbor all-in-one Dockerfile template

Each vLLM task keeps its own self-contained `environment/Dockerfile`. The
generated files install a reproducible CPU or CUDA-native environment. Their
frozen inputs are the base commit and dependency cutoff read from `task.toml`.
Tasks with `[environment].gpus > 0` use `Dockerfile.gpu`; CPU tasks use
`Dockerfile`.

GPU tasks must additionally declare digest-pinned runtime and PyTorch wheel
families:

```toml
[metadata]
gpu_base_image = "pytorch/pytorch:<cuda-devel-tag>@sha256:<digest>"
torch_backend = "cu128"

[environment]
gpus = 1
gpu_types = ["A100", "H100", "H20"]
```

The GPU runtime image must contain a CUDA development toolchain and a CUDA
PyTorch build compatible with the target vLLM commit. The generated image builds
vLLM's CUDA extensions from the exact pre-PR source. Image build success is not
GPU evidence because normal Docker builds do not expose a device; Base/Oracle
and verifier controls must run with `--gpus` and record the observed device,
compute capability, driver/runtime, and candidate source path.

Generate one or more task Dockerfiles from the repository root:

```bash
python3 templates/vllm-harbor-all-in-one/generate.py \
  tasks/vllm-tool-argument-union \
  tasks/vllm-kv-admission-thrashing
```

Verify that checked-in Dockerfiles still match the template and task metadata:

```bash
python3 templates/vllm-harbor-all-in-one/generate.py --check tasks/vllm-*
```

Edit the template, not a generated Dockerfile. The generated Dockerfiles do not
refer back to this directory and can therefore be distributed with an
individual Harbor task.

Build and retain a task image under its canonical local tag:

```bash
python3 templates/vllm-harbor-all-in-one/build.py \
  tasks/vllm-pyav-target-frame-selection
```

The builder sends an empty context to BuildKit, so task tests, solutions, and
curator files cannot enter the image. It writes `environment/image-manifest.json`
after a successful build and does not remove the canonical image tag. Dynamic
Buildx attestations are disabled so cache-only rebuilds keep the same image
digest; provenance is recorded in the checked-in manifest instead.

When `environment/lock/requirements.txt` exists, the generator embeds the
complete exact-version requirements text into the generated Dockerfile. The lock
file is provenance input, not a build-time dependency: the generated Dockerfile
still builds by itself from an empty context.

A task that needs real tokenizer or template behavior may define
`runtime_asset_repository`, `runtime_asset_revision`, `runtime_asset_path`, and
`runtime_asset_files` in `[metadata]`. The revision must be a full commit SHA and
the generated Dockerfile downloads only the named files. Model tensor suffixes
are rejected both while generating the Dockerfile and while building the image.

A task that needs one public reproduction fixture may instead define
`runtime_file_url`, `runtime_file_sha256`, `runtime_file_path`,
`runtime_file_license`, and `runtime_file_attribution`. The generated Dockerfile
downloads the HTTPS resource, rejects content whose SHA-256 differs, and writes
an attribution file beside it.

Cache reuse is split by trust boundary:

- source-independent OCI layers and package downloads are shared;
- C/C++ compiler output and CMake FetchContent are keyed by the full base SHA
  plus dependency-lock digest;
- dependency resolution excludes packages newer than the base commit timestamp;
- no remote build cache is imported.

The image build also rejects a pre-existing installed `vllm` distribution and
checks that the final editable import resolves inside `/workspace/vllm`.
