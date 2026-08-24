# syntax=docker/dockerfile:1
# Public-source, source-bound build. The build stage may access the public
# dependency sources declared in lock/public-build-manifest.json. Agent and verifier
# runtime networking is disabled by task.toml/Harbor.
ARG BASE_IMAGE=pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel@sha256:97ec2a667dd7560c615bf50a95b2fb85a673ae233a55da1706e8e04e6d6d518e
ARG RUNTIME_BASE_IMAGE=pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime@sha256:f0ca81b440e252399d9954a45b616ee2540959466aacf3dfc3f856691eee66e8

FROM ${BASE_IMAGE} AS build-base
ARG BASE_IMAGE
RUN case "$BASE_IMAGE" in \
      *@sha256:97ec2a667dd7560c615bf50a95b2fb85a673ae233a55da1706e8e04e6d6d518e) ;; \
      *) echo "BASE_IMAGE must use the frozen builder manifest digest" >&2; exit 1 ;; \
    esac

USER 0:0
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.11.2.dev278 \
    VLLM_VERSION_OVERRIDE=0.11.2.dev278

# The digest-pinned devel image already supplies CUDA, Python, PyTorch, GCC,
# make, CMake, Ninja and zlib headers. Public builds deliberately do not invoke
# apt: some controlled builders allow PyPI/GitHub while blocking Ubuntu and
# NVIDIA package repositories. Missing base-toolchain inputs are a hard error.
RUN command -v gcc && command -v g++ && command -v make && \
    command -v cmake && command -v ninja && \
    test -f /usr/include/zlib.h

COPY lock/ /opt/bench/lock/
COPY scripts/ /opt/bench/scripts/
RUN python -I /opt/bench/scripts/verify_artifacts.py \
      --lock-root /opt/bench/lock --mode declarations && \
    python -I /opt/bench/scripts/fetch_base_source.py \
      --manifest /opt/bench/lock/sources.lock.json \
      --output /opt/base-pristine/vllm && \
    python -I /opt/bench/scripts/verify_artifacts.py \
      --lock-root /opt/bench/lock --mode source \
      --source-root /opt/base-pristine/vllm && \
    chmod -R a-w /opt/base-pristine

FROM build-base AS native-builder

# Install the complete public dependency closure into a relocatable venv.
# Python's ensurepip seeds setuptools 79.0.1 in this digest-pinned image. Remove
# that seed first so the locked setuptools wheel is the only installed metadata;
# --ignore-installed would leave the stale seed dist-info beside the locked one.
RUN python -m venv /opt/bench-venv && \
    /opt/bench-venv/bin/python -m pip uninstall --yes setuptools && \
    /opt/bench-venv/bin/python -c \
      "import importlib.util; assert importlib.util.find_spec('setuptools') is None" && \
    /opt/bench-venv/bin/python -m pip install \
      --index-url https://pypi.org/simple \
      --only-binary=:all: \
      --no-cache-dir \
      --retries 5 \
      --require-hashes \
      --timeout 180 \
      -r /opt/bench/lock/requirements.lock && \
    /opt/bench-venv/bin/python -m pip check && \
    /opt/bench-venv/bin/python -I /opt/bench/scripts/verify_artifacts.py \
      --lock-root /opt/bench/lock --mode installed \
      --source-root /opt/base-pristine/vllm

RUN /opt/bench-venv/bin/python -I /opt/bench/scripts/fetch_native_sources.py \
      --manifest /opt/bench/lock/native-build-deps-manifest.json \
      --output /opt/native-src

# Build a source- and hash-pinned Git that supports local worktree operations.
# Network transports and optional language integrations are excluded because
# candidate runtime is no-network. The resulting binary depends only on glibc
# and libz, both supplied by the frozen runtime base.
RUN make -C /opt/native-src/git -j32 \
      prefix=/opt/bench-git GIT_VERSION=2.47.2 \
      NO_CURL=YesPlease NO_EXPAT=YesPlease NO_GETTEXT=YesPlease \
      NO_OPENSSL=YesPlease NO_PERL=YesPlease NO_PYTHON=YesPlease \
      NO_TCLTK=YesPlease NO_INSTALL_HARDLINKS=YesPlease all && \
    make -C /opt/native-src/git \
      prefix=/opt/bench-git GIT_VERSION=2.47.2 \
      NO_CURL=YesPlease NO_EXPAT=YesPlease NO_GETTEXT=YesPlease \
      NO_OPENSSL=YesPlease NO_PERL=YesPlease NO_PYTHON=YesPlease \
      NO_TCLTK=YesPlease NO_INSTALL_HARDLINKS=YesPlease install && \
    test "$(/opt/bench-git/bin/git --version)" = "git version 2.47.2"

# Compilation never runs in the immutable pristine tree.  The build copy is
# verified before compilation; the pristine tree is verified again afterwards.
RUN mkdir -p /opt/build-source && \
    cp -a /opt/base-pristine/vllm /opt/build-source/vllm && \
    chmod -R u+w /opt/build-source/vllm && \
    python -I /opt/bench/scripts/verify_artifacts.py \
      --lock-root /opt/bench/lock --mode source \
      --source-root /opt/build-source/vllm

WORKDIR /opt/build-source/vllm
ENV PATH=/opt/bench-git/bin:/opt/bench-venv/bin:/usr/local/cuda/bin:${PATH} \
    CMAKE_BUILD_TYPE=Release \
    FLASH_MLA_SRC_DIR=/opt/native-src/flashmla \
    MAX_JOBS=32 \
    NVCC_THREADS=2 \
    QUTLASS_SRC_DIR=/opt/native-src/qutlass \
    TORCH_CUDA_ARCH_LIST=8.0 \
    TRITON_KERNELS_SRC_DIR=/opt/native-src/triton/python/triton_kernels/triton_kernels \
    VLLM_CUTLASS_SRC_DIR=/opt/native-src/cutlass \
    VLLM_FLASH_ATTN_SRC_DIR=/opt/native-src/vllm-flash-attention \
    VLLM_TARGET_DEVICE=cuda

RUN python setup.py bdist_wheel --dist-dir /opt/vllm-dist --py-limited-api=cp38
RUN python -I /opt/bench/scripts/verify_artifacts.py \
      --lock-root /opt/bench/lock --mode source \
      --source-root /opt/base-pristine/vllm && \
    python -I /opt/bench/scripts/prepare_runtime_tree.py \
      --pristine-source /opt/base-pristine/vllm \
      --build-source /opt/build-source/vllm \
      --source-manifest /opt/bench/lock/sources.lock.json \
      --wheel-dir /opt/vllm-dist \
      --contract /opt/bench/lock/native-manifest.json \
      --source-binding /opt/bench/lock/native-source-binding.json \
      --runtime-output /opt/runtime/app \
      --overlay-output /opt/runtime/native-overlay \
      --metadata-output /opt/runtime/site-packages \
      --git-exclude-output /opt/runtime/native-runtime.gitignore \
      --manifest-output /opt/runtime/native-build-manifest.json

FROM ${RUNTIME_BASE_IMAGE} AS runtime
ARG RUNTIME_BASE_IMAGE
RUN case "$RUNTIME_BASE_IMAGE" in \
      *@sha256:f0ca81b440e252399d9954a45b616ee2540959466aacf3dfc3f856691eee66e8) ;; \
      *) echo "RUNTIME_BASE_IMAGE must use the frozen runtime manifest digest" >&2; exit 1 ;; \
    esac

USER 0:0
ENV PATH=/opt/bench-git/bin:/opt/bench-venv/bin:/usr/local/cuda/bin:${PATH} \
    HOME=/home/agent \
    HF_HUB_OFFLINE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INDEX=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.11.2.dev278 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_VERSION_OVERRIDE=0.11.2.dev278

COPY --from=native-builder /opt/bench-git/ /opt/bench-git/
COPY --from=native-builder /opt/bench-venv/ /opt/bench-venv/
COPY --from=native-builder /opt/runtime/app/ /app/
COPY --from=native-builder /opt/runtime/site-packages/ /opt/bench-venv/lib/python3.11/site-packages/
COPY --from=native-builder /opt/runtime/native-build-manifest.json /opt/bench/native-build-manifest.json
COPY --from=native-builder /opt/runtime/native-runtime.gitignore /opt/bench/native-runtime.gitignore
COPY lock/ /opt/bench/lock/
COPY scripts/ /opt/bench/scripts/
COPY Dockerfile /opt/bench/Dockerfile
COPY .dockerignore /opt/bench/.dockerignore

# Candidate tooling receives one synthetic base commit and no upstream Git
# objects, generated runtime artifacts, remotes, tags, reflog, hidden tests, or
# solution artifacts.  At this point /app is exactly the pristine public tree.
RUN cd /app && \
    git init -q && \
    git config user.name "AI Infra Benchmark" && \
    git config user.email "benchmark@invalid.example" && \
    git add -f -A && \
    GIT_AUTHOR_DATE="2000-01-01T00:00:00Z" \
    GIT_COMMITTER_DATE="2000-01-01T00:00:00Z" \
      git commit -q -m "sanitized benchmark base" && \
    git reflog expire --expire=now --all && \
    test "$(git rev-list --count HEAD)" = "1" && \
    expected_tree=$(/opt/bench-venv/bin/python -I -c \
      'import json,pathlib; print(json.loads(pathlib.Path("/opt/bench/lock/sources.lock.json").read_text())["git_tree_sha1"])') && \
    test "$(git rev-parse 'HEAD^{tree}')" = "$expected_tree" && \
    test -z "$(git remote)" && \
    test -z "$(git tag)" && \
    test ! -e /tests && \
    test ! -e /solution

# Apply only the source-built, manifest-verified runtime overlay after the
# synthetic source commit.  Exact generated paths are ignored through
# .git/info/exclude; neither broad source globs nor candidate Python files are
# hidden from git status.
COPY --from=native-builder /opt/runtime/native-overlay/ /app/
RUN cat /opt/bench/native-runtime.gitignore >> /app/.git/info/exclude && \
    printf '/app\n' > /opt/bench-venv/lib/python3.11/site-packages/vllm-editable.pth && \
    test "$(git --version)" = "git version 2.47.2" && \
    /opt/bench-venv/bin/python -m pip check && \
    cd /tmp && /opt/bench-venv/bin/python -I /opt/bench/scripts/verify_artifacts.py \
      --lock-root /opt/bench/lock \
      --mode runtime \
      --runtime-root /app \
      --native-build-manifest /opt/bench/native-build-manifest.json && \
    cd /tmp && /opt/bench-venv/bin/python -I -c \
      "import importlib.util,pathlib,torch,vllm; root=pathlib.Path('/app'); assert pathlib.Path(vllm.__file__).resolve().is_relative_to(root); spec=importlib.util.find_spec('vllm._C'); assert spec and pathlib.Path(spec.origin).resolve().is_relative_to(root); assert torch.__version__.startswith('2.9.0')" && \
    cd /app && test -z "$(git status --porcelain=v1 --untracked-files=all)" && \
    /opt/bench-venv/bin/python -I -c \
      "import json,pathlib,subprocess; root=pathlib.Path('/app'); contract=json.loads(pathlib.Path('/opt/bench/lock/native-manifest.json').read_text()); tracked=set(subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()); generated={x['path'] for x in contract['artifacts']}; overlap=tracked & generated; assert not overlap, sorted(overlap)" && \
    test ! -e /app/build && test ! -e /app/dist && \
    printf 'runtime-size ' && du -sh /app && \
    printf 'synthetic-git-size ' && du -sh /app/.git && \
    printf 'build-tree-size 0 (absent) /app/build\n' && \
    git count-objects -vH && \
    find /app -xdev -type f ! -path '/app/.git/*' -printf '%s %p\n' \
      | sort -nr | head -n 30

RUN groupadd --gid 10001 agent && \
    useradd --uid 10001 --gid 10001 --create-home --home-dir /home/agent agent && \
    chown -R agent:agent /app && \
    chmod -R a-w /opt/bench && \
    mkdir -p /tmp/verifier-home && \
    chown agent:agent /tmp/verifier-home

WORKDIR /app
USER 10001:10001
CMD ["python", "-I", "-c", "import torch,vllm,vllm._C; print(vllm.__file__); print(torch.__version__); print(torch.version.cuda)"]
