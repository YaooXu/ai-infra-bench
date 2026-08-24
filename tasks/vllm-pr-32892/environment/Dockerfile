# syntax=docker/dockerfile:1

ARG BASE_IMAGE=vllm/vllm-openai@sha256:1d6866b87630d94f5e0cdae55ab5abb4ce0b03fcb84d9d10612f9d518d19d4fd
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG VLLM_BASE_SHA=dc917cceb877dfd13f98c538c4c96158047d98bd
ARG VLLM_SOURCE_SHA256=de9e3f1e4782c9ff5c01bbc4f2daaeaa16f52bfe0ddfca2843b0ed4ecf849421
ARG VLLM_SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/dc917cceb877dfd13f98c538c4c96158047d98bd
ARG CUTLASS_SOURCE_SHA256=a4513ba33ae82fd754843c6d8437bee1ac71a6ef1c74df886de2338e3917d4df
ARG CUTLASS_SOURCE_URL=https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v4.2.1
ARG CMAKE_WHEEL_SHA256=2f766bb46367e5e0559fa33184653754bce044583a06014dcaebf8e6dff8a1f1
ARG CMAKE_WHEEL_URL=https://files.pythonhosted.org/cmake-3.31.10.whl
ARG CMAKE_WHEEL_FILE=cmake-3.31.10-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
ARG GIT_DEBS_SHA256=b33d71799bececf4aec897980fcbcae2d3c4da3e5cda2a0f6fe4be70139b761b
ARG GIT_DEBS_URL=https://packages.invalid/git-debs.tar.gz

USER 0:0
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_TARGET_DEVICE=cuda \
    TORCH_CUDA_ARCH_LIST=8.0 \
    VLLM_CUTLASS_SRC_DIR=/opt/cutlass \
    MAX_JOBS=8 \
    NVCC_THREADS=2 \
    HOME=/home/agent

RUN case "${BASE_IMAGE}" in \
      *@sha256:1d6866b87630d94f5e0cdae55ab5abb4ce0b03fcb84d9d10612f9d518d19d4fd) ;; \
      *) echo "BASE_IMAGE must be the frozen v0.14.0 linux/amd64 manifest" >&2; exit 1 ;; \
    esac

# All network inputs are byte-locked. Validation serves them from A100
# loopback, so proxy credentials never enter build arguments or image layers.
RUN set -eu; \
    curl --fail --location --retry 5 --connect-timeout 30 \
      "${VLLM_SOURCE_URL}" -o /tmp/vllm-source.tar.gz; \
    curl --fail --location --retry 5 --connect-timeout 30 \
      "${CUTLASS_SOURCE_URL}" -o /tmp/cutlass-source.tar.gz; \
    curl --fail --location --retry 5 --connect-timeout 30 \
      "${CMAKE_WHEEL_URL}" -o "/tmp/${CMAKE_WHEEL_FILE}"; \
    curl --fail --location --retry 5 --connect-timeout 30 \
      "${GIT_DEBS_URL}" -o /tmp/git-debs.tar.gz; \
    printf '%s  %s\n' "${VLLM_SOURCE_SHA256}" /tmp/vllm-source.tar.gz \
      | sha256sum --check --strict; \
    printf '%s  %s\n' "${CUTLASS_SOURCE_SHA256}" /tmp/cutlass-source.tar.gz \
      | sha256sum --check --strict; \
    printf '%s  %s\n' "${CMAKE_WHEEL_SHA256}" "/tmp/${CMAKE_WHEEL_FILE}" \
      | sha256sum --check --strict; \
    printf '%s  %s\n' "${GIT_DEBS_SHA256}" /tmp/git-debs.tar.gz \
      | sha256sum --check --strict; \
    mkdir -p /tmp/git-debs /app /opt/cutlass; \
    tar -xzf /tmp/git-debs.tar.gz -C /tmp/git-debs; \
    dpkg -i \
      /tmp/git-debs/liberror-perl_0.17029-1_all.deb \
      /tmp/git-debs/git-man_1%3a2.34.1-1ubuntu1.17_all.deb \
      /tmp/git-debs/git_1%3a2.34.1-1ubuntu1.17_amd64.deb; \
    python3 -m pip install --no-index --no-deps "/tmp/${CMAKE_WHEEL_FILE}"; \
    tar -xzf /tmp/vllm-source.tar.gz -C /app --strip-components=1; \
    tar -xzf /tmp/cutlass-source.tar.gz -C /opt/cutlass --strip-components=1; \
    rm -rf /tmp/git-debs /tmp/vllm-source.tar.gz \
      /tmp/cutlass-source.tar.gz "/tmp/${CMAKE_WHEEL_FILE}" \
      /tmp/git-debs.tar.gz; \
    test "$(cmake --version | sed -n '1s/.* //p')" = 3.31.10; \
    test "$(git --version | sed 's/git version //')" = 2.34.1

# Reconstruct a one-commit task repository and prove its tree is the canonical
# Git tree of VLLM_BASE_SHA. No upstream remotes, branches or tags survive.
RUN set -eu; \
    cd /app; \
    git init -q; \
    git config user.name 'AI Infra Benchmark'; \
    git config user.email 'benchmark@invalid.example'; \
    git add -f -A; \
    GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' \
    GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' \
      git commit -q -m 'sanitized benchmark base'; \
    git reflog expire --expire=now --all; \
    test "$(git rev-list --count HEAD)" = 1; \
    test "$(git rev-parse 'HEAD^{tree}')" = 89beecb205e031cbb82e2eea9d2cd0f350135b8c; \
    test -z "$(git remote)"; \
    test -z "$(git tag)"

COPY source_gate.py rebuild_native.sh runtime_smoke.py reproduce_baseline.py /opt/bench/
COPY lock/ /opt/bench/lock/

RUN set -eu; \
    python3 -I /opt/bench/source_gate.py /app; \
    chmod 0555 /opt/bench/rebuild_native.sh; \
    /opt/bench/rebuild_native.sh; \
    native="$(find /app/vllm -maxdepth 1 -type f -name '_moe_C*.so' -print -quit)"; \
    test -n "${native}"; \
    /usr/local/cuda/bin/cuobjdump --list-elf "${native}" | grep -q 'sm_80'; \
    printf '/build/\n/.deps/\n/vllm/_moe_C*.so\n' >> /app/.git/info/exclude; \
    test -z "$(cd /app && git status --porcelain=v1 --untracked-files=all)"

# Candidate Python and the exact-source native extension both resolve under
# /app. Other release extensions are not copied into the candidate tree.
RUN set -eu; \
    site_dir="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    printf 'import sys; sys.path.insert(0, "/app")\n' \
      > "${site_dir}/vllm-candidate-source.pth"; \
    cd /tmp; \
    python3 -I -c 'import importlib.util,pathlib,torch,vllm; root=pathlib.Path("/app"); assert pathlib.Path(vllm.__file__).resolve().is_relative_to(root); spec=importlib.util.find_spec("vllm._moe_C"); assert spec and pathlib.Path(spec.origin).resolve().is_relative_to(root); torch.ops.load_library(spec.origin); assert torch.ops._moe_C.moe_permute_unpermute_supported()'; \
    test -z "$(cd /app && git status --porcelain=v1 --untracked-files=all)"

RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/agent agent \
    && chown -R agent:agent /app /home/agent \
    && chmod -R a-w /opt/bench /opt/cutlass

WORKDIR /app
USER 10001:10001
ENTRYPOINT []
CMD ["python3", "-I", "/opt/bench/runtime_smoke.py"]
