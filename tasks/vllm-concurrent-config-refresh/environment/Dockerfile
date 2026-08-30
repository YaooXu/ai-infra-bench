# syntax=docker/dockerfile:1
# Template input: the base-SHA token below is replaced from task.toml by generate.py.
# Generated task Dockerfiles are self-contained and differ only in this SHA.

ARG VLLM_BASE_SHA=910cc8543a6907c9cc87c417f8f2420969278bf5
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git

######################### PINNED SOURCE #########################
FROM alpine:3.20 AS source
ARG VLLM_BASE_SHA
ARG VLLM_REPO
RUN test -n "$VLLM_BASE_SHA" \
 && sed -i 's|dl-cdn.alpinelinux.org|mirrors.aliyun.com|g' /etc/apk/repositories \
 && apk add --no-cache ca-certificates git \
 && mkdir -p /src/vllm \
 && git -C /src/vllm init -q \
 && git -C /src/vllm config core.logAllRefUpdates false \
 && git -C /src/vllm remote add origin "$VLLM_REPO" \
 && git -C /src/vllm fetch --no-tags origin "$VLLM_BASE_SHA" \
 && git -C /src/vllm checkout -q -b main "$VLLM_BASE_SHA" \
 && test "$(git -C /src/vllm rev-parse HEAD)" = "$VLLM_BASE_SHA" \
 && git -C /src/vllm remote remove origin \
 && rm -rf /src/vllm/.git/logs \
 && rm -f /src/vllm/.git/FETCH_HEAD /src/vllm/.git/ORIG_HEAD \
 && test "$(git -C /src/vllm branch --show-current)" = main \
 && test "$(git -C /src/vllm rev-list --all --count)" -gt 1 \
 && test ! -e /src/vllm/.git/shallow \
 && test -z "$(git -C /src/vllm remote)" \
 && test -z "$(git -C /src/vllm for-each-ref --format='%(refname)' refs/remotes refs/tags)" \
 && test ! -e /src/vllm/.git/logs \
 && test ! -e /src/vllm/.git/FETCH_HEAD \
 && test ! -e /src/vllm/.git/ORIG_HEAD \
 && test -z "$(git -C /src/vllm fsck --full --no-reflogs --unreachable --no-progress 2>/dev/null)" \
 && test -z "$(git -C /src/vllm status --porcelain)"

######################### ALL-IN-ONE CPU ENVIRONMENT #########################
FROM python:3.12-slim-trixie
ARG VLLM_BASE_SHA
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=500 \
    UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
    UV_INDEX_STRATEGY=unsafe-best-match \
    CARGO_HOME=/root/.cargo \
    RUSTUP_HOME=/root/.rustup \
    PATH=/root/.cargo/bin:$PATH \
    CARGO_BUILD_JOBS=4 \
    CARGO_INCREMENTAL=0 \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    CCACHE_DIR=/root/.cache/ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    MAX_JOBS=8

RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g; s|http://deb.debian.org/debian-security|https://mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates ccache curl ffmpeg \
      git jq libgl1 libnuma-dev libsm6 libssl-dev libtcmalloc-minimal4 \
      libxext6 lsof make pkg-config sudo unzip wget xz-utils \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      --index-url https://mirrors.aliyun.com/pypi/simple uv==0.11.28

ENV RUSTUP_DIST_SERVER=https://rsproxy.cn \
    RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain none \
 && rustup toolchain install 1.95 --profile minimal --component rustfmt,clippy \
 && rustup default 1.95

RUN case "$TARGETARCH" in \
      amd64) nextest_platform=linux ;; \
      arm64) nextest_platform=linux-arm ;; \
      *) echo "Unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac \
 && curl --http1.1 -L --retry 10 --retry-all-errors --connect-timeout 30 --max-time 600 \
      --proto '=https' --tlsv1.2 -sSf \
      "https://get.nexte.st/0.9.143/$nextest_platform" \
      -o /tmp/cargo-nextest.tar.gz \
 && tar -xzf /tmp/cargo-nextest.tar.gz -C /root/.cargo/bin \
 && rm -f /tmp/cargo-nextest.tar.gz \
 && cargo nextest --version

COPY --from=source /src/vllm/ /workspace/vllm/
WORKDIR /workspace/vllm
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+g${VLLM_BASE_SHA} \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.0.0+g${VLLM_BASE_SHA}

# All five task classes use the same environment: CPU-native vLLM, Python test
# tooling, and the Rust toolchain. Repositories predating rust/Cargo.toml simply
# skip the Rust workspace prebuild while retaining the installed toolchain.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/workspace/vllm/.deps,sharing=locked \
    uv pip install --system -r requirements/cpu.txt --torch-backend cpu \
 && uv pip install --system \
      'cmake>=3.26.1' ninja 'packaging>=24.2' \
      'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8' \
      'setuptools-rust>=1.9.0' wheel 'jinja2>=3.1.6' regex \
 && VLLM_TARGET_DEVICE=cpu \
      uv pip install --system --no-build-isolation -e . --torch-backend cpu \
 && uv pip install --system pytest pytest-asyncio pytest-forked \
      pytest-rerunfailures pytest-shard pytest-timeout pytest-cov tblib \
 && if [ -f tests/vllm_test_utils/setup.py ]; then \
      uv pip install --system -e tests/vllm_test_utils; \
    fi

RUN if [ -f rust/Cargo.toml ]; then \
      if [ -x tools/install_protoc.sh ]; then tools/install_protoc.sh; fi; \
      cargo fetch --manifest-path rust/Cargo.toml --locked; \
    fi

# Keep .git and every commit reachable from the pinned base on local main.
# Remove fetch-only metadata and reject any hidden, unreachable Git objects.
RUN find . -type d -name __pycache__ -prune -exec rm -rf {} + \
 && find . -type d \( -name '*.egg-info' -o -name '.pytest_cache' \) -prune -exec rm -rf {} + \
 && test "$(git rev-parse HEAD)" = "$VLLM_BASE_SHA" \
 && test "$(git branch --show-current)" = main \
 && test "$(git rev-list --all --count)" -gt 1 \
 && test ! -e .git/shallow \
 && test -z "$(git remote)" \
 && test -z "$(git for-each-ref --format='%(refname)' refs/remotes refs/tags)" \
 && test ! -e .git/logs \
 && test ! -e .git/FETCH_HEAD \
 && test ! -e .git/ORIG_HEAD \
 && test -z "$(git fsck --full --no-reflogs --unreachable --no-progress 2>/dev/null)" \
 && test -z "$(git status --porcelain)"

LABEL ai.infra.bench.environment-template=vllm-harbor-all-in-one
ENTRYPOINT []
CMD ["sleep", "infinity"]
