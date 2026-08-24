# syntax=docker/dockerfile:1.7

FROM alpine/git:2.49.1@sha256:c0280cf9572316299b08544065d3bf35db65043d5e3963982ec50647d2746e26 AS git_runtime
RUN tar -czf /git-runtime.tar \
    /usr/bin/git \
    /usr/libexec/git-core \
    /usr/share/git-core \
    /lib/ld-musl-x86_64.so.1 \
    /usr/lib/libpcre2-8.so.0.14.0 \
    /usr/lib/libz.so.1.3.1

# v0.10.0 is the first official release after the PR base (2025-07-23).
# Its toolchain matches the base requirement (torch 2.7.1/CUDA).  The complete
# candidate extension _C and all Python source below come from SOURCE_ARCHIVE;
# unrelated native extensions are explicitly labeled same-version donors.
FROM vllm/vllm-openai:v0.10.0@sha256:05a31dc4185b042e91f4d2183689ac8a87bd845713d5c3f987563c5899878271

ARG SOURCE_ARCHIVE=vllm-base.tar.gz
ARG VLLM_SOURCE_COMMIT=14bf19e39f601163265b7c7d58d972b8a83d8896
ARG VLLM_SOURCE_SHA256=ef6429c4011bb3955ac9249833aa0a82f36b7855dcad7ac556deb4849d5f3339
ARG CUTLASS_SHA256=44a121c5878827875856c175ebe82e955062e37cd61fcdf31ebe2e8874f2fc5c
ARG FLASH_ATTN_SHA256=cca19d7e53af08aa6d6f0c4fd9dd78d30314497e38fb03b1368b3d5a77ab4b5c
ARG FLASH_ATTN_CUTLASS_SHA256=78816d6c6d97793b5b59ef2a702174cb85b78dfcefc8fe2489964de2e42f17d2
ARG FLASH_MLA_SHA256=42ebf11c1a4cd17c4221705de5cff4fcaedf29a9dbdfcb8f2459dc550faf66f7

LABEL ai-infra-bench.vllm-source-commit="${VLLM_SOURCE_COMMIT}" \
      ai-infra-bench.vllm-native-origin="_C=exact-source:${VLLM_SOURCE_COMMIT};other=v0.10.0-donor" \
      ai-infra-bench.vllm-native-scope="candidate-path:_C" \
      ai-infra-bench.cuda-arches="8.0"

USER root

COPY --from=git_runtime /git-runtime.tar /tmp/git-runtime.tar
RUN tar -xzf /tmp/git-runtime.tar -C / \
    && rm /tmp/git-runtime.tar \
    && ln -s libpcre2-8.so.0.14.0 /usr/lib/libpcre2-8.so.0 \
    && ln -s libz.so.1.3.1 /usr/lib/libz.so.1 \
    && git --version

COPY ${SOURCE_ARCHIVE} /tmp/vllm-source.tar.gz
COPY cutlass-v4.0.0.tar.gz /tmp/cutlass.tar.gz
COPY flash-attn-1c2624e.tar.gz /tmp/flash-attn.tar.gz
COPY flash-attn-cutlass-62750a2.tar.gz /tmp/flash-attn-cutlass.tar.gz
COPY flashmla-575f772.tar.gz /tmp/flashmla.tar.gz

# Build with no package-index or FetchContent access.  CMake dependencies and
# the FlashAttention submodule are immutable, digest-checked archives.  sm80 is
# the A100 gate and avoids compiling unrelated GPU architectures.
ENV PIP_NO_INDEX=1 \
    VLLM_TARGET_DEVICE=cuda \
    TORCH_CUDA_ARCH_LIST=8.0 \
    MAX_JOBS=8 \
    NVCC_THREADS=2 \
    VLLM_CUTLASS_SRC_DIR=/opt/vllm-deps/cutlass \
    VLLM_FLASH_ATTN_SRC_DIR=/opt/vllm-deps/flash-attn \
    FLASH_MLA_SRC_DIR=/opt/vllm-deps/flashmla \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.10.0.dev0

RUN set -eu; \
    printf '%s  %s\n' "${VLLM_SOURCE_SHA256}" /tmp/vllm-source.tar.gz | sha256sum -c -; \
    printf '%s  %s\n' "${CUTLASS_SHA256}" /tmp/cutlass.tar.gz | sha256sum -c -; \
    printf '%s  %s\n' "${FLASH_ATTN_SHA256}" /tmp/flash-attn.tar.gz | sha256sum -c -; \
    printf '%s  %s\n' "${FLASH_ATTN_CUTLASS_SHA256}" /tmp/flash-attn-cutlass.tar.gz | sha256sum -c -; \
    printf '%s  %s\n' "${FLASH_MLA_SHA256}" /tmp/flashmla.tar.gz | sha256sum -c -; \
    mkdir -p /workspace/repo /opt/vllm-canonical-source /opt/vllm-deps/cutlass /opt/vllm-deps/flash-attn /opt/vllm-deps/flashmla; \
    tar -xzf /tmp/vllm-source.tar.gz --strip-components=1 -C /workspace/repo; \
    tar -xzf /tmp/vllm-source.tar.gz --strip-components=1 -C /opt/vllm-canonical-source; \
    tar -xzf /tmp/cutlass.tar.gz --strip-components=1 -C /opt/vllm-deps/cutlass; \
    tar -xzf /tmp/flash-attn.tar.gz --strip-components=1 -C /opt/vllm-deps/flash-attn; \
    tar -xzf /tmp/flash-attn-cutlass.tar.gz --strip-components=1 -C /opt/vllm-deps/flash-attn/csrc/cutlass; \
    tar -xzf /tmp/flashmla.tar.gz --strip-components=1 -C /opt/vllm-deps/flashmla; \
    rm /tmp/vllm-source.tar.gz /tmp/cutlass.tar.gz /tmp/flash-attn.tar.gz /tmp/flash-attn-cutlass.tar.gz /tmp/flashmla.tar.gz; \
    test ! -e /workspace/repo/.git; \
    test "$(find /opt/vllm-canonical-source -type f | wc -l)" -gt 1000; \
    diff -qr /opt/vllm-canonical-source /workspace/repo; \
    donor_pkg="$(python3 -c 'import importlib.metadata; print(importlib.metadata.distribution("vllm").locate_file("vllm"))')"; \
    mkdir -p /opt/vllm-donor-native; \
    cd "$donor_pkg"; \
    find . -type f -name '*.so' -exec cp --parents '{}' /opt/vllm-donor-native/ \;; \
    cd /workspace/repo; \
    cp setup.py /tmp/vllm-setup.py.canonical; \
    sed -i '/^package_data = {/i ext_modules = [CMakeExtension(name="vllm._C")]\n' setup.py; \
    python3 -m pip install --no-build-isolation --no-deps -e /workspace/repo; \
    mv /tmp/vllm-setup.py.canonical setup.py; \
    python3 -c "import importlib.util,pathlib,vllm; p=pathlib.Path(vllm.__file__).resolve(); s=importlib.util.find_spec('vllm._C'); print(p,s.origin if s else None); assert str(p).startswith('/workspace/repo/vllm/') and s is not None and str(s.origin).startswith('/workspace/repo/vllm/')"; \
    find /opt/vllm-canonical-source -type f -print0 | while IFS= read -r -d '' f; do rel="${f#/opt/vllm-canonical-source/}"; cmp "$f" "/workspace/repo/$rel"; done; \
    rm -rf /opt/vllm-canonical-source; \
    cd /opt/vllm-donor-native; \
    find . -type f -name '*.so' ! -name '_C*.so' -exec cp --no-clobber --parents '{}' /workspace/repo/vllm/ \;; \
    rm -rf /opt/vllm-donor-native; \
    cd /workspace/repo; \
    git init /workspace/repo; \
    git -C /workspace/repo config user.name "AI Infra Bench"; \
    git -C /workspace/repo config user.email "noreply@ai-infra-bench.local"; \
    git -C /workspace/repo add -f -A; \
    git -C /workspace/repo commit -m "Synthetic benchmark base"; \
    git -C /workspace/repo branch -M benchmark-base; \
    test -z "$(git -C /workspace/repo remote)"; \
    test "$(git -C /workspace/repo rev-list --all --count)" = 1; \
    test -z "$(git -C /workspace/repo status --porcelain)"; \
    useradd --create-home --uid 1000 --shell /bin/bash agent; \
    chown -R agent:agent /workspace/repo /home/agent

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/home/agent

WORKDIR /workspace/repo
USER agent

RUN test "$(id -un)" = agent \
    && test -z "$(git remote)" \
    && test "$(git rev-list --all --count)" = 1 \
    && test -z "$(git status --porcelain)"

ENTRYPOINT []
CMD ["sleep", "infinity"]
