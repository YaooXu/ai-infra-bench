ARG BASE_IMAGE=vllm/vllm-openai:v0.14.0@sha256:1d6866b87630d94f5e0cdae55ab5abb4ce0b03fcb84d9d10612f9d518d19d4fd
ARG NATIVE_IMAGE=vllm/vllm-openai:v0.15.1@sha256:8c9aaddfa6011b9651d06834d2fb90bdb9ab6ced4b420ec76925024eb12b22d0

FROM ${NATIVE_IMAGE} AS native-artifacts

USER root
COPY lock/native-donor.sha256 /tmp/native-donor.sha256
RUN set -eux; \
    VLLM_SITE="$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')"; \
    cd "${VLLM_SITE}"; \
    sha256sum -c /tmp/native-donor.sha256; \
    for rel in \
        _C.abi3.so \
        _flashmla_C.abi3.so \
        _flashmla_extension_C.abi3.so \
        _moe_C.abi3.so \
        cumem_allocator.abi3.so \
        vllm_flash_attn/_vllm_fa2_C.abi3.so \
        vllm_flash_attn/_vllm_fa3_C.abi3.so; do \
        test -f "${rel}"; \
        test ! -L "${rel}"; \
        readelf -h "${rel}" >/dev/null; \
        install -D -m 0755 "${rel}" "/opt/native-vllm/${rel}"; \
    done; \
    test "$(find /opt/native-vllm -type f | wc -l)" = 7; \
    rm -f /tmp/native-donor.sha256

FROM ${BASE_IMAGE}

ARG VLLM_BASE_SHA=0118cdcc02ae16a137645e2289bf41f5e3da9d80
ARG VLLM_BASE_TREE=eb2267901a78dcd021c505f5a6bd50ccc6632a9b
ARG VLLM_SOURCE_SHA256=75b2632ec1ea5f92539b9c5f6a7e3cd3357874f04cfd6953c9ae851f1b992957

USER root

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

COPY --from=native-artifacts /opt/native-vllm /opt/native-vllm
COPY lock/final-native.sha256 /tmp/final-native.sha256

# Materialize the exact source cutoff, prove its canonical tree, then create a
# single synthetic benchmark commit. Generated/native artifacts are installed
# only after that commit and remain ignored by Git.
RUN set -eux; \
    apt-get \
        -o Acquire::http::Proxy=false \
        -o Acquire::https::Proxy=false \
        update; \
    DEBIAN_FRONTEND=noninteractive apt-get \
        -o Acquire::http::Proxy=false \
        -o Acquire::https::Proxy=false \
        install -y --no-install-recommends ca-certificates git; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fL \
        --retry 8 \
        --retry-all-errors \
        --retry-delay 3 \
        --connect-timeout 30 \
        --max-time 600 \
        "https://codeload.github.com/vllm-project/vllm/tar.gz/${VLLM_BASE_SHA}" \
        -o /tmp/vllm-source.tar.gz; \
    echo "${VLLM_SOURCE_SHA256}  /tmp/vllm-source.tar.gz" | sha256sum -c -; \
    mkdir -p /workspace/repo; \
    tar -xzf /tmp/vllm-source.tar.gz --strip-components=1 -C /workspace/repo; \
    rm -f /tmp/vllm-source.tar.gz; \
    git -C /workspace/repo init; \
    git -C /workspace/repo config user.name "AI Infra Bench"; \
    git -C /workspace/repo config user.email "noreply@ai-infra-bench.local"; \
    git -C /workspace/repo add -f -A; \
    test "$(git -C /workspace/repo write-tree)" = "${VLLM_BASE_TREE}"; \
    git -C /workspace/repo commit --quiet -m "Synthetic benchmark base"; \
    git -C /workspace/repo branch -M benchmark-base; \
    test "$(git -C /workspace/repo rev-parse 'HEAD^{tree}')" = "${VLLM_BASE_TREE}"; \
    test -z "$(git -C /workspace/repo remote)"; \
    test "$(git -C /workspace/repo rev-list --all --count)" = 1; \
    VLLM_SITE="$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')"; \
    install -D -m 0644 "${VLLM_SITE}/_version.py" /workspace/repo/vllm/_version.py; \
    for rel in \
        _C.abi3.so \
        _flashmla_C.abi3.so \
        _flashmla_extension_C.abi3.so \
        _moe_C.abi3.so \
        cumem_allocator.abi3.so \
        vllm_flash_attn/_vllm_fa2_C.abi3.so \
        vllm_flash_attn/_vllm_fa3_C.abi3.so; do \
        install -D -m 0755 "/opt/native-vllm/${rel}" "/workspace/repo/vllm/${rel}"; \
    done; \
    cd /workspace/repo/vllm; \
    sha256sum -c /tmp/final-native.sha256; \
    test "$(find /workspace/repo/vllm -type f -name '*.so' | wc -l)" = 7; \
    test -z "$(git -C /workspace/repo status --porcelain)"; \
    rm -rf /opt/native-vllm /tmp/final-native.sha256 /root/.bash_history

RUN set -eux; \
    if ! id -u agent >/dev/null 2>&1; then \
        useradd --create-home --shell /bin/bash agent; \
    fi; \
    chown -R agent:agent /workspace; \
    rm -f /home/agent/.bash_history

ENV PYTHONPATH=/workspace/repo \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

USER agent
WORKDIR /workspace/repo
ENTRYPOINT []
CMD ["/bin/bash"]
