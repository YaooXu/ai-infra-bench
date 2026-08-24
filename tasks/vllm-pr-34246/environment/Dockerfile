ARG BASE_IMAGE=vllm/vllm-openai:v0.18.1@sha256:228113d30448941e7a845f57ef0b3d3ea74ffda81be72ded4f8d6dfab0124fe6
ARG STABLE_LIBTORCH_IMAGE=vllm/vllm-openai:v0.19.0@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504
FROM ${STABLE_LIBTORCH_IMAGE} AS stable_libtorch_donor

FROM ${BASE_IMAGE}

ARG VLLM_BASE_SHA=31a719bcd37a195107711dc8b498288e49ef8576
ARG VLLM_BASE_TREE=a45811fe928b168245e519e4205bbd99c6fa3f57
ARG VLLM_SOURCE_SHA256=f248739d9eb5591e5282be31b323a499a3643426221aa1dd8d58caba3baf7f67

USER root

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# Acquire the exact survey base and synthesize one canonical benchmark commit.
# Docker's predefined proxy args are not redeclared, keeping proxy values out
# of image configuration and history.
RUN set -eux; \
    apt-get \
        -o Acquire::http::Proxy=false \
        -o Acquire::https::Proxy=false \
        update; \
    DEBIAN_FRONTEND=noninteractive apt-get \
        -o Acquire::http::Proxy=false \
        -o Acquire::https::Proxy=false \
        install -y --no-install-recommends git; \
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
    test "$(git -C /workspace/repo rev-list --all --count)" = 1

COPY environment/lock/base-native.sha256 /tmp/base-native.sha256
COPY environment/lock/native.sha256 /tmp/native.sha256
COPY --from=stable_libtorch_donor \
    /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so \
    /tmp/native-stage/_C_stable_libtorch.abi3.so

# Bind exact candidate Python to a hash-locked native whitelist. The closest
# pre-cutoff release supplies all but one object; a same-Torch post-cutoff image
# supplies only the explicitly named stable-libtorch object needed by Base.
# No release Python directory is visible, and every object is a regular ELF.
RUN set -eux; \
    VLLM_SITE=/usr/local/lib/python3.12/dist-packages/vllm; \
    cd "${VLLM_SITE}"; \
    sha256sum -c /tmp/base-native.sha256; \
    for rel in \
        _C.abi3.so \
        _flashmla_C.abi3.so \
        _flashmla_extension_C.abi3.so \
        _moe_C.abi3.so \
        cumem_allocator.abi3.so \
        vllm_flash_attn/_vllm_fa2_C.abi3.so \
        vllm_flash_attn/_vllm_fa3_C.abi3.so; do \
        test -f "${VLLM_SITE}/${rel}"; \
        test ! -L "${VLLM_SITE}/${rel}"; \
        readelf -h "${VLLM_SITE}/${rel}" >/dev/null; \
        install -D -m 0755 "${VLLM_SITE}/${rel}" "/workspace/repo/vllm/${rel}"; \
    done; \
    test -f /tmp/native-stage/_C_stable_libtorch.abi3.so; \
    test ! -L /tmp/native-stage/_C_stable_libtorch.abi3.so; \
    readelf -h /tmp/native-stage/_C_stable_libtorch.abi3.so >/dev/null; \
    install -D -m 0755 \
        /tmp/native-stage/_C_stable_libtorch.abi3.so \
        /workspace/repo/vllm/_C_stable_libtorch.abi3.so; \
    install -D -m 0644 "${VLLM_SITE}/_version.py" "/workspace/repo/vllm/_version.py"; \
    cd /workspace/repo/vllm; \
    sha256sum -c /tmp/native.sha256; \
    test "$(find /workspace/repo/vllm -type f -name '*.so' | wc -l)" = 8; \
    test -z "$(git -C /workspace/repo status --porcelain)"; \
    rm -rf \
        /tmp/base-native.sha256 \
        /tmp/native.sha256 \
        /tmp/native-stage \
        /root/.bash_history

RUN set -eux; \
    if ! id -u agent >/dev/null 2>&1; then \
        useradd --create-home --shell /bin/bash agent; \
    fi; \
    chown -R agent:agent /workspace; \
    rm -f /home/agent/.bash_history

COPY --chown=agent:agent environment/public_dev /workspace/public_dev

RUN chmod 755 /workspace/public_dev/run.sh

ENV PYTHONPATH=/workspace/repo \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

USER agent
WORKDIR /workspace/repo
ENTRYPOINT []
CMD ["/bin/bash"]
