ARG BASE_IMAGE=vllm/vllm-openai:v0.20.2@sha256:70a098d90dbab428a001d9e852fc0fc8d67da5beb03e7851a22247653bf35923
FROM ${BASE_IMAGE}

ARG VLLM_BASE_SHA=c7560af42487b1570c4e6f4cea5df1605a4d59fc
ARG VLLM_BASE_TREE=00dedbf38a187d7be24007ab92ce0158c646d024
ARG VLLM_SOURCE_SHA256=b88f58e5acd39d0b0c98e008635d48fcde8ed1b0d4d6c698a9b4fa59aa44352f

USER root

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

COPY lock/native.sha256 /tmp/native.sha256

# Acquire the exact survey base and construct one canonical synthetic commit.
# Docker's predefined proxy args are deliberately not redeclared, so proxy
# values are not persisted in image configuration or history.
RUN set -eux; \
    apt-get \
        -o Acquire::http::Proxy=false \
        -o Acquire::https::Proxy=false \
        update; \
    DEBIAN_FRONTEND=noninteractive apt-get \
        -o Acquire::http::Proxy=false \
        -o Acquire::https::Proxy=false \
        install -y --no-install-recommends \
        git=1:2.34.1-1ubuntu1.17 \
        git-man=1:2.34.1-1ubuntu1.17 \
        liberror-perl=0.17029-1; \
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

# Bind the exact Python source to an explicit, hash-locked whitelist of native
# and generated artifacts from the closest pre-cutoff official release. The
# release package itself is never added to PYTHONPATH.
RUN set -eux; \
    VLLM_SITE=/usr/local/lib/python3.12/site-packages/vllm; \
    cd "${VLLM_SITE}"; \
    sha256sum -c /tmp/native.sha256; \
    for rel in \
        _C.abi3.so \
        _C_stable_libtorch.abi3.so \
        _flashmla_C.abi3.so \
        _flashmla_extension_C.abi3.so \
        _moe_C.abi3.so \
        cumem_allocator.abi3.so \
        third_party/deep_gemm/_C.cpython-312-x86_64-linux-gnu.so \
        vllm_flash_attn/_vllm_fa2_C.abi3.so \
        vllm_flash_attn/_vllm_fa3_C.abi3.so; do \
        test -f "${VLLM_SITE}/${rel}"; \
        test ! -L "${VLLM_SITE}/${rel}"; \
        readelf -h "${VLLM_SITE}/${rel}" >/dev/null; \
        install -D -m 0755 "${VLLM_SITE}/${rel}" "/workspace/repo/vllm/${rel}"; \
    done; \
    install -D -m 0644 "${VLLM_SITE}/_version.py" "/workspace/repo/vllm/_version.py"; \
    cd /workspace/repo/vllm; \
    sha256sum -c /tmp/native.sha256; \
    test "$(find /workspace/repo/vllm -type f -name '*.so' | wc -l)" = 9; \
    test -z "$(git -C /workspace/repo status --porcelain)"; \
    rm -f /tmp/native.sha256 /root/.bash_history

RUN set -eux; \
    if ! id -u agent >/dev/null 2>&1; then \
        useradd --create-home --shell /bin/bash agent; \
    fi; \
    PYTHONPATH=/workspace/repo python3 -c 'import importlib.util,pathlib,subprocess,vllm; root=pathlib.Path("/workspace/repo").resolve(); src=pathlib.Path(vllm.__file__).resolve(); native=pathlib.Path(importlib.util.find_spec("vllm._C").origin).resolve(); assert src.is_relative_to(root), src; assert native.is_relative_to(root), native; assert not subprocess.check_output(["git", "-C", str(root), "remote"]).strip(); assert not subprocess.check_output(["git", "-C", str(root), "tag"]).strip(); assert subprocess.check_output(["git", "-C", str(root), "rev-list", "--all", "--count"]).strip() == b"1"; print("candidate_source", src)'; \
    chown -R agent:agent /workspace; \
    rm -f /home/agent/.bash_history

ENV PYTHONPATH=/workspace/repo \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

USER agent
WORKDIR /workspace/repo
ENTRYPOINT []
CMD ["/bin/bash"]
