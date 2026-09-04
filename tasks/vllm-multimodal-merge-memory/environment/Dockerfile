# syntax=docker/dockerfile:1.7

FROM alpine/git:2.49.1@sha256:c0280cf9572316299b08544065d3bf35db65043d5e3963982ec50647d2746e26 AS git_runtime

# The accepted squash parent predates v0.19.0 by roughly two days. v0.19.0 is
# the first official same-Torch image that contains the complete native set
# imported by this source snapshot. All native objects come from this one
# internally consistent image; no mixed-release object is used.
FROM vllm/vllm-openai:v0.19.0@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504

ARG VLLM_BASE_SHA=36d7f19897843c9cbdb701ba88d0f2c29954fe44
ARG VLLM_BASE_TREE=647523aac911ae44a1dcaab79ef66d58290b478e
ARG VLLM_SOURCE_SHA256=8e71d0480773b8532f7397e52257e5600b0ef0ff56654eba5e78afc48be0cbab
ARG VLLM_SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/36d7f19897843c9cbdb701ba88d0f2c29954fe44

LABEL ai-infra-bench.source-commit="${VLLM_BASE_SHA}" \
      ai-infra-bench.native-scope="v0.19.0-donor; verifier exercises Python/PyTorch merge primitive only"

USER root

COPY --from=git_runtime /usr/bin/git /usr/local/bin/git
COPY --from=git_runtime /usr/libexec/git-core /usr/libexec/git-core
COPY --from=git_runtime /usr/share/git-core /usr/share/git-core
COPY --from=git_runtime /lib/ld-musl-x86_64.so.1 /lib/ld-musl-x86_64.so.1
COPY --from=git_runtime /usr/lib/libpcre2-8.so.0.14.0 /usr/lib/libpcre2-8.so.0.14.0
COPY --from=git_runtime /usr/lib/libz.so.1.3.1 /usr/lib/libz.so.1.3.1

COPY lock/native.sha256 /tmp/native.sha256

# Network is needed only to acquire the immutable source archive. Runtime is
# offline. Docker proxy values are predefined build arguments and are neither
# redeclared nor persisted in image configuration.
RUN set -eux; \
    ln -s libpcre2-8.so.0.14.0 /usr/lib/libpcre2-8.so.0; \
    ln -s libz.so.1.3.1 /usr/lib/libz.so.1; \
    git --version; \
    curl -fL --retry 8 --retry-all-errors --retry-delay 3 \
        --connect-timeout 30 --max-time 600 \
        "${VLLM_SOURCE_URL}" \
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
    VLLM_SITE=/usr/local/lib/python3.12/dist-packages/vllm; \
    cd "${VLLM_SITE}"; \
    sha256sum -c /tmp/native.sha256; \
    for rel in \
        _C.abi3.so \
        _C_stable_libtorch.abi3.so \
        _flashmla_C.abi3.so \
        _flashmla_extension_C.abi3.so \
        _moe_C.abi3.so \
        cumem_allocator.abi3.so \
        vllm_flash_attn/_vllm_fa2_C.abi3.so \
        vllm_flash_attn/_vllm_fa3_C.abi3.so; do \
        test -f "${rel}"; \
        test ! -L "${rel}"; \
        readelf -h "${rel}" >/dev/null; \
        install -D -m 0755 "${rel}" "/workspace/repo/vllm/${rel}"; \
    done; \
    install -D -m 0644 _version.py /workspace/repo/vllm/_version.py; \
    test "$(find /workspace/repo/vllm -type f -name '*.so' | wc -l)" = 8; \
    test -z "$(git -C /workspace/repo status --porcelain)"; \
    rm -f /tmp/native.sha256; \
    if ! id -u agent >/dev/null 2>&1; then \
        useradd --create-home --uid 1000 --shell /bin/bash agent; \
    fi; \
    if ! command -v python >/dev/null 2>&1; then \
        ln -s "$(command -v python3)" /usr/local/bin/python; \
    fi; \
    chown -R agent:agent /workspace/repo /home/agent

ENV PYTHONPATH=/workspace/repo \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PIP_NO_INDEX=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/home/agent

USER agent
WORKDIR /workspace/repo

RUN python3 -c "import pathlib,vllm; from vllm.model_executor.models.utils import _merge_multimodal_embeddings; p=pathlib.Path(vllm.__file__).resolve(); print(p); assert str(p).startswith('/workspace/repo/vllm/')" \
    && test "$(id -un)" = agent \
    && test -z "$(git remote)" \
    && test "$(git rev-list --all --count)" = 1 \
    && test -z "$(git status --porcelain)"

ENTRYPOINT []
CMD ["/bin/bash"]
