ARG BASE_IMAGE=vllm/vllm-openai:v0.21.0@sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b
FROM ${BASE_IMAGE}

ARG VLLM_BASE_SHA=9b9d5dbaab852a1c615fe83a7f92881d353503db
ARG VLLM_SOURCE_SHA256=cc81853110ee854c6c64cbfc9bca8d75887d48ea6bc27c38f63131ce0a71f1f7
ARG VLLM_SOURCE_TREE=94c86336cf2ea962766d00bb389d43a4d6aaf697
ARG VLLM_SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/9b9d5dbaab852a1c615fe83a7f92881d353503db

USER root

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# The archive is the exact survey base and contains no upstream Git metadata.
# Native artifacts come from the closest official pre-cutoff release image;
# this PR changes only Python frontend orchestration code.
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git=1:2.34.1-1ubuntu1.17 \
        git-man=1:2.34.1-1ubuntu1.17 \
        liberror-perl=0.17029-1; \
    rm -rf /var/lib/apt/lists/*; \
    command -v curl; \
    command -v git; \
    curl -fL \
        --retry 8 \
        --retry-all-errors \
        --retry-delay 3 \
        --connect-timeout 30 \
        --max-time 600 \
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
    test "$(git -C /workspace/repo write-tree)" = "${VLLM_SOURCE_TREE}"; \
    git -C /workspace/repo commit --quiet -m "Synthetic benchmark base"; \
    git -C /workspace/repo branch -M benchmark-base; \
    test -z "$(git -C /workspace/repo remote)"; \
    test "$(git -C /workspace/repo rev-list --all --count)" = 1; \
    VLLM_SITE="$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')"; \
    find "${VLLM_SITE}" -type f \( -name '*.so' -o -name '_version.py' \) \
        -exec sh -c 'for src do rel="${src#"$0"/}"; dst="/workspace/repo/vllm/$rel"; mkdir -p "$(dirname "$dst")"; cp -a "$src" "$dst"; done' "${VLLM_SITE}" {} +; \
    test -z "$(git -C /workspace/repo status --porcelain)"; \
    PYTHONPATH=/workspace/repo python3 -c 'import importlib.util, pathlib, vllm; root=pathlib.Path("/workspace/repo").resolve(); src=pathlib.Path(vllm.__file__).resolve(); native=pathlib.Path(importlib.util.find_spec("vllm._C").origin).resolve(); assert src.is_relative_to(root), src; assert native.is_relative_to(root), native; print("candidate_source", src); print("candidate_native", native)'; \
    rm -f /root/.bash_history

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
