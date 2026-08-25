ARG BASE_IMAGE=vllm/vllm-openai:v0.15.1@sha256:8c9aaddfa6011b9651d06834d2fb90bdb9ab6ced4b420ec76925024eb12b22d0
FROM ${BASE_IMAGE}

ARG VLLM_BASE_SHA=e94ec597334d9a3e9b0d04bc17152e2747c83d51
ARG VLLM_REPOSITORY=https://github.com/vllm-project/vllm.git

USER root

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# Build networking is used only to materialize the exact survey base. The
# original repository is exported without its Git metadata, then replaced by
# one synthetic benchmark commit. Keep all fetch/removal steps in one layer so
# upstream history is not retained in an earlier image layer.
RUN set -eux; \
    apt-get \
        -o Acquire::http::Proxy=false \
        -o Acquire::https::Proxy=false \
        update; \
    DEBIAN_FRONTEND=noninteractive apt-get \
        -o Acquire::http::Proxy=false \
        -o Acquire::https::Proxy=false \
        install -y --no-install-recommends \
        ca-certificates \
        git; \
    rm -rf /var/lib/apt/lists/*; \
    git init /tmp/vllm-upstream; \
    git -C /tmp/vllm-upstream fetch --depth=1 "${VLLM_REPOSITORY}" "${VLLM_BASE_SHA}"; \
    git -C /tmp/vllm-upstream checkout --detach FETCH_HEAD; \
    test "$(git -C /tmp/vllm-upstream rev-parse HEAD)" = "${VLLM_BASE_SHA}"; \
    mkdir -p /workspace/repo; \
    git -C /tmp/vllm-upstream archive HEAD | tar -x -C /workspace/repo; \
    rm -rf /tmp/vllm-upstream; \
    git -C /workspace/repo init; \
    git -C /workspace/repo config user.name "AI Infra Bench"; \
    git -C /workspace/repo config user.email "noreply@ai-infra-bench.local"; \
    git -C /workspace/repo add -A; \
    git -C /workspace/repo commit --quiet -m "Synthetic benchmark base"; \
    git -C /workspace/repo branch -M benchmark-base; \
    test -z "$(git -C /workspace/repo remote)"; \
    test "$(git -C /workspace/repo rev-list --all --count)" = 1; \
    VLLM_SITE="$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')"; \
    find "${VLLM_SITE}" -type f \( -name '*.so' -o -name '_version.py' \) \
        -exec sh -c 'for src do rel="${src#"$0"/}"; dst="/workspace/repo/vllm/$rel"; mkdir -p "$(dirname "$dst")"; cp -a "$src" "$dst"; done' "${VLLM_SITE}" {} +; \
    test -z "$(git -C /workspace/repo status --porcelain)"; \
    rm -f /root/.bash_history

RUN set -eux; \
    if ! id -u agent >/dev/null 2>&1; then \
        useradd --create-home --shell /bin/bash agent; \
    fi; \
    chown -R agent:agent /workspace; \
    rm -f /home/agent/.bash_history

ENV PYTHONPATH=/workspace/repo \
    LD_LIBRARY_PATH=/lib/x86_64-linux-gnu

USER agent
WORKDIR /workspace/repo
ENTRYPOINT []
CMD ["sleep", "infinity"]
