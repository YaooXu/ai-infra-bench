# The scored behavior is pure Python/CPU request lifecycle logic. A CPU donor
# avoids making every Agent run download and initialize an unrelated CUDA
# serving stack while the exact pre-fix Python source is still overlaid below.
ARG BASE_IMAGE=vllm/vllm-openai-cpu:v0.17.1-x86_64@sha256:d19978a2d4bb2289c740a6c89d4cc15fbcf4d20d916f1e268168b8bbad3b776b
FROM ${BASE_IMAGE}

ARG VLLM_BASE_SHA=e94ec597334d9a3e9b0d04bc17152e2747c83d51
ARG VLLM_SOURCE_TREE=9207a921243a2aa82fdf29f5019af966b70d1b50
ARG VLLM_SOURCE_SHA256=9a429cb976d5ca39550795573fa6caf92d995eec611391c70f1c026fd52d2e2e
ARG VLLM_SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/e94ec597334d9a3e9b0d04bc17152e2747c83d51

USER root

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# Build networking is used only to download the immutable source archive. The
# archive has no upstream Git metadata and is replaced by one synthetic commit.
RUN set -eux; \
    command -v curl; \
    command -v git; \
    curl -fL --retry 8 --retry-all-errors --retry-delay 3 \
        --connect-timeout 30 --max-time 600 \
        "${VLLM_SOURCE_URL}" -o /tmp/vllm-source.tar.gz; \
    echo "${VLLM_SOURCE_SHA256}  /tmp/vllm-source.tar.gz" | sha256sum -c -; \
    mkdir -p /workspace/repo; \
    tar -xzf /tmp/vllm-source.tar.gz --strip-components=1 -C /workspace/repo; \
    rm -f /tmp/vllm-source.tar.gz; \
    git -C /workspace/repo init; \
    git -C /workspace/repo config user.name "AI Infra Bench"; \
    git -C /workspace/repo config user.email "noreply@ai-infra-bench.local"; \
    git -C /workspace/repo add -A; \
    test "$(git -C /workspace/repo write-tree)" = "${VLLM_SOURCE_TREE}"; \
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
    if ! command -v python >/dev/null 2>&1; then \
        ln -s "$(command -v python3)" /usr/local/bin/python; \
    fi; \
    chown -R agent:agent /workspace; \
    rm -f /home/agent/.bash_history

COPY --chown=agent:agent public_dev/reproduce_retention.py /opt/ai-infra-bench/dev/reproduce_retention.py
COPY public_dev/run.sh /usr/local/bin/agent-test
RUN chmod 0555 /usr/local/bin/agent-test

ENV PYTHONPATH=/workspace/repo \
    LD_LIBRARY_PATH=/lib/x86_64-linux-gnu

USER agent
WORKDIR /workspace/repo
ENTRYPOINT []
CMD ["sleep", "infinity"]
