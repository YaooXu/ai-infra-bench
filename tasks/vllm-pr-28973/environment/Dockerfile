ARG BASE_IMAGE=vllm/vllm-openai:v0.14.0@sha256:1d6866b87630d94f5e0cdae55ab5abb4ce0b03fcb84d9d10612f9d518d19d4fd
ARG NATIVE_IMAGE=vllm/vllm-openai:v0.15.1@sha256:8c9aaddfa6011b9651d06834d2fb90bdb9ab6ced4b420ec76925024eb12b22d0

FROM ${NATIVE_IMAGE} AS native-artifacts

USER root
RUN set -eux; \
    VLLM_SITE="$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')"; \
    mkdir -p /opt/native-vllm; \
    find "${VLLM_SITE}" -type f -name '*.so' \
        -exec sh -c 'for src do rel="${src#"$0"/}"; dst="/opt/native-vllm/$rel"; mkdir -p "$(dirname "$dst")"; cp -a "$src" "$dst"; done' "${VLLM_SITE}" {} +; \
    test "$(find /opt/native-vllm -type f ! -name '*.so' | wc -l)" = 0

FROM ${BASE_IMAGE}

ARG VLLM_BASE_SHA=0118cdcc02ae16a137645e2289bf41f5e3da9d80
ARG VLLM_SOURCE_SHA256=75b2632ec1ea5f92539b9c5f6a7e3cd3357874f04cfd6953c9ae851f1b992957

USER root

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

COPY --from=native-artifacts /opt/native-vllm /opt/native-vllm

# Build networking is used only to materialize the exact survey base. The
# digest-checked codeload archive contains no Git metadata; create one synthetic
# benchmark commit after extraction. Predefined Docker proxy args are not
# redeclared so their values are excluded from image history and config.
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
    git -C /workspace/repo add -A; \
    git -C /workspace/repo commit --quiet -m "Synthetic benchmark base"; \
    git -C /workspace/repo branch -M benchmark-base; \
    test -z "$(git -C /workspace/repo remote)"; \
    test "$(git -C /workspace/repo rev-list --all --count)" = 1; \
    VLLM_SITE="$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')"; \
    cp -a "${VLLM_SITE}/_version.py" /workspace/repo/vllm/_version.py; \
    find /opt/native-vllm -type f -name '*.so' \
        -exec sh -c 'for src do rel="${src#"$0"/}"; dst="/workspace/repo/vllm/$rel"; mkdir -p "$(dirname "$dst")"; cp -a "$src" "$dst"; done' /opt/native-vllm {} +; \
    rm -rf /opt/native-vllm; \
    test -z "$(git -C /workspace/repo status --porcelain)"; \
    rm -f /root/.bash_history

COPY environment/public_dev /workspace/public_dev

RUN set -eux; \
    python3 /workspace/public_dev/make_tiny_model.py; \
    if ! id -u agent >/dev/null 2>&1; then \
        useradd --create-home --shell /bin/bash agent; \
    fi; \
    chown -R agent:agent /workspace /opt/models/tiny-streaming; \
    chmod 755 /workspace/public_dev/run.sh; \
    rm -f /home/agent/.bash_history

ENV PYTHONPATH=/workspace/repo \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

USER agent
WORKDIR /workspace/repo
ENTRYPOINT []
CMD ["/bin/bash"]
