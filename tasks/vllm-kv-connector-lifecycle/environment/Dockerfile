ARG BASE_IMAGE=vllm/vllm-openai:v0.20.1@sha256:9eff9734a30b6713a8566217d36f8277630fd2d31cec7f0a0292835901a23aa4
FROM ${BASE_IMAGE}

ARG VLLM_BASE_SHA=f80aa53c9dc2273a19a6855092069db7e1306fff
ARG VLLM_BASE_TREE=2af517bd7880077a9fed9a39dc0e8b1e244a48b1
ARG VLLM_SOURCE_SHA256=a2923ff0ff39b1c32b18ba6eb6255c646e2fc49280552d0637d522e9434baebe

USER root

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

COPY lock/native-paths.txt /tmp/native-paths.txt

# Acquire a digest-checked archive of the exact survey base. It contains no
# upstream Git metadata; create one synthetic benchmark commit after extraction.
# Docker's predefined proxy args are not redeclared, keeping values out of image
# configuration and history.
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
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
    git -C /workspace/repo add -f -A; \
    test "$(git -C /workspace/repo write-tree)" = "${VLLM_BASE_TREE}"; \
    git -C /workspace/repo commit --quiet -m "Synthetic benchmark base"; \
    git -C /workspace/repo branch -M benchmark-base; \
    test "$(git -C /workspace/repo rev-parse 'HEAD^{tree}')" = "${VLLM_BASE_TREE}"; \
    test -z "$(git -C /workspace/repo remote)"; \
    test "$(git -C /workspace/repo rev-list --all --count)" = 1; \
    VLLM_SITE="$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')"; \
    while IFS= read -r rel; do \
        test -n "${rel}"; \
        test -f "${VLLM_SITE}/${rel}"; \
        test ! -L "${VLLM_SITE}/${rel}"; \
        mode=0644; \
        case "${rel}" in \
            *.so) readelf -h "${VLLM_SITE}/${rel}" >/dev/null; mode=0755 ;; \
        esac; \
        install -D -m "${mode}" "${VLLM_SITE}/${rel}" "/workspace/repo/vllm/${rel}"; \
    done < /tmp/native-paths.txt; \
    test "$(wc -l < /tmp/native-paths.txt)" = 10; \
    test "$(find /workspace/repo/vllm -type f -name '*.so' | wc -l)" = 9; \
    test -z "$(git -C /workspace/repo status --porcelain)"; \
    rm -f /tmp/native-paths.txt /root/.bash_history

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
