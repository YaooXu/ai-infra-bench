ARG BASE_IMAGE=vllm/vllm-openai:v0.9.2@sha256:37cd5bd18d220a0f4c70401ce1d4a0cc588fbfe03cc210579428f2c47e6eac33
FROM ${BASE_IMAGE}

ARG VLLM_BASE_SHA=b854321ffe50fd04c6b1ac58eecdab4caf5b4295
ARG VLLM_SOURCE_SHA256=fd58dd99ebcef8c7470fe87c714dd29be68b20cb78f0c1e7be2dea53d43e908d
ARG VLLM_SOURCE_TREE=d2526d84114539875b099b79ca4002f1d39c977e
ARG VLLM_SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/b854321ffe50fd04c6b1ac58eecdab4caf5b4295

USER root
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /workspace

# Package the exact survey base. This eligibility image deliberately does not
# install a future DeepGEMM checkout: PR 20087's target v2 SM100 kernel cannot
# be built or executed faithfully on the available SM80 A100.
RUN set -eux; \
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

COPY --chown=agent:agent environment/public_dev /workspace/public_dev
RUN chmod 755 /workspace/public_dev/run.sh

ENV PYTHONPATH=/workspace/repo \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

USER agent
WORKDIR /workspace/repo
ENTRYPOINT []
CMD ["/bin/bash"]
