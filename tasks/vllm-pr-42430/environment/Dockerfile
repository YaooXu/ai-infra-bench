ARG BASE_IMAGE=vllm/vllm-openai:v0.21.0@sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b
FROM ${BASE_IMAGE}

ARG VLLM_BASE_SHA=737bfa3a43ce386bd1894792f3302d9f3f9d73fa
ARG VLLM_SOURCE_SHA256=4269a850e2ed7fcf4213bde48556eccdcaeca0266e4668e0ec64e3d69d785761
ARG VLLM_SOURCE_TREE=f1ee5dc01c843ebb52d20e1714a93f96ec07cb96
ARG VLLM_SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/737bfa3a43ce386bd1894792f3302d9f3f9d73fa

USER root
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /workspace

# v0.21.0 is the closest official pre-cutoff release and matches the exact
# base's Torch 2.11 / CUDA 13 ABI. Exact base Python and Triton sources replace
# the donor package; copied native artifacts are checked against the manifest.
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
    PYTHONPATH=/workspace/repo python3 -c 'import importlib.util, pathlib, vllm, vllm.vllm_flash_attn; root=pathlib.Path("/workspace/repo").resolve(); src=pathlib.Path(vllm.__file__).resolve(); native=pathlib.Path(importlib.util.find_spec("vllm._C").origin).resolve(); flash=pathlib.Path(vllm.vllm_flash_attn.__file__).resolve(); assert src.is_relative_to(root), src; assert native.is_relative_to(root), native; assert flash.is_relative_to(root), flash; print("candidate_source", src); print("candidate_native", native); print("candidate_flash_wrapper", flash)'; \
    rm -f /root/.bash_history

COPY lock/native-donor.json /workspace/environment-lock/native-donor.json

RUN python3 -c 'import json, pathlib; manifest=json.loads(pathlib.Path("/workspace/environment-lock/native-donor.json").read_text()); root=pathlib.Path("/workspace/repo/vllm"); actual=sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and (p.suffix == ".so" or p.name == "_version.py")); assert actual == manifest["copied_artifacts"], (actual, manifest["copied_artifacts"]); assert manifest["target_native_intersection"] == []; assert manifest["agent_contains_head_files"] is False; print("native_manifest_verified", len(actual))'

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
CMD ["sleep", "infinity"]
