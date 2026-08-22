# A100-metal currently has this image locally. This remains a host-specific
# development base until a portable image is published.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ARG VLLM_BASE_SHA=ac1886588fd8799ff874b860b6c266a84d5a2b2b

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    VLLM_TARGET_DEVICE=empty

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libnuma1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
        "setuptools>=77.0.3,<81.0.0" \
        "setuptools-scm>=8.0" \
        "packaging>=24.2" \
        "cmake>=3.26.1" \
        ninja \
        wheel \
        jinja2 \
        pytest

RUN python -m pip uninstall -y torch torchvision torchaudio \
    && python -m pip install --no-cache-dir \
        torch==2.9.0 \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace

# Fetch the exact upstream base, export only its worktree, and create one new
# synthetic commit in the same build layer. The final image therefore contains
# no upstream objects, refs, remotes, reflogs, or fetch metadata.
RUN git init repo \
    && git -C repo remote add origin https://github.com/vllm-project/vllm.git \
    && git -C repo fetch --depth=1 origin "${VLLM_BASE_SHA}" \
    && git -C repo checkout --detach FETCH_HEAD \
    && rm -rf repo/.git \
    && git -C repo init \
    && git -C repo config user.name "AI Infra Bench" \
    && git -C repo config user.email "noreply@ai-infra-bench.local" \
    && git -C repo add -A \
    && git -C repo commit -m "Synthetic benchmark base" \
    && git -C repo branch -M benchmark-base \
    && rm -rf \
        repo/.git/logs \
        repo/.git/refs/remotes \
        repo/.git/FETCH_HEAD \
        repo/.git/ORIG_HEAD \
        repo/.git/COMMIT_EDITMSG \
    && test -z "$(git -C repo remote)" \
    && test "$(git -C repo rev-list --all --count)" = "1"

WORKDIR /workspace/repo
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+infrabench \
    python -m pip install --no-cache-dir --no-build-isolation -e .

RUN python -c "import vllm; from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEModularKernel; print('vllm import smoke passed')"

WORKDIR /workspace/repo
