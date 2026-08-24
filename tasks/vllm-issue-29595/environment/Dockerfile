# syntax=docker/dockerfile:1.7

FROM alpine/git:2.49.1@sha256:c0280cf9572316299b08544065d3bf35db65043d5e3963982ec50647d2746e26 AS git_runtime

# Issue #29595 reports vLLM 0.11.1 with torch 2.9.0+cu129.  Keep the
# published image immutable; the registry mirror, if any, is configured on the
# Docker daemon and never embedded here.
FROM vllm/vllm-openai:v0.11.1@sha256:d5b12dfb74d605615f8b29ebafaa52294c118bcac7bc9e941785c4108fdb913a

ARG VLLM_SOURCE_COMMIT=439368496db48d8f992ba8c606a0c0b1eebbfa69
ARG VLLM_SOURCE_SHA256=b36c716c28bfd08359e566db92775d3621bcf31a1fd18505f78cb94128a22972

LABEL ai-infra-bench.vllm-source-commit="${VLLM_SOURCE_COMMIT}"

USER root

# The official runtime omits Git. Copy a digest-pinned, minimal Git runtime so
# the coding agent gets a normal clean-diff workflow without online apt.
COPY --from=git_runtime /usr/bin/git /usr/local/bin/git
COPY --from=git_runtime /usr/libexec/git-core /usr/libexec/git-core
COPY --from=git_runtime /usr/share/git-core /usr/share/git-core
COPY --from=git_runtime /lib/ld-musl-x86_64.so.1 /lib/ld-musl-x86_64.so.1
COPY --from=git_runtime /usr/lib/libpcre2-8.so.0.14.0 /usr/lib/libpcre2-8.so.0.14.0
COPY --from=git_runtime /usr/lib/libz.so.1.3.1 /usr/lib/libz.so.1.3.1
RUN ln -s libpcre2-8.so.0.14.0 /usr/lib/libpcre2-8.so.0 \
    && ln -s libz.so.1.3.1 /usr/lib/libz.so.1 \
    && git --version

# The source archive is fetched once by the environment-preparation step and
# verified here.  Keeping network I/O outside docker build avoids a flaky Git
# fetch through the build network and makes rebuilds offline after base pull.
COPY vllm-source.tar.gz /tmp/vllm-source.tar.gz

# The official wheel supplies the native CUDA extensions.  Use the exact
# release source as the importable Python tree, then fill only files absent
# from the archive (native extensions, generated _version.py, and packaged
# flash-attention wrappers). `--no-clobber` ensures wheel files never replace
# Agent-editable upstream source. This avoids an unnecessary CUDA rebuild while
# ensuring edits under /workspace/repo/vllm are what the runtime executes.
RUN set -eu; \
    printf '%s  %s\n' "${VLLM_SOURCE_SHA256}" /tmp/vllm-source.tar.gz | sha256sum -c -; \
    mkdir -p /workspace/repo; \
    tar -xzf /tmp/vllm-source.tar.gz --strip-components=1 -C /workspace/repo; \
    rm /tmp/vllm-source.tar.gz; \
    installed_pkg="$(python3 -c 'import importlib.metadata; print(importlib.metadata.distribution("vllm").locate_file("vllm"))')"; \
    cd "${installed_pkg}"; \
    find . -type f -exec cp --no-clobber --parents '{}' /workspace/repo/vllm/ \;; \
    site_dir="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    printf "import sys; sys.path.insert(0, '/workspace/repo')\n" > "${site_dir}/000_ai_infra_bench_vllm_source.pth"; \
    test ! -e /workspace/repo/.git; \
    git init /workspace/repo; \
    git -C /workspace/repo config user.name "AI Infra Bench"; \
    git -C /workspace/repo config user.email "noreply@ai-infra-bench.local"; \
    git -C /workspace/repo add -A; \
    git -C /workspace/repo commit -m "Synthetic benchmark base"; \
    git -C /workspace/repo branch -M benchmark-base; \
    test -z "$(git -C /workspace/repo remote)"; \
    test "$(git -C /workspace/repo rev-list --all --count)" = 1; \
    test -z "$(git -C /workspace/repo status --porcelain)"; \
    useradd --create-home --uid 1000 --shell /bin/bash agent; \
    chown -R agent:agent /workspace/repo /home/agent

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PIP_NO_INDEX=1 \
    VLLM_NO_USAGE_STATS=1 \
    MODEL_DIR=/models/Qwen3-VL-4B-Instruct \
    HOME=/home/agent

WORKDIR /workspace/repo
USER agent

# Build-time smoke check: Python must resolve to the editable source tree, and
# the matching CUDA extension must be discoverable there. Loading it is a
# runtime GPU check because Docker builds do not expose libcuda.so.1.
RUN python3 -c "import importlib.util,pathlib,vllm; p=pathlib.Path(vllm.__file__).resolve(); s=importlib.util.find_spec('vllm._C'); print(vllm.__version__, p, s.origin if s else None); assert str(p).startswith('/workspace/repo/vllm/') and s is not None and str(s.origin).startswith('/workspace/repo/vllm/')" \
    && test "$(id -un)" = agent \
    && test -z "$(git remote)" \
    && test "$(git rev-list --all --count)" = 1 \
    && test -z "$(git status --porcelain)"

# Clear the OpenAI serving entrypoint inherited from the official image.  The
# benchmark harness controls service startup and enforces runtime no-network.
ENTRYPOINT []
CMD ["sleep", "infinity"]
