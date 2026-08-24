# syntax=docker/dockerfile:1.7

FROM alpine/git:2.49.1@sha256:c0280cf9572316299b08544065d3bf35db65043d5e3963982ec50647d2746e26 AS git_runtime
RUN tar -czf /git-runtime.tar \
    /usr/bin/git \
    /usr/libexec/git-core \
    /usr/share/git-core \
    /lib/ld-musl-x86_64.so.1 \
    /usr/lib/libpcre2-8.so.0.14.0 \
    /usr/lib/libz.so.1.3.1

# The PR base is 283 commits after v0.11.1 and includes native-source changes.
# An exact offline rebuild was attempted but CMake requires five uncached
# external source projects. The constrained fallback overlays the *entire*
# exact-base source tree and reuses only wheel-only/native files from the
# closest official runtime. The native-origin label makes this ABI risk
# machine-visible; this is not the rejected single-Python-file construction.
FROM vllm/vllm-openai:v0.11.1@sha256:d5b12dfb74d605615f8b29ebafaa52294c118bcac7bc9e941785c4108fdb913a

ARG VLLM_SOURCE_COMMIT=b07555d26f4c7ad9a2d1ec45428a9d4287db612c
ARG VLLM_SOURCE_SHA256=5f5d48bb58d898d89d65bb4310b916f3472f84db0cb17f994203084e3aafce1f

LABEL ai-infra-bench.vllm-source-commit="${VLLM_SOURCE_COMMIT}" \
      ai-infra-bench.vllm-native-origin="v0.11.1:439368496db48d8f992ba8c606a0c0b1eebbfa69"

USER root

COPY --from=git_runtime /git-runtime.tar /tmp/git-runtime.tar
RUN tar -xzf /tmp/git-runtime.tar -C / \
    && rm /tmp/git-runtime.tar \
    && ln -s libpcre2-8.so.0.14.0 /usr/lib/libpcre2-8.so.0 \
    && ln -s libz.so.1.3.1 /usr/lib/libz.so.1 \
    && git --version

COPY vllm-source.tar.gz /tmp/vllm-source.tar.gz
RUN set -eu; \
    printf '%s  %s\n' "${VLLM_SOURCE_SHA256}" /tmp/vllm-source.tar.gz | sha256sum -c -; \
    mkdir -p /workspace/repo; \
    tar -xzf /tmp/vllm-source.tar.gz --strip-components=1 -C /workspace/repo; \
    rm /tmp/vllm-source.tar.gz; \
    test ! -e /workspace/repo/.git; \
    installed_pkg="$(python3 -c 'import importlib.metadata; print(importlib.metadata.distribution("vllm").locate_file("vllm"))')"; \
    cd "${installed_pkg}"; \
    find . -type f -exec cp --no-clobber --parents '{}' /workspace/repo/vllm/ \;; \
    site_dir="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    printf "import sys; sys.path.insert(0, '/workspace/repo')\n" > "${site_dir}/000_ai_infra_bench_vllm_source.pth"

ENV PIP_NO_INDEX=1

# Create the coding baseline after the exact source and wheel-only runtime
# surface are assembled, so native files are tracked rather than untracked.
RUN set -eu; \
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
    VLLM_NO_USAGE_STATS=1 \
    HOME=/home/agent

WORKDIR /workspace/repo
USER agent

RUN python3 -c "import importlib.util,pathlib,vllm; p=pathlib.Path(vllm.__file__).resolve(); s=importlib.util.find_spec('vllm._C'); print(vllm.__version__,p,s.origin if s else None); assert str(p).startswith('/workspace/repo/vllm/') and s is not None and str(s.origin).startswith('/workspace/repo/vllm/')" \
    && test "$(id -un)" = agent \
    && test -z "$(git remote)" \
    && test "$(git rev-list --all --count)" = 1 \
    && test -z "$(git status --porcelain)"

ENTRYPOINT []
CMD ["sleep", "infinity"]
