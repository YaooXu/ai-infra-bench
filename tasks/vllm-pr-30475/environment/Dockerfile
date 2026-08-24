# syntax=docker/dockerfile:1

# vLLM 0.11.2 is the nearest released official image to base commit
# 676db55eecf8b6d9ec38ea243cf6f35ea8378ec6 (0.11.2.dev278). The amd64
# platform manifest is pinned, rather than the mutable release tag.
ARG BASE_IMAGE=vllm/vllm-openai@sha256:47a9896f86818fea323b2d38082758c62d9a0155d6fe6c4dbd7d735c556f680a
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG VLLM_BASE_SHA=676db55eecf8b6d9ec38ea243cf6f35ea8378ec6
ARG VLLM_SOURCE_SHA256=ac774fb6bbf75b083d997dcb70b5cdf8746c6e6bfa8d95b3c2415a6c30240624
ARG VLLM_SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/676db55eecf8b6d9ec38ea243cf6f35ea8378ec6

USER 0:0
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VLLM_VERSION_OVERRIDE=0.11.2.dev278 \
    HOME=/home/agent

# Reject accidental tag-only or cross-platform overrides. Runtime dependencies
# and CUDA extensions come from the official v0.11.2 amd64 image.
RUN case "${BASE_IMAGE}" in \
      *@sha256:47a9896f86818fea323b2d38082758c62d9a0155d6fe6c4dbd7d735c556f680a) ;; \
      *) echo "BASE_IMAGE must be the frozen v0.11.2 amd64 manifest" >&2; exit 1 ;; \
    esac

# Preserve the official image's compiled extensions and generated version
# module separately. Candidate Python source is then materialized from the
# exact, hash-verified base commit.
# Force-add is required because the upstream archive contains a few tracked
# files that also match the repository's own ignore rules.
RUN set -eu; \
    installed_vllm="$(python3 -c 'import pathlib,vllm; print(pathlib.Path(vllm.__file__).resolve().parent)')"; \
    mkdir -p /opt/vllm-native-overlay; \
    cd "${installed_vllm}"; \
    find . -type f \( -name '*.so' -o -name '_version.py' \) -print0 \
      | tar --null --files-from=- -cf - \
      | tar -C /opt/vllm-native-overlay -xf -; \
    test -n "$(find /opt/vllm-native-overlay -type f -name '*.so' -print -quit)"

RUN set -eu; \
    curl --fail --location --retry 5 --connect-timeout 30 \
      "${VLLM_SOURCE_URL}" \
      -o /tmp/vllm-source.tar.gz; \
    printf '%s  %s\n' "${VLLM_SOURCE_SHA256}" /tmp/vllm-source.tar.gz \
      | sha256sum --check --strict; \
    mkdir -p /app; \
    tar -xzf /tmp/vllm-source.tar.gz -C /app --strip-components=1; \
    rm /tmp/vllm-source.tar.gz; \
    cd /app; \
    git init -q; \
    git config user.name 'AI Infra Benchmark'; \
    git config user.email 'benchmark@invalid.example'; \
    git add -f -A; \
    GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' \
    GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' \
      git commit -q -m 'sanitized benchmark base'; \
    git reflog expire --expire=now --all; \
    test "$(git rev-list --count HEAD)" = 1; \
    test "$(git rev-parse 'HEAD^{tree}')" = aafa39e6544cbeaf83b72985f83aaefd4e9e3456; \
    test -z "$(git remote)"; \
    test -z "$(git tag)"

# Apply only the compiled shared objects from the same official release line.
# Their exact runtime locations are ignored without hiding any candidate Python
# source. An executable .pth places /app before the installed wheel package.
RUN set -eu; \
    cp -a /opt/vllm-native-overlay/. /app/vllm/; \
    cd /app; \
    find /opt/vllm-native-overlay -type f -printf '/vllm/%P\n' \
      | sort >> .git/info/exclude; \
    site_dir="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    printf 'import sys; sys.path.insert(0, "/app")\n' \
      > "${site_dir}/vllm-candidate-source.pth"; \
    cd /tmp; \
    python3 -I -c 'import importlib.util,pathlib,torch,vllm; root=pathlib.Path("/app"); assert pathlib.Path(vllm.__file__).resolve().is_relative_to(root); spec=importlib.util.find_spec("vllm._C"); assert spec and pathlib.Path(spec.origin).resolve().is_relative_to(root); assert torch.__version__.startswith("2.9."); assert torch.version.cuda == "12.9"'; \
    cd /app; \
    test -z "$(git status --porcelain=v1 --untracked-files=all)"

COPY runtime_smoke.py reproduce_baseline.py /opt/bench/
COPY lock/ /opt/bench/lock/

RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/agent agent \
    && chown -R agent:agent /app /home/agent \
    && chmod -R a-w /opt/bench /opt/vllm-native-overlay

WORKDIR /app
USER 10001:10001
ENTRYPOINT []
CMD ["python3", "-I", "/opt/bench/runtime_smoke.py"]
