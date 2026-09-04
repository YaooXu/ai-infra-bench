# syntax=docker/dockerfile:1

ARG BASE_IMAGE=vllm/vllm-openai-cpu@sha256:d19978a2d4bb2289c740a6c89d4cc15fbcf4d20d916f1e268168b8bbad3b776b
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG VLLM_BASE_SHA=d88f28da05b12bc7d63ebe3dcedf445ecb274343
ARG VLLM_SOURCE_SHA256=7808db147610c4413d1d813d559eeb22dcbf37e39285746092c508c680cf3fee
ARG VLLM_SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/d88f28da05b12bc7d63ebe3dcedf445ecb274343

USER 0:0
ENV PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    TOKENIZERS_PARALLELISM=false \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    HOME=/home/agent

RUN case "${BASE_IMAGE}" in \
      *@sha256:d19978a2d4bb2289c740a6c89d4cc15fbcf4d20d916f1e268168b8bbad3b776b) ;; \
      *) echo "BASE_IMAGE must be the frozen v0.17.1 x86_64 CPU manifest" >&2; exit 1 ;; \
    esac

# Keep only compiled extensions and generated version metadata from the
# digest-pinned official release. Candidate Python source is supplied below.
RUN set -eu; \
    installed_vllm="$(python3 -c 'import pathlib,vllm; print(pathlib.Path(vllm.__file__).resolve().parent)')"; \
    mkdir -p /opt/vllm-native-overlay; \
    cd "${installed_vllm}"; \
    find . -type f \( -name '*.so' -o -name '_version.py' \) -print0 \
      | tar --null --files-from=- -cf - \
      | tar -C /opt/vllm-native-overlay -xf -; \
    test -n "$(find /opt/vllm-native-overlay -type f -name '*.so' -print -quit)"

# Archive bytes and the reconstructed canonical Git tree are both locked.
# Force-add recovers upstream-tracked files that also match upstream ignores.
RUN set -eu; \
    curl --fail --location --retry 5 --connect-timeout 30 \
      "${VLLM_SOURCE_URL}" -o /tmp/vllm-source.tar.gz; \
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
    test "$(git rev-parse 'HEAD^{tree}')" = 037cc4e534167733500093aa225de6ab64a1028e; \
    test -z "$(git remote)"; \
    test -z "$(git tag)"

RUN set -eu; \
    cp -a /opt/vllm-native-overlay/. /app/vllm/; \
    cd /app; \
    find /opt/vllm-native-overlay -type f -printf '/vllm/%P\n' \
      | sort >> .git/info/exclude; \
    site_dir="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    printf 'import sys; sys.path.insert(0, "/app")\n' \
      > "${site_dir}/vllm-candidate-source.pth"; \
    cd /tmp; \
    python3 -I -c 'import importlib.util,pathlib,torch,vllm; root=pathlib.Path("/app"); assert pathlib.Path(vllm.__file__).resolve().is_relative_to(root); spec=importlib.util.find_spec("vllm._C"); assert spec and pathlib.Path(spec.origin).resolve().is_relative_to(root); assert torch.__version__ == "2.10.0+cpu"; assert torch.version.cuda is None; assert not torch.cuda.is_available()'; \
    cd /app; \
    test -z "$(git status --porcelain=v1 --untracked-files=all)"

COPY model-fixture/ /opt/bench/model-fixture/

RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/agent agent \
    && chown -R agent:agent /app /home/agent \
    && chmod -R a-w /opt/bench /opt/vllm-native-overlay

WORKDIR /app
USER 10001:10001
ENTRYPOINT []
CMD ["sleep", "infinity"]
