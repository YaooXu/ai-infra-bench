ARG VLLM_RUNTIME_IMAGE=vllm/vllm-openai-cpu:v0.17.1-x86_64@sha256:d19978a2d4bb2289c740a6c89d4cc15fbcf4d20d916f1e268168b8bbad3b776b
FROM ${VLLM_RUNTIME_IMAGE}

ARG VLLM_BASE_SHA=d88f28da05b12bc7d63ebe3dcedf445ecb274343
ARG VLLM_PYTHON_BIN_DIR=/opt/venv/bin

ENV PATH="${VLLM_PYTHON_BIN_DIR}:${PATH}"
ENV TORCH_DEVICE_BACKEND_AUTOLOAD=0

USER root

RUN command -v git >/dev/null \
    && command -v python3 >/dev/null \
    && command -v tar >/dev/null

# Keep the compiled extensions from the nearby v0.17.1 wheel/image. The task
# source snapshot is immediately before that release and uses the same ABI.
RUN python3 -c "import importlib.util, pathlib, shutil; spec = importlib.util.find_spec('vllm'); assert spec and spec.submodule_search_locations; shutil.copytree(pathlib.Path(next(iter(spec.submodule_search_locations))), pathlib.Path('/opt/vllm-wheel'), dirs_exist_ok=True)"

WORKDIR /app

# Bundle the immutable source snapshot in the task so builds do not depend on
# GitHub availability or expose merged history to the agent.
COPY vllm-d88f28da05b12bc7d63ebe3dcedf445ecb274343.tar.gz /tmp/vllm-base.tar.gz

RUN tar -xzf /tmp/vllm-base.tar.gz -C /app --strip-components=1 \
    && rm /tmp/vllm-base.tar.gz \
    && find /opt/vllm-wheel -type f -name '*.so' \
        | while IFS= read -r extension; do \
            relative_path="${extension#/opt/vllm-wheel/}"; \
            mkdir -p "/app/vllm/$(dirname "${relative_path}")"; \
            ln -sf "${extension}" "/app/vllm/${relative_path}"; \
          done

# Scheduler unit tests only need model metadata. Supplying it locally keeps the
# verifier independent of Hugging Face availability while downloading no model
# weights or tokenizer assets.
COPY model-fixture/ /app/facebook/opt-125m/
COPY lock/requirements.txt /tmp/task-requirements.txt

RUN python3 -m pip install --no-cache-dir -r /tmp/task-requirements.txt \
    && rm /tmp/task-requirements.txt \
    && git init \
    && git config user.name "Harbor Task Builder" \
    && git config user.email "harbor@example.invalid" \
    && git add . \
    && git commit -m "Initial task snapshot" \
    && useradd --create-home --shell /bin/bash agent \
    && chown -R agent:agent /app /home/agent

ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false
ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
ENV HF_HUB_OFFLINE=1

WORKDIR /app

# The upstream serving image starts `vllm serve`; Harbor needs a general shell
# environment in which an agent can inspect, edit, and test the repository.
ENTRYPOINT []
CMD ["/bin/bash"]
