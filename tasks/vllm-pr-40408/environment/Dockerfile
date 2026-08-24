# syntax=docker/dockerfile:1

ARG BASE_IMAGE=vllm/vllm-openai@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG VLLM_BASE_SHA=ea0e501bb18c12b80acc05ff8c7f013db515ba80
ARG VLLM_SOURCE_SHA256=28f46bea58c2a2ac64ea5beaddc58c26a9f3fe05121e65afdbd125a4e65ce665
ARG VLLM_SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/ea0e501bb18c12b80acc05ff8c7f013db515ba80

USER 0:0
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_TARGET_DEVICE=cuda \
    HOME=/home/agent

RUN case "${BASE_IMAGE}" in \
      *@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc) ;; \
      *) echo "BASE_IMAGE must be the frozen v0.20.0 linux/amd64 manifest" >&2; exit 1 ;; \
    esac

# The codeload archive contains no Git history or remotes. It is retained as an
# exact, editable source tree for structural inspection. This A100 probe image
# deliberately does not overlay the v0.20.0 release extension into /app: the
# release extension is not an exact native build of VLLM_BASE_SHA.
RUN set -eu; \
    curl --fail --location --retry 5 --connect-timeout 30 \
      "${VLLM_SOURCE_URL}" -o /tmp/vllm-source.tar.gz; \
    printf '%s  %s\n' "${VLLM_SOURCE_SHA256}" /tmp/vllm-source.tar.gz \
      | sha256sum --check --strict; \
    mkdir -p /app; \
    tar -xzf /tmp/vllm-source.tar.gz -C /app --strip-components=1; \
    rm /tmp/vllm-source.tar.gz; \
    test ! -e /app/.git; \
    test -f /app/csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x.cu

COPY source_gate.py hardware_probe.py /opt/bench/
COPY lock/ /opt/bench/lock/

RUN set -eu; \
    python3 -I /opt/bench/source_gate.py /app; \
    python3 -I -c 'import importlib.util,pathlib,torch,vllm; root=pathlib.Path("/app"); assert not pathlib.Path(vllm.__file__).resolve().is_relative_to(root); spec=importlib.util.find_spec("vllm._C"); assert spec and not pathlib.Path(spec.origin).resolve().is_relative_to(root); assert vllm.__version__ == "0.20.0"; assert torch.__version__ == "2.11.0+cu130"; assert torch.version.cuda == "13.0"'

RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/agent agent \
    && chown -R agent:agent /app /home/agent \
    && chmod -R a-w /opt/bench

LABEL org.ai-infra-bench.task="vllm-pr-40408" \
      org.ai-infra-bench.base-sha="ea0e501bb18c12b80acc05ff8c7f013db515ba80" \
      org.ai-infra-bench.status="hardware-blocked-a100-sm80" \
      org.ai-infra-bench.image-role="eligibility-and-structural-probe-only"

WORKDIR /app
USER 10001:10001
ENTRYPOINT []
CMD ["python3", "-I", "/opt/bench/hardware_probe.py"]
