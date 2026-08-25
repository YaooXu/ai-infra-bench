# Build after the local public-source agent image. The default is the stable
# recipe-only tag produced by environment/build_images.sh; callers may still
# override it with another local tag for isolated validation.
ARG AGENT_IMAGE=ai-infra-bench/vllm-mm-encoder-cache-compaction-agent:oss
FROM ${AGENT_IMAGE}
ARG AGENT_IMAGE
RUN test -n "$AGENT_IMAGE"

USER 0:0
WORKDIR /tmp
# The agent image keeps /app on sys.path through this editable-path file so
# candidates can import the source they are modifying.  A separate verifier
# must not inherit it: Python processes site .pth files before ``-I`` user code
# starts, which would let a candidate /app/sitecustomize.py execute before the
# trusted verifier entrypoint.  The verifier adds /app explicitly only after
# importing its trusted pytest framework (see pytest_entry.py).
RUN test -f /opt/bench-venv/lib/python3.11/site-packages/vllm-editable.pth && \
    rm /opt/bench-venv/lib/python3.11/site-packages/vllm-editable.pth && \
    cd /tmp && \
    /opt/bench-venv/bin/python -I -c \
      "import sys; assert '/app' not in sys.path, sys.path"
# The agent-derived source must not survive as a pristine verifier fallback.
# Harbor rematerializes the submitted artifact at its original source /app.
RUN rm -rf /app && \
    mkdir -p /app /logs/verifier && \
    chown 10001:10001 /app /logs/verifier
COPY --chown=10001:10001 . /tests/
# integrity.json cannot hash itself. This root-owned image-layer anchor closes
# that loop and is required by the containerized verifier.
RUN sha256sum /tests/integrity.json > /opt/bench/verifier-integrity.sha256 && \
    chmod -R a-w /tests /opt/bench/verifier-integrity.sha256
ENV BENCH_INTEGRITY_ANCHOR=/opt/bench/verifier-integrity.sha256 \
    BENCH_REQUIRE_INTEGRITY_ANCHOR=1
USER 10001:10001
WORKDIR /app
