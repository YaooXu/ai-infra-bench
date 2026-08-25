#!/usr/bin/env bash
set -Eeuo pipefail

on_error() {
  local status=$?
  printf 'ERROR: recipe-only image build failed at line %s (exit=%s)\n' \
    "${BASH_LINENO[0]}" "$status" >&2
  exit "$status"
}
trap on_error ERR

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

VLLM_MM_CACHE_AGENT_TAG=${VLLM_MM_CACHE_AGENT_TAG:-ai-infra-bench/vllm-mm-encoder-cache-compaction-agent:oss}
VLLM_MM_CACHE_VERIFIER_TAG=${VLLM_MM_CACHE_VERIFIER_TAG:-ai-infra-bench/vllm-mm-encoder-cache-compaction-verifier:oss}
VLLM_MM_CACHE_BASE_IMAGE=${VLLM_MM_CACHE_BASE_IMAGE:-pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel@sha256:97ec2a667dd7560c615bf50a95b2fb85a673ae233a55da1706e8e04e6d6d518e}
VLLM_MM_CACHE_RUNTIME_BASE_IMAGE=${VLLM_MM_CACHE_RUNTIME_BASE_IMAGE:-pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime@sha256:f0ca81b440e252399d9954a45b616ee2540959466aacf3dfc3f856691eee66e8}

BUILD_PROXY_ARGS=()
for proxy_name in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if test -n "${!proxy_name:-}"; then
    BUILD_PROXY_ARGS+=(--build-arg "$proxy_name=${!proxy_name}")
  fi
done

command -v docker >/dev/null
test -f "$SCRIPT_DIR/Dockerfile"
test -f "$TASK_ROOT/tests/Dockerfile"
test -f "$SCRIPT_DIR/lock/public-build-manifest.json"
test ! -e "$SCRIPT_DIR/lock/starter"

printf '[1/2] Building public-source agent image: %s\n' \
  "$VLLM_MM_CACHE_AGENT_TAG"
docker build \
  --network default \
  --no-cache \
  --pull=false \
  --build-arg "BASE_IMAGE=$VLLM_MM_CACHE_BASE_IMAGE" \
  --build-arg "RUNTIME_BASE_IMAGE=$VLLM_MM_CACHE_RUNTIME_BASE_IMAGE" \
  "${BUILD_PROXY_ARGS[@]}" \
  --file "$SCRIPT_DIR/Dockerfile" \
  --tag "$VLLM_MM_CACHE_AGENT_TAG" \
  "$SCRIPT_DIR"

printf '[2/2] Building no-network separate verifier image: %s\n' \
  "$VLLM_MM_CACHE_VERIFIER_TAG"
docker build \
  --network none \
  --no-cache \
  --pull=false \
  --build-arg "AGENT_IMAGE=$VLLM_MM_CACHE_AGENT_TAG" \
  --file "$TASK_ROOT/tests/Dockerfile" \
  --tag "$VLLM_MM_CACHE_VERIFIER_TAG" \
  "$TASK_ROOT/tests"

agent_id=$(docker image inspect --format '{{.Id}}' "$VLLM_MM_CACHE_AGENT_TAG")
verifier_id=$(docker image inspect --format '{{.Id}}' "$VLLM_MM_CACHE_VERIFIER_TAG")

printf '\nRECIPE_ONLY_IMAGES_READY\n'
printf 'agent_tag=%s\nagent_image_id=%s\n' \
  "$VLLM_MM_CACHE_AGENT_TAG" "$agent_id"
printf 'verifier_tag=%s\nverifier_image_id=%s\n' \
  "$VLLM_MM_CACHE_VERIFIER_TAG" "$verifier_id"
printf '%s\n' 'NOTE: local image IDs are build evidence, not OCI RepoDigests.'
