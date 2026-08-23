# vLLM multimodal encoder-cache compaction task

This directory is a Harbor schema 1.3 task for the vLLM multimodal encoder
cache placeholder/embedding-count contract. `instruction.md` is the only task
description intended for solving agents; this README documents packaging and
curator validation.

## Packaging protocol

- The candidate source is fetched at image-build time from the exact public
  base commit archive in `environment/lock/sources.lock.json`.
- The outer archive SHA-256, upstream Git tree, every tracked file, executable
  bit and aggregate semantic tree are verified before compilation.
- Python dependencies are installed from public PyPI using a complete
  `--require-hashes` lock.
- vLLM native extensions are rebuilt for SM80 from the verified base source and
  seven hash-pinned public source archives. No curator wheel/native bundle is
  required.
- Agent and separate-verifier runtime networking remains disabled.
- The agent image contains one editable synthetic Git commit under `/app` and
  never contains tests or the solution.
- Harbor restores the submitted artifact at its original source path `/app` in
  the independent verifier environment.

See `environment/lock/ARTIFACTS.md` for the full supply-chain contract.

## Local builds

```bash
docker build --network default --no-cache \
  -f environment/Dockerfile \
  -t vllm-mm-cache-agent:local \
  environment

docker build --network none --no-cache \
  --build-arg AGENT_IMAGE=vllm-mm-cache-agent:local \
  -f tests/Dockerfile \
  -t vllm-mm-cache-verifier:local \
  tests
```

The first build may access only the public sources declared by the manifests.
Runtime containers must use `--network none`; the canonical validation hardware
is one NVIDIA A100-SXM4-40GB.

## Verifier outputs

`/logs/verifier/reward.json` is deliberately binary: it contains only
`{"reward": 1.0}` when all declared correctness requirements pass, otherwise
`{"reward": 0.0}`. Continuous completion, group scores, validity and
infrastructure diagnostics are retained separately in `scoring.json`.

The release image fields remain empty until maintainers publish the exact
validated images and obtain registry manifest RepoDigests. Local image IDs and
archive-file hashes are never substituted for those identities.
