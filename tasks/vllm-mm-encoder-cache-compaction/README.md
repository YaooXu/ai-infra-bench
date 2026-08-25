# vLLM multimodal encoder-cache compaction task

This directory is a Harbor schema 1.3 task for the vLLM multimodal encoder
cache placeholder/embedding-count contract. `instruction.md` is the only task
description intended for solving agents; this README documents packaging and
validation.

## Packaging protocol

- The candidate source is fetched at image-build time from the exact public
  base commit archive in `environment/lock/sources.lock.json`.
- The outer archive SHA-256, upstream Git tree, every tracked file, executable
  bit and aggregate semantic tree are verified before compilation.
- Python dependencies are installed from public PyPI using a complete
  `--require-hashes` lock.
- vLLM native extensions are rebuilt for SM80 in a build-only copy of the
  verified base source and seven hash-pinned public source archives. `/app` is
  materialized from the untouched pristine copy, then receives only the 51
  manifest-declared runtime paths.
- Agent and separate-verifier runtime networking remains disabled.
- The agent image contains one editable, source-only synthetic Git commit under
  `/app`; generated runtime paths are exactly ignored and neither tests nor the
  solution are present.
- Harbor restores the submitted artifact at its original source path `/app` in
  the independent verifier environment.

See `environment/lock/ARTIFACTS.md` for the full supply-chain contract.

## Local builds

```bash
bash environment/build_images.sh
```

This produces the stable local recipe tags:

- `ai-infra-bench/vllm-mm-encoder-cache-compaction-agent:oss`
- `ai-infra-bench/vllm-mm-encoder-cache-compaction-verifier:oss`

The agent build may access only the public sources declared by the manifests.
The verifier build uses `--network none`. Both runtime environments use
`network_mode = "no-network"`.

## Verifier outputs

`/logs/verifier/reward.json` is deliberately binary: it contains only
`{"reward": 1.0}` when all declared correctness requirements pass, otherwise
`{"reward": 0.0}`. Continuous completion, group scores, validity and
infrastructure diagnostics are retained separately in `scoring.json`.

This task is released as a recipe-only/local-build task and intentionally has
no `metadata.image_digest`. No task image is published to a registry. Local
image IDs are recorded only as validation evidence and are never represented
as OCI registry RepoDigests.
