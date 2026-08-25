# Public build and artifact contract

This is a recipe-only/local-build task. It is built only from its committed
Dockerfiles and public, immutable inputs; no prebuilt task image is published
or required.

## Network boundary

- **Agent image build:** public network access is allowed for the two
  digest-pinned base images, PyPI and the exact GitHub codeload URLs in the
  source manifests. No APT repository is used.
- **Verifier image build:** `no-network`; it adds trusted tests and the scorer
  to the already-built local agent image.
- **Agent runtime:** `no-network`.
- **Verifier runtime:** `no-network`, in a separate environment.

The public `environment/docker-compose.yaml` and `tests/docker-compose.yaml`
match the resource and network policy declared by `task.toml`.

There is no network fallback after a hash mismatch. In particular, the outer
SHA-256 of the GitHub base archive is strict. Even if its extracted semantic
tree appears unchanged, a different archive hash fails the build until a
maintainer independently reviews and updates the lock.

## Base source

`sources.lock.json` binds the vLLM repository to base commit
`676db55eecf8b6d9ec38ea243cf6f35ea8378ec6` at four levels:

1. an exact public codeload URL and archive SHA-256;
2. the upstream Git tree identity;
3. 3,575 path, size, content-SHA256 and executable-bit records;
4. an aggregate semantic tree SHA-256.

`scripts/fetch_base_source.py` rejects absolute paths, traversal, duplicate
entries, links, devices and unlisted files before extraction. Regular file
modes are normalized to Git's canonical `100644`/`100755` semantics. The
extracted tree must exactly match the locked file and aggregate tree manifests.

The final image deletes the downloaded archive and builder tree by starting a
new runtime stage. Compilation runs only in `/opt/build-source`; the immutable
`/opt/base-pristine` tree is reverified after compilation. `/app` first receives
only that clean public source and creates exactly one synthetic Git commit with
no remote, tags, reflog or upstream objects. The 51 generated runtime paths are
overlaid afterwards and listed exactly in `.git/info/exclude`, so native
binaries and wheel-generated files are available at runtime but are not part of
the synthetic source commit.

## Python dependencies

`requirements.lock` pins all 178 direct and transitive distributions to exact
versions and SHA-256 hashes. The image build runs pip in hash-checking mode
against `https://pypi.org/simple`. `wheelhouse-manifest.json` records the exact
public wheel filenames audited against PyPI. A changed or missing distribution
fails the build.

`base-requirements.cuda.txt` is an immutable byte-for-byte copy of the base
commit's `requirements/cuda.txt` and records the authoring-time resolver input.
The Docker build consumes only the fully resolved `requirements.lock`. vLLM is
explicitly forbidden from the PyPI closure because it is compiled from the
locked base source during the same image build.

The locked closure is installed into a fresh venv without system-site-package
inheritance. The digest-pinned base therefore cannot silently satisfy a Python
requirement with an unverified preinstalled distribution. The ensurepip seed of
setuptools is removed before installation so only the locked distribution and
metadata remain.

## Native build

The native builder uses the digest-pinned CUDA/PyTorch devel image and:

1. verifies all committed source, dependency and runtime declarations;
2. fetches and strictly verifies the base commit archive;
3. downloads the seven archives in `native-build-deps-manifest.json`, checking
   every SHA-256 before safe extraction;
4. builds a minimal Git 2.47.2 without network transports;
5. copies the immutable pristine tree to a distinct build tree and compiles an
   SM80 vLLM wheel only in that build tree;
6. extracts only the 51 generated/runtime paths in `native-manifest.json`;
7. emits and checks `native-build-manifest.json` in the runtime image.

The runtime-tree preparation rejects `build`, `dist`, `*.egg-info`, Python
caches and pytest caches. Release validation also measures the exact excluded
`tar czf` operation used by Harbor v0.20: one archive must complete within 60
seconds, and three concurrent archives must remain below Harbor's fixed
120-second transfer timeout.

No precompiled `.so` is accepted from the host or arbitrary `site-packages`.
`native-source-binding.json` binds the 258 native build inputs to an aggregate
SHA-256, while `prepare_runtime_tree.py` rejects a source-built wheel missing
any required extension. Optional modules produced by a full build remain in
the builder stage and are recorded rather than copied.

Docker build stages do not receive NVIDIA driver injection, so the build checks
extension location and metadata without loading `libcuda.so.1`. The verifier
imports the candidate Python modules from `/app`; native-extension loading is
recorded separately as runtime diagnostic evidence.

## Reproducible local build order

From the task directory:

```bash
bash environment/build_images.sh
```

Equivalent explicit commands are:

```bash
docker build --network default --no-cache --pull=false \
  --file environment/Dockerfile \
  --tag ai-infra-bench/vllm-mm-encoder-cache-compaction-agent:oss \
  environment

docker build --network none --no-cache --pull=false \
  --build-arg AGENT_IMAGE=ai-infra-bench/vllm-mm-encoder-cache-compaction-agent:oss \
  --file tests/Dockerfile \
  --tag ai-infra-bench/vllm-mm-encoder-cache-compaction-verifier:oss \
  tests
```

The first build needs public network access. The second build performs no
network request. Both resulting runtime environments run with
`network_mode = "no-network"`. `tests/Dockerfile` defaults to the stable local
agent tag, and the separate verifier contains trusted tests/scoring code while
the agent image contains neither `tests/` nor `solution/`.

`build_images.sh` forwards the standard Docker proxy build arguments when they
are already present in the caller's environment. A controlled builder may also
override `VLLM_MM_CACHE_BASE_IMAGE` and
`VLLM_MM_CACHE_RUNTIME_BASE_IMAGE` to an anonymous local mirror reference for
the same public manifests. The Dockerfile independently rejects either value
unless it ends in the exact frozen SHA-256 digest above; this does not permit a
different or mutable base image.

The stable tags identify locally built recipe outputs only. This release does
not publish task images and therefore omits `metadata.image_digest`. Local image
IDs, `docker save` hashes and archive hashes are validation evidence, not OCI
registry RepoDigests.
