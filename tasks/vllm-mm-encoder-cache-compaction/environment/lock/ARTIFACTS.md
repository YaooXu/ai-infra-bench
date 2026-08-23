# Public build and artifact contract

This task is built only from its committed recipe and public, immutable inputs.
A clean clone does **not** need the curator's historical 5.3 GB wheel/native
cache. That cache may remain available to curators for offline recovery, but it
is neither a build input nor a release artifact.

## Network boundary

- **Image build:** public network access is allowed for the two digest-pinned
  base images, PyPI and the exact GitHub codeload URLs in the source manifests.
  No APT repository is used.
- **Agent runtime:** `no-network`.
- **Verifier runtime:** `no-network`, in a separate environment.
- **Curator offline cache:** optional and outside the public task.

There is no network fallback after a hash mismatch. In particular, the outer
SHA-256 of the GitHub base archive is strict. Even if its extracted semantic
tree appears unchanged, a different archive hash fails the build until a
curator independently reviews and updates the lock.

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
previous vendored starter had identical bytes and executable bits; its only raw
mode difference was non-semantic group-write permission introduced by the local
copy. The public archive tree and the old starter are otherwise zero-difference.

The final image deletes the downloaded archive and builder tree by starting a
new runtime stage. `/app` receives the verified source plus only the generated
runtime artifacts, then creates exactly one synthetic Git commit with no
remote, tags, reflog or upstream objects.

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
5. compiles an SM80 vLLM wheel from the verified base source;
6. extracts only the 51 generated/runtime paths in `native-manifest.json`;
7. emits and checks `native-build-manifest.json` in the runtime image.

No precompiled `.so` is accepted from the host, the curator cache or arbitrary
`site-packages`. `native-source-binding.json` binds the 258 native build inputs
to an aggregate SHA-256, while `prepare_runtime_tree.py` rejects a source-built
wheel missing any required extension. Optional modules produced by a full build
remain in the builder stage and are recorded rather than copied.

Docker build stages do not receive NVIDIA driver injection, so the build checks
extension location and metadata without loading `libcuda.so.1`. The no-network
A100 runtime smoke imports `vllm._C` with GPU injection and is the authoritative
ABI/driver gate.

## Reproducible local build order

From the task directory:

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

The first build needs public network access; both resulting runtime environments
run with `network_mode = "no-network"`. The verifier build only adds the frozen
tests to the exact agent image and performs no network request.

Local tags and image IDs are development evidence only. `metadata.image_digest`
stays empty until a maintainer pushes the release image and records its OCI
registry manifest digest. After that, the agent RepoDigest becomes the default
`AGENT_IMAGE` in `tests/Dockerfile`; a prebuilt verifier RepoDigest may also be
declared in `[verifier.environment]` if that is the repository's chosen release
mode. Harbor 0.20 does not require that table to build `tests/Dockerfile` in
separate mode when using recipe-based environments.
