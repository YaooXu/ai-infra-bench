# Environment sources and lock rationale

This environment represents the parent of vLLM PR 30475:
`676db55eecf8b6d9ec38ea243cf6f35ea8378ec6`. The immutable GitHub commit
archive is verified with SHA-256 before extraction, and its canonical Git tree
must equal `aafa39e6544cbeaf83b72985f83aaefd4e9e3456`.

The nearest official release environment is `vllm/vllm-openai:v0.11.2`.
Docker uses its linux/amd64 platform manifest directly:
`sha256:47a9896f86818fea323b2d38082758c62d9a0155d6fe6c4dbd7d735c556f680a`.
The Dockerfile does not resolve the mutable tag or the multi-platform manifest
list at build time.

Python dependencies and CUDA libraries are inherited unchanged from that
digest-pinned official image. Candidate Python source comes only from the
verified base commit. Compiled vLLM shared objects come from the official
v0.11.2 image, are copied into `/app/vllm`, and are ignored as generated
artifacts. The generated release `_version.py` is copied by the same exact-path
overlay; no other wheel Python source is overlaid. The Docker build checks that
both `vllm.__file__` and `vllm._C` resolve under `/app`, while PyTorch is 2.9.x
with CUDA 12.9. The validated official image reports PyTorch `2.9.0+cu129`.

The runtime is configured offline and must additionally be launched with
`--network none`. No model or dataset is required for the structural baseline
reproduction: it uses production `PlaceholderRange` and `EncoderCacheManager`
classes with a sparse `P=100`, `E=8` placeholder.

This simplified survey environment does not rebuild native extensions from the
exact base commit. They are inherited from the nearest digest-pinned release,
v0.11.2. That tradeoff is acceptable for this task's Python-only PR surface and
is guarded by import/ABI smoke tests, but it is not interchangeable with the
full source-built environment when a task modifies C++/CUDA code or depends on
commit-specific native behavior.
