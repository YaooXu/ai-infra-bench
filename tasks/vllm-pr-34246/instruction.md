Work in `/workspace/repo`.

We have started seeing CUDA out-of-memory failures when serving long,
multi-image Qwen3-VL requests. The traceback points to multimodal embeddings
being merged near the end of available VRAM, even though the allocation named
in the error is comparatively small. Eager execution fails as well, so turning
off CUDA graphs does not work around it. In this path the placeholder mask is
prepared on CPU while the token and multimodal embeddings are on CUDA.

Please reduce synchronization and temporary GPU-memory pressure in this merge.
The path where the mask is already on CUDA must keep working too, and placeholder
order, output dtype, in-place behavior, supported input forms, and useful
cardinality errors must not regress.

I do not have a small reproduction or a proposed implementation. Measure the
current behavior, choose a narrow fix, and add correctness and resource
regression coverage through the production path.
