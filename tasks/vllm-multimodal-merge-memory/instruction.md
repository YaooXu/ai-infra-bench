Work in `/workspace/repo`.

I am serving long, multi-image Qwen3-VL requests close to the GPU memory limit,
and some of them now fail with CUDA out-of-memory errors while the multimodal
embeddings are being merged. The allocation named in the traceback is fairly
small, and eager mode fails too, so disabling CUDA graphs has not helped. In
this workload the placeholder mask is on CPU while the token and multimodal
embeddings are on CUDA.

Please reduce the synchronization and temporary GPU-memory pressure of this
operation. The case where the mask is already on CUDA must continue to work,
and placeholder order, output dtype, in-place behavior, supported input forms,
and useful cardinality errors must not regress.

I do not have a small reproduction or an implementation in mind. Please measure
the current behavior, choose a focused fix, and add correctness and resource
regression coverage through the production path.
