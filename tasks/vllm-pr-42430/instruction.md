Work in `/workspace/repo`.

Fix Mamba attention metadata construction for uniform one-token batches in FULL CUDA-graph mode. A scheduler-labelled prefill row with query length one and an existing sequence state represents an update/decode and must use decode metadata. A true first-token prompt with sequence length one must remain a prefill.

Apply the distinction in the production Mamba metadata builder and preserve CUDA-backed persistent decode state. Do not reclassify every one-token prefill indiscriminately, and keep behavior compatible with ordinary decode and speculative configuration paths.

