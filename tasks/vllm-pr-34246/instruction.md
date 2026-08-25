Work in `/workspace/repo`.

Fix multimodal embedding merging so position masks may remain on CPU while the input and multimodal embeddings are CUDA tensors. The merge must preserve exact placeholder order and dtype, update the input tensor in place, avoid a CUDA synchronization, and avoid a target-sized temporary CUDA allocation. Invalid placeholder/embedding cardinalities must raise a precise error instead of truncating values.

Update the affected multimodal model and runner consumers so they preserve this CPU-mask contract. Do not move the mask to CUDA as a workaround, disable synchronization checks, bypass the production merge helper, or weaken cardinality validation.

Keep the change narrowly scoped and leave the repository buildable.
