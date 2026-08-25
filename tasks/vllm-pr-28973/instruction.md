# Streaming-session continuation in the GPU runner

When a scheduler emits a new-request record for an ID that is already cached by
the GPU model runner, the record represents continuation of the same streaming
session rather than a distinct request.

Update the production runner lifecycle so that this continuation:

- preserves the existing cached request object;
- removes its stale persistent-batch entry before it is re-added;
- refreshes prompt tokens or embeddings, multimodal features, sampling and
  pooling parameters, block IDs, and computed-token count from the new record;
- recalculates the prompt length; and
- clears output tokens that have become part of the continued prompt.

Ordinary new requests and runners without M-RoPE must continue to work. Do not
disable the persistent batch, bypass the production update lifecycle, or
special-case the verifier.
