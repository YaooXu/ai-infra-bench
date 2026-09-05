We keep a request alive as a streaming session and send each continuation
through the normal new-request scheduler channel with the same request ID.
After the second or third update, the GPU runner can lose cached state, leave a
duplicate or stale persistent-batch row, or retain output tokens that have
already become part of the next prompt.

Please make the production `GPUModelRunner` lifecycle treat multiple updates
as one continuous session while leaving ordinary new requests unchanged.
Prompt-token and prompt-embedding continuations, multimodal
metadata, sampling and pooling state, block state, computed-token accounting,
output-token absorption, and M-RoPE metadata must all stay consistent.

The change is limited to the `GPUModelRunner` continuation-state slice; it does
not need to implement the rest of the user-facing Streaming Session stack
across the API, scheduler, or input/output processors.

Run `/workspace/public_dev/run.sh` to reproduce three continuations of one
session. The same behavior must hold when two sessions are interleaved, when
different numbers of output tokens are absorbed, with either prompt
representation, and with randomized request IDs. Keep the persistent batch and
the production update lifecycle in place rather than special-casing the public
example.
