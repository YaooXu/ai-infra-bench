# Avoid pruning Mooncake receive masks twice with Eagle

When Mooncake external KV loading is used with Eagle or MTP speculative
decoding, lookup already shortens the reported cache hit so the final block is
recomputed locally. The receive-side load-mask calculation applies the same
last-block adjustment again, leaving one processable chunk without a mask slot
and silently skipping its KV transfer.

Keep the lookup-stage Eagle adjustment, but ensure the receive mask covers
every chunk in the already-adjusted token range. Preserve behavior without
Eagle and preserve hybrid full-attention plus sliding-window mask semantics.

Work in `/workspace/vllm`. Leave the source change in the working tree. Do not
modify task metadata or verifier files.
