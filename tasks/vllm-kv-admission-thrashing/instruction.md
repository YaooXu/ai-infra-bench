# Prevent KV-cache admission thrashing with long prompts

With chunked prefill enabled, the scheduler may admit a long request because
its first chunk fits even though the request's full input cannot fit in the
remaining KV cache. The request later exhausts the cache, is preempted, returns
to the queue, and repeats the same prefill work. Under load this can starve
decode traffic and collapse output throughput.

Add an admission policy that can account for the full input sequence while
respecting prefix-cache hits, already-computed tokens, encoder tokens, sliding
window limits, and the configured maximum model length. Preserve existing
scheduler behavior outside this admission decision and expose the policy
through the normal configuration path.

Work in `/workspace/vllm`. Leave the source change in the working tree. Do not
modify task metadata or verifier files.
