# Preserve exact token accounting for asynchronous KV loads

When an asynchronous KV connector reports an external cache hit that ends in
the middle of a block, the scheduler can later infer the computed-token count
from the number of allocated blocks. That rounds the hit forward past the
tokens actually loaded and can make execution consume KV data that was never
received.

Carry the connector's exact computed-token count through the asynchronous wait
and completion lifecycle. A partial-block hit must remain exact when the
request becomes runnable, while full-prompt hits must retain the existing
last-token recomputation behavior and load failures must remain recoverable.

Work in `/workspace/vllm`. Leave the source change in the working tree. Do not
modify task metadata or verifier files.
