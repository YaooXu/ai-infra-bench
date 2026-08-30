# Keep stale speculative frames from corrupting async placeholder accounting

With asynchronous scheduling, resetting the prefix cache can force-preempt a
running request while model-runner frames are still in flight. The request is
then resumed with fresh output placeholders and a positive
`async_tokens_to_discard` count. When one of those stale frames contains
rejected speculative tokens, its pre-reset rejection accounting must not be
applied to the resumed request before the frame is discarded; doing so can make
the placeholder count negative and also corrupt computed-token accounting.

Update the scheduler so a stale speculative frame that is pending discard
leaves the resumed request's placeholder and computed-token counters unchanged,
decrements the discard count exactly once through the existing async discard
path, and keeps the request running. Preserve the existing speculative
rejection behavior for ordinary frames whose discard count is zero.

Work in `/workspace/vllm`. Leave the source change in the working tree. Do not
modify task metadata or verifier files.
