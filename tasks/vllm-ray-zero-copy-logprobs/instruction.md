# Release Ray shared-memory ownership from returned logprobs

Ray compiled-DAG channels may deserialize NumPy logprob arrays as read-only
zero-copy views backed by Ray shared memory. If model-runner outputs retain
those views after a result is returned to the scheduler, later channel reads
can stall while trying to acquire the shared-memory read lock.

Ensure outputs leaving Ray executor result boundaries no longer retain
read-only NumPy logprob buffers owned by the channel. Cover blocking and
non-blocking execution, with and without KV-output aggregation. Preserve array
values and leave ordinary writable arrays and unrelated output fields intact.

Work in `/workspace/vllm`. Leave the source change in the working tree. Do not
modify task metadata or verifier files.
