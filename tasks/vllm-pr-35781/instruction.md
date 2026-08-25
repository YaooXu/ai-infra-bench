Work in `/app`.

Optimize the vLLM scheduler so requests blocked on asynchronous dependencies are not repeatedly rescanned through the ordinary waiting queue on every idle scheduling round. Introduce an explicit blocked/skipped waiting path and promote requests only when their dependency becomes ready.

Preserve FCFS and priority ordering, request counts and statistics, abort/removal behavior, structured-output readiness, streaming-input behavior, and remote-KV failure recovery. Requests waiting for remote KV must remain visible as unfinished while avoiding per-round callback work proportional to the blocked queue size.

