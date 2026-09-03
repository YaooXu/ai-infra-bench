Work in `/app`.

We use vLLM for disaggregated serving with asynchronous remote KV transfer. I
noticed that the scheduler consumes a surprising amount of CPU whenever a
large batch is waiting for remote KV data. If no transfer has completed, the
engine has no useful work to do, but each scheduler tick still becomes slower
as I increase the number of waiting requests. From tracing the connector I can
also see the same requests being checked repeatedly even though their external
state has not changed.

Please investigate and fix this overhead. An idle scheduling tick should not
do work proportional to the number of requests whose remote dependency is
unchanged. Once the connector reports completion or failure, the affected
requests must still resume or recover in the expected order. Please avoid
regressions in request accounting and in the existing abort,
structured-output, streaming-input, and prefix-cache paths.

I do not have a small reproduction script. Add a deterministic regression test
for the idle overhead and relevant lifecycle behavior, and keep the change
narrow enough for production use.
