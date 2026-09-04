Work in `/app`.

I use vLLM for disaggregated serving with asynchronous remote KV transfer. When
a large batch is waiting for remote KV data, I see unexpectedly high scheduler
CPU usage. If no transfer has completed there is no useful work to do, yet an
idle scheduler tick gets slower as I add waiting requests. Connector tracing
also shows the same requests being checked repeatedly even though their remote
state has not changed.

Please investigate and remove this idle overhead. A tick with no new remote
event should not do work proportional to the number of unchanged blocked
requests. When the connector reports completion or failure, the affected
requests must still resume or recover in the expected order. Request accounting,
abort handling, structured output, streaming input, and prefix caching must
continue to work.

I do not have a reduced script. Please construct a deterministic regression
test for the idle behavior and its lifecycle boundaries, then keep the
production change as focused as possible.
