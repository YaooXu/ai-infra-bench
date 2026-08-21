# Reduce scheduler overhead for blocked waiting requests

In P/D disaggregated serving with asynchronous remote KV loading, requests can
remain blocked for many consecutive scheduler steps.

The scheduler currently revisits and moves temporarily blocked requests during
every scheduling iteration even when their readiness state has not changed.
Under high concurrency and high remote-KV latency, this causes significant
scheduler CPU overhead and unnecessary waiting-queue churn.

Optimize the scheduler's handling of blocked waiting requests.

The hot path should avoid performing remote-KV readiness/update work for every
blocked request when no completion event has arrived, and blocked requests
should not repeatedly disturb the queue used for immediately schedulable work.

The implementation must preserve:

- FCFS and priority scheduling order across all waiting request types.
- Correct handling and promotion of requests waiting for remote KVs.
- Correct handling of requests waiting for FSM compilation or streaming input.
- Request abort and cleanup behavior.
- KV load failure and invalid-block recovery.
- Waiting-request counts and scheduler statistics.

Do not change externally visible request scheduling semantics. Relevant
CPU-only regression tests must continue to pass.
