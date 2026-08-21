# PR 35781 Harbor test plan

This task evaluates the complete production change from vLLM PR 35781. The PR
reported unchanged GSM8K accuracy and an end-to-end total-token throughput
increase from 3411.78 to 3611.85 tok/s (about 5%) in a two-GPU P/D setup.

The Harbor task is deliberately CPU-only and network-isolated. It therefore
uses upstream CPU unit tests for correctness and deterministic hot-path work
proxies plus a paired CPU microbenchmark for performance. It does not claim to
reproduce the original two-GPU throughput number.

## Correctness test points

| ID | Test point | Main assertions | Source |
|---|---|---|---|
| C1 | Mixed FCFS ordering | FSM, remote-KV, streaming, regular and tail requests preserve order before and after promotion | Custom + original PR scheduler tests |
| C2 | Remote-KV promotion | Only completion-signaled requests are promoted; an FSM prefix remains ahead under FCFS | Original PR scheduler test |
| C3 | Priority scheduling | Priority and arrival-time ordering are preserved across blocked and ready queues; blocked high-priority work does not starve ready work | Custom |
| C4 | Remote prefill lifecycle | Allocation, waiting, completion, promotion, interleaving, cannot-receive and cannot-schedule transitions remain correct | Original PR remote lifecycle suite |
| C5 | Counts, stats and abort | Both waiting populations are counted; abort removes requests from every scheduler structure | Custom + original PR abort tests |
| C6 | Error propagation | Async invalid-block failures using the `fail` policy produce `FINISHED_ERROR` and clean queues | Original PR error propagation suite |
| C7 | Invalid-block correctness | Invalid async blocks are not spuriously cached and valid blocks keep correct cache state | Original PR invalid-block suite |
| C8 | KV-load recovery | Recompute policy handles single, multiple, shared and progressive invalid blocks without corrupting request state | Original PR load-failure suite |
| C9 | Unchanged regression | Existing basic scheduler behavior continues to pass | Upstream `test_schedule` |

Correctness is split into four normalized groups:

- ordering: 30%
- lifecycle: 25%
- recovery: 30%
- unchanged regressions: 15%

## Performance test points

| ID | Test point | Stable pass condition |
|---|---|---|
| P1 | Event-driven remote update work | Across 96 blocked requests and eight scheduler passes with no completion event, the expensive remote update routine is called zero times |
| P2 | Schedulable-queue churn | Across the same workload, the ordinary waiting queue performs zero blocked-request pops or prepends |
| P3 | Paired CPU microbenchmark | The scheduler hot path is compared with an explicit legacy per-request readiness loop in the same process; callback count must be zero and elapsed time must be below 50% of the paired legacy measurement |

The paired benchmark injects deterministic CPU work into the readiness callback.
This makes the improvement large enough to distinguish reliably while keeping
both measurements on the same host, interpreter and process. Operation-count
assertions P1 and P2 remain the primary, non-flaky performance evidence.

Performance is split into:

- deterministic hot-path work: 70%
- paired timing check: 30%

## Final score

```text
correctness = weighted(C1..C9 groups)
performance = 0.70 * hotpath + 0.30 * timing
reward      = 0.65 * correctness + 0.35 * performance
```

Every group score is the fraction of its JUnit cases that pass. The verifier
writes the aggregate and all submetrics to `/logs/verifier/reward.json`.

## Hidden upstream test installation

`tests/heldout/upstream-tests.patch` contains only the five test-file changes
from the original PR. At verifier startup, `grader.py` applies those changes to
the base snapshot's tests. The production fix is not present in this patch;
Oracle uses `solution/reference.patch`, while evaluated agents must implement
their own fix.
