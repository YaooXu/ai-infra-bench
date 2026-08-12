# Tasks

Validated Harbor tasks will live in this directory. No task is considered release-ready until it passes the quality gates in the [benchmark design](../docs/BENCHMARK_DESIGN.md).

Task lifecycle:

```text
sourced -> scoped -> reproduced -> packaged -> verifier_ready
        -> human_solved -> maintainer_approved -> validated -> frozen
```

Use [`templates/harbor-task`](../templates/harbor-task/) when creating a new task. Source recommendations and partially investigated candidates remain under `data/` or curator working manifests; they do not belong here.

