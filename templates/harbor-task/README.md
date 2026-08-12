# Harbor Task Template

Copy this directory to `tasks/<task-id>/` when materializing a validated task.

The included `task.toml` defines benchmark-specific metadata and Harbor resource requirements. Add:

```text
instruction.md
environment/Dockerfile
environment/lock/
solution/solve.sh
tests/test.sh
tests/required/
tests/heldout/
```

Reference-solution identifiers and held-out test details should remain in curator-only storage until the task is released. The agent environment must contain only the clean base state and offline dependencies.

