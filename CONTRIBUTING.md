# Contributing

AI Infra Bench welcomes task proposals, environment work, verifiers, workload analysis, and evaluation tooling from inference-system maintainers and researchers.

## Propose a source

A useful source is a real issue or pull request with clear AI-infrastructure value. Please include:

- canonical repository and issue/PR URL;
- why the work is representative, memorable, or technically diagnostic;
- the workload and subsystem categories;
- known model, dependency, hardware, and checkpoint requirements;
- an informal human-effort estimate, kept as free text;
- known reproduction or leakage risks.

Survey evidence belongs in `data/vllm_survey_results.jsonl`. Do not include respondent email addresses or private discussion.

## Build a task

Start from [`templates/harbor-task/task.toml`](templates/harbor-task/task.toml) and follow the [benchmark design](docs/BENCHMARK_DESIGN.md). A task contribution should provide:

- a pre-solution repository state with no future Git objects;
- a solution-neutral instruction;
- a pinned offline environment;
- a stable reproducer;
- required and held-out execution tests;
- reference and plausible-wrong solutions for verifier QA;
- a hardware and checkpoint manifest;
- a short curation report explaining scope and risks.

## Validation gates

A task is not ready merely because the reference patch applies. Before review, demonstrate:

```text
base fails
reference solution passes
no-op fails
plausible wrong solution fails
human solve passes
maintainer review passes
```

Performance tasks must also quantify measurement variance and show that the reference improvement is larger than environmental noise.

Do not edit survey wording for style. Preserve respondent text under `survey`, and place curator inference under `benchmark`.

Before opening a pull request, confirm that JSONL records remain one valid JSON object per line, stable source IDs remain unique, public documentation links resolve, and no respondent identity or personal profile/comment link has been introduced.
