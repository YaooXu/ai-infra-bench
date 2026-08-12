# AI Infra Bench

AI Infra Bench measures how much real AI-inference engineering work frontier coding agents can complete.

We are building a public benchmark from real vLLM maintainer workloads. The first release targets 100 expert-reviewed tasks:

- 76 representative tasks sampled from day-to-day bugs, features, performance work, refactors, and tests;
- 24 memorable tasks nominated by maintainers;
- reproducible, offline environments with execution-based correctness and performance verification;
- coverage of single-GPU, multi-GPU, distributed serving, kernels, and heterogeneous accelerators.

## Models and harnesses

| Harness | Models |
| --- | --- |
| Claude Code | Claude Opus 5, Hunyuan 3, Qwen 3.8 Max, Kimi K3, GLM 5.2, MiniMax M3 |
| Codex | GPT-5.6 |
| mini-swe-agent | Common-harness baseline across supported models |

Each release will freeze the exact model IDs, harness versions, budgets, and sampling settings.

## Links

- [Project notes and maintainer survey](https://docs.google.com/document/d/16E7Xm08JTIwKT6pbC5YezCEu-UbbNDzA1PTjk0ZH724/edit?usp=sharing)
- [Benchmark design](docs/BENCHMARK_DESIGN.md)
- [Dataset card](docs/DATASET_CARD.md)
- [Contributing](CONTRIBUTING.md)

## License

Software and documentation are licensed under [Apache-2.0](LICENSE). The survey dataset under `data/` is licensed under [CC BY 4.0](data/LICENSE).
