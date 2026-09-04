Work in `/workspace/repo`.

We are starting to roll out Model Runner V2, but asking every operator to know
when it is safe and set `VLLM_USE_V2_MODEL_RUNNER` manually is proving fragile.
Could vLLM choose the runner from the resolved serving configuration when the
setting is absent?

For the initial rollout, supported dense, unquantized Qwen3 text-generation
configurations should choose V2 automatically, while configurations V2 cannot
run should stay on V1. We still need an explicit escape hatch: `0` must select
V1 and `1` must select V2. If a user explicitly asks for V2 with an unsupported
configuration, startup should explain the incompatibility rather than silently
changing the choice.

One engine startup must use a consistent selection in every production
consumer. No target module or test matrix is supplied. Implement the
user-visible behavior and construct configuration and consumer tests.
