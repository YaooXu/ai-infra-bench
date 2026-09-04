# vLLM Model Runner V2 selection

## What the task asks

Implement tri-state behavior for `VLLM_USE_V2_MODEL_RUNNER` and propagate the resolved selection through `VllmConfig`/`GPUWorker` consistently.

## Environment

- Base image: pinned digest in `environment/Dockerfile`
- Workdir: `/workspace/repo`
- Runtime policy: offline (`network_mode = "no-network"`)
- GPU: one A100-class accelerator
- Agent budget: 10 hours

## Verifier

- A separate hidden verifier runs `/tests/verify_runner_consumers.py`
- Checks defaults, explicit overrides, incompatibility reporting, and real GPU worker consumption path
- Reward is written to `/logs/verifier/reward.txt`

## Layout

- `instruction.md`: user-facing behavioral contract
- `task.toml`: task config and resource constraints
- `environment/`: deterministic base and native runtime checks
- `solution/`: Oracle patch + solve script
- `tests/`: behavioral + hidden-mode checks
- `validation/`: control manifest and evidence for the frozen snapshot

## Run

- Oracle: `harbor run -p tasks/vllm-runner-v2-selection -a oracle`
- Agent: `harbor run -p tasks/vllm-runner-v2-selection -a agent -m claude-opus-4-8`
