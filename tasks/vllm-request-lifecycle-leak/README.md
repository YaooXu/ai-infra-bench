# vLLM request lifecycle retention

## What the Agent does

Ensure completed requests release request-owned multimodal payloads promptly without breaking prefix-cache bookkeeping. The user-facing contract is in [instruction.md](instruction.md).

## Environment

A digest-pinned vLLM CPU image with the exact Base source, offline runtime, and a 10-hour Agent budget. Its image must be rebuilt after removing Agent-visible reproduction assets.

## Verifier

The separate hidden verifier drives the production request-completion lifecycle and checks prompt-hash behavior plus prompt and multimodal object reclamation. Full credit is binary and is written to `/logs/verifier/reward.txt`.

## Layout

- `instruction.md`: user-facing behavioral request.
- `task.toml`: Harbor metadata, resources, isolation, and artifact paths.
- `environment/`: exact Base source image and dependency provenance.
- `solution/`: Oracle patch and application script, hidden from the Agent.
- `tests/`: separate-verifier entrypoint and behavioral checks.
- `validation/`: control manifest and evidence for the frozen snapshot.

## Running

After the pending image and validation records are finalized:

```bash
harbor run -p tasks/vllm-request-lifecycle-leak -a oracle
harbor run -p tasks/vllm-request-lifecycle-leak -a terminus-2 -m anthropic/claude-opus-4-8
```

