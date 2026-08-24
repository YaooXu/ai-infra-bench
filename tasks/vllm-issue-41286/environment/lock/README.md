# Environment lock: intentionally blocked

No publishable Agent Dockerfile or dependency lock is emitted for
`vllm__issue__41286`.

This survey item is an open, continuously edited migration roadmap. It has no
`base_sha`, `head_sha`, closing PR, or unique hardware/model/topology. Linked
children span many unrelated source bases and independently migrate dense, MoE,
quantized, pooling, multimodal, speculative-decoding, connector, compile, and
distributed paths. Choosing any child or a time-inferred repository HEAD would
silently redefine the issue-level task and cannot satisfy an exact-base or
canonical-tree contract.

The missing Dockerfile is deliberate. It prevents a valid child environment or
a generic A100 import probe from being mislabeled as the Agent image for the
whole migration program. This task directory contains no solved source, model
weights, Oracle patch, or public test that could pass vacuously.

Unblocking requires a curator to select one atomic child and supply:

- exact `base_sha` and expected canonical root tree;
- one mapped merged solution SHA;
- a model/tokenizer artifact digest;
- supported GPU and TP/PP/connector topology;
- a failing Base and passing Oracle behavior contract.

PR #39337 already has its own narrow config-contract survey environment. It is
one `[1/N]` migration step and must not be reused as the umbrella issue gold.
