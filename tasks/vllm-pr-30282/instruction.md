Work in `/workspace/repo`.

Refactor modular fused-MoE configuration ownership so every modular kernel can
receive the layer's explicit `FusedMoEParallelConfig`. The kernel must preserve
that exact configuration object and derive the DP+EP decision from its MoE
fields. Construction without a configuration remains a supported non-DP+EP
compatibility entry and must not depend on process-global vLLM configuration.

Keep the real Triton MoE path numerically unchanged for an ordinary single-rank
configuration. Propagate the explicit configuration through the modular method
and affected modular backends, while leaving non-modular functional backends
free of the superseded generic `ParallelConfig` argument. The old
`parallel_config=` kernel keyword must no longer be accepted.
