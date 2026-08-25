Work in `/workspace/repo`.

Move Model Runner V2 selection from a process-wide boolean into `VllmConfig` and propagate the resolved decision to the production GPU-worker consumer.

`VLLM_USE_V2_MODEL_RUNNER` must be tri-state: explicit `0` forces V1, explicit `1` forces V2, and an unset variable delegates to configuration defaults. In this migration step, only dense, unquantized `Qwen3ForCausalLM` generation models default to V2. Non-whitelisted architectures, MoE, quantized, pooling, and other unsupported configurations must remain on V1. When V2 is explicitly forced, unsupported features must still raise a useful validation error rather than silently falling back.

The GPU worker must use the resolved `VllmConfig` decision instead of rereading the environment, so an unset variable can select V2 and unsupported cases can fall back consistently. Preserve both explicit override directions. Scheduler migration, Distributed FlashInfer/NIXL behavior, and model-weight inference are outside this focused task.
