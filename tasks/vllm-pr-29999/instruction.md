Work in `/workspace/repo`.

Fix the spurious `Current vLLM config is not set.` warning emitted while modular fused-MoE kernels allocate profile buffers in a valid DP+EP configuration. The relevant parallel configuration exists when the kernel is constructed, but the process-global vLLM-config context may have ended by the time the real CUDA profile forward runs.

A valid configuration must not warn in that lifecycle. Genuine access with no available configuration must continue to emit the warning. Do not globally suppress, downgrade, or filter the diagnostic, and preserve the actual Triton MoE forward behavior.

