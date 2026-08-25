Work in `/workspace/repo`.

Add a native CUDA implementation of INT8 per-token-group quantization and expose it through the existing vLLM `_C` operator namespace. CUDA callers of `per_token_group_quant_int8` should use the native operator; unsupported platforms must retain the existing Triton path.

Support contiguous FP16, BF16, and FP32 inputs, configurable group size, epsilon, and INT8 bounds. Quantized values may differ from the deterministic Triton/reference result by at most one integer step, and scales must remain numerically equivalent. Rebuild the exact candidate `_C` extension after modifying native sources.

