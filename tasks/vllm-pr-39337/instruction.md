Work in `/workspace/repo`.

Add automatic selection between the production Model Runner V1 and V2.

The existing `VLLM_USE_V2_MODEL_RUNNER` setting becomes a three-way user
choice: `0` selects V1, `1` selects V2, and leaving it unset lets vLLM choose
from the resolved serving configuration. For the initial rollout, supported
dense, unquantized Qwen3 text-generation configurations should automatically
use V2; configurations V2 does not support must remain on V1. Explicitly
requesting V2 for an unsupported configuration must explain the incompatibility
instead of silently changing the user's choice.

One engine startup must use a consistent selection in every production
consumer. No target module or test matrix is supplied. Implement the
user-visible behavior and construct configuration and consumer tests.
