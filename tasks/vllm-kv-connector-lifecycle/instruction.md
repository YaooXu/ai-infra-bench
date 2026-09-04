Work in `/workspace/repo`.

I maintain an external KV connector and am updating it for vLLM v0.12. While
testing the migration, I found that the engine can still fall back to the
retired connector constructor. An old plugin therefore appears to load and
only fails later during initialization, which hides the actual migration error.

Please finish removing the pre-v0.12 constructor API. A current connector must
receive the engine's resolved KV-cache configuration when it is created. A
plugin that still uses the retired constructor should fail before its
initialization code runs and give its author useful migration information,
rather than being invoked through a compatibility fallback.

I also need current connectors to be constructed exactly once per engine
initialization with the same configuration object the engine resolved. Normal
Python argument errors and exceptions raised inside a current plugin must stay
distinguishable and must not trigger a compatibility retry. Existing in-tree
connectors must continue to work.

I do not have target files or tests to provide. Please update the lifecycle
coherently and add plugin-facing regression coverage.
