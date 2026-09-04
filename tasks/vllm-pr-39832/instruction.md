We maintain an external KV connector and are updating it for vLLM v0.12. During
that work I found that the engine can still fall back to the retired connector
constructor, even though its removal was already announced. That makes an old
plugin appear to load and then fail later in initialization, which hides the
actual migration mistake.

Please complete the removal of the pre-v0.12 constructor API. A connector using
the current API should receive the engine's KV-cache configuration when it is
created. A plugin that still uses the retired constructor should fail before
its initialization code runs and tell its author how to migrate, instead of
being invoked through a compatibility fallback.

The engine must create a current connector once per initialization and pass the
same configuration object it resolved. Normal Python argument errors and
exceptions raised inside a current plugin must remain distinguishable and must
not cause a compatibility retry. In-tree connectors must keep working.

No target files or tests are supplied. Update the public lifecycle coherently
and add plugin-author-facing regression coverage.
