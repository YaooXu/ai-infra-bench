Complete the announced removal of the pre-v0.12 constructor API for external KV
connector plugins. A connector written for the current API receives the
engine's KV-cache configuration when it is created. A plugin that still uses
the retired constructor must fail before its initialization code runs, with a
migration-oriented error, rather than being invoked through a compatibility
fallback.

The engine must create a current connector once per initialization and pass the
same configuration object it resolved. Normal Python argument errors and
exceptions raised inside a current plugin must remain distinguishable and must
not cause a compatibility retry. In-tree connectors must keep working.

No target files or tests are supplied. Update the public lifecycle coherently
and add plugin-author-facing regression coverage.
