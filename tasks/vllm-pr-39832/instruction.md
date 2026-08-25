# Enforce the current external KV connector lifecycle

External v1 KV connectors must use the current constructor contract and receive
the engine's `KVCacheConfig` as their third constructor argument. The deprecated
two-argument compatibility path must be removed.

Update the production factory, base connector, and KV-transfer initialization
lifecycle so that:

- a current connector is created by the real KV-transfer consumer exactly once
  and receives the same configuration object;
- an external two-argument connector is rejected before its constructor runs;
- omitting the third argument from the base lifecycle raises `TypeError`; and
- a `TypeError` raised inside a current connector is propagated unchanged and
  is not mistaken for an old signature or retried.

Do not retain the old path behind warnings, catch and rewrite connector-internal
errors, or special-case the verifier.
