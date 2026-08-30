# Separate incompatible persistent KV-cache layouts

Persistent KV-offload files may be reused across parallel configurations when
their layout is classified as parallelism-agnostic. A model runner with a
different serialized tensor order can have the same per-block byte count and
single-rank dimensions, so size and parallelism fields alone do not prevent it
from reading incompatible cached bytes and silently producing incorrect output.

Include layout compatibility in the persistent namespace identity. Layouts
that are not safe to share across parallel configurations must not collide with
the legacy parallelism-agnostic namespace, while genuinely agnostic layouts
must continue to collapse equivalent TP/PP configurations.

Work in `/workspace/vllm`. Leave the source change in the working tree. Do not
modify task metadata or verifier files.
