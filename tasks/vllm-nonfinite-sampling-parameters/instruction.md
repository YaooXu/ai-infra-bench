# Reject non-finite sampling parameters

Sampling parameters currently allow some IEEE-754 non-finite values to bypass
ordinary range comparisons. Those values can propagate into execution
backends and cause undefined behavior or runtime failures.

Reject `NaN`, positive infinity, and negative infinity for `temperature` and
`repetition_penalty` during parameter validation. Finite values satisfying the
existing range contract must continue to work, and existing exception types
and messages should remain consistent with nearby validation failures.

Work in `/workspace/vllm`. Leave the source change in the working tree. Do not
modify task metadata or verifier files.
