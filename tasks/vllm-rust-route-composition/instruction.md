# Add an extension point for the finalized Rust HTTP router

The Rust OpenAI-compatible server needs an opt-in way for embedding callers to
add routes after the standard Axum router has been assembled, without exposing
internal application state or changing the behavior of the existing `serve`
entry point.

Provide a public asynchronous `serve_with_router_extension` API. It must accept
the normal server configuration and shutdown token plus a one-shot
`Router -> Router` callback. The existing `serve` API must remain source
compatible and behave as the identity-extension case.

Work in `/workspace/vllm`. Leave the source change in the working tree. Do not
modify task metadata or verifier files.
