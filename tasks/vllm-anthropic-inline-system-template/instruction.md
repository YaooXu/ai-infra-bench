# Adapt Anthropic inline system messages to the selected chat template

Anthropic requests may contain `system` messages inside the conversation, not
only in the top-level `system` field. Some model chat templates accept those
messages in place, while others reject every system message that is not first.
The server currently preserves all inline system messages, so a valid Anthropic
request can fail during rendering with the latter templates.

At service initialization, determine whether the configured chat template can
render a conversation containing a system message after a user turn. Treat a
missing template or a template-rendering error conservatively as requiring a
merge. For templates that require system-first ordering, merge non-billing
inline system text into the leading top-level system block and omit the original
inline entries. For templates that accept inline system messages, preserve their
positions so existing prefix-cache behavior is retained.

Apply the selected behavior consistently to both message generation and token
counting requests. Preserve existing billing-header filtering and conversion of
non-system messages.

Work in `/workspace/vllm`. Leave the source change in the working tree. Do not
modify task metadata or verifier files.
