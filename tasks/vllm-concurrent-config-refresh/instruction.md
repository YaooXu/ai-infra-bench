# Tolerate transient configuration disappearance during cache refresh

Several API-server processes may load the same model configuration while a
shared Hugging Face cache entry is being refreshed. The cache refresh briefly
replaces a symlink non-atomically, so one process can observe a missing or empty
configuration even though the file is valid immediately before and after that
window. Startup then fails intermittently while peer processes succeed.

Make configuration parsing tolerate this short-lived race with a bounded retry
using the repository's existing retry conventions. Persistent missing,
malformed, or unsupported configurations must still fail after retries rather
than being silently accepted.

Work in `/workspace/vllm`. Leave the source change in the working tree. Do not
modify task metadata or verifier files.
