Work in `/workspace/repo`.

Fix the request-lifetime regression in vLLM. With prefix caching enabled, completed requests that used a block hasher can retain the `Request` object and its multimodal feature payload until cyclic garbage collection runs. Once the normal owner releases a completed request, both objects must become reclaimable immediately without requiring `gc.collect()`.

Preserve initial and incremental block hashing, including append-token and streaming-session paths. Do not solve the task by disabling prefix caching, dropping multimodal data early, forcing garbage collection, or suppressing hashing.

Keep the change narrowly scoped and leave the repository in a buildable state.

