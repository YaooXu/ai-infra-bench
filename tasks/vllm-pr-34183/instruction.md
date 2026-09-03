Work in `/workspace/repo`.

Fix unbounded CPU-memory growth during sustained multimodal serving with prefix
caching enabled. The production report uses Qwen2.5-VL requests and sees host
memory continue to rise over repeated prompt batches even after those requests
have completed; disabling the affected workload avoids the growth.

Completed request state and its large multimodal payload must become reclaimable
through the normal engine lifecycle. Prefix-cache results must remain correct
for initial requests and later token updates, including continued or streaming
requests. Live multimodal inputs must not be discarded.

Do not force garbage collection or disable prefix caching or multimodal input.
No reduced reproduction, suspected cause, or target file is supplied. Reproduce
the lifecycle symptom, diagnose it, and add regression coverage.
