Work in `/workspace/repo`.

We run a long-lived Qwen2.5-VL service with prefix caching enabled. While
replaying batches of multimodal prompts, I noticed that host memory keeps
growing even after the requests finish. The process eventually has to be
restarted; the growth disappears when we stop this workload.

Please find and fix the lifetime problem behind this behavior.

Completed request state and its large multimodal payload must become reclaimable
through the normal engine lifecycle. Prefix-cache results must remain correct
for initial requests and later token updates, including continued or streaming
requests. Live multimodal inputs must not be discarded.

Do not force garbage collection or disable prefix caching or multimodal input.
No reduced reproduction, suspected cause, or target file is supplied. Reproduce
the lifecycle symptom, diagnose it, and add regression coverage.
