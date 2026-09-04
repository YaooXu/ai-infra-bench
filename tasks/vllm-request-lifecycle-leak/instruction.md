Work in `/workspace/repo`.

I run a long-lived Qwen2.5-VL service with prefix caching enabled. When I replay
batches of multimodal prompts, the process keeps using more host memory even
after those requests have finished. Eventually I have to restart the service;
the growth stops when I stop this workload.

Please find and fix whatever is keeping the completed requests alive. A
finished request and its large multimodal payload should be reclaimable through
the normal engine lifecycle, while live multimodal input must remain available.
Prefix-cache results must also stay correct for the initial request and for
later token updates, including continued or streaming requests.

Please do not work around the problem by forcing garbage collection or by
disabling prefix caching or multimodal input. I do not have a reduced test case
or a suspected source file, so please reproduce the lifecycle problem yourself
and add regression coverage for the behavior you fix.
