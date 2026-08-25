Work in `/workspace/repo`.

Complete support for asynchronous scheduling with pipeline parallelism. The last pipeline rank must broadcast sampled token IDs directly as GPU tensors, and earlier ranks must receive them into the input batch without a CPU-object round trip.

On the receiver, preserve request ordering, rebuild the request-ID-to-index map while excluding discarded requests, and append the output placeholder only to retained requests. Keep scheduler placeholder accounting consistent so async PP does not schedule an unnecessary extra step. The implementation must work through a real two-rank NCCL process group.

