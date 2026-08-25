Work in `/app`.

Correct vLLM's multimodal encoder-cache accounting. A multimodal placeholder can span many input-token positions while only some positions correspond to encoder embedding rows. Cache capacity, allocation, eviction, scheduling budgets, and partial embedding selection must consistently operate in embedding-row units rather than placeholder-token units.

For example, a placeholder of length 100 with an embedding mask containing eight true positions must fit in an encoder cache with capacity eight. Preserve the behavior of placeholders without a mask and update all affected callers instead of adding a one-off special case.

Keep the implementation internally consistent across request, scheduler, profiling, cache-manager, and multimodal helper paths.

