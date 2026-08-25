Work in `/workspace/repo`.

Add decode-context-parallel support to GPU model runner v2's attention metadata and block tables. Compute rank-local sequence lengths and map virtual KV-cache blocks to the correct local slots for the configured DCP rank and interleave size.

The production Triton slot-mapping path must return padding for positions owned by other ranks, map owned positions to the correct local block/offset, and replay correctly from a CUDA graph over persistent input buffers. Keep helper allocations outside the graph-capture boundary and wire the new metadata through input-batch, model-runner, and CUDA-graph paths.

