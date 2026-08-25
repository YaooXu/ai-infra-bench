Work in `/app`.

Optimize the native `_moe_C.moe_permute` aligned-routing path. Its current CUDA implementation repeatedly derives aligned expert offsets inside every expansion block, causing poor scaling as the routed token batch grows. Compute the aligned prefix information once and let the expansion kernel perform constant-time destination mapping.

Preserve exact expert offsets, inverse/permuted mappings, expert ranges, payload bytes, and sentinel handling for aligned and unaligned cases. Rebuild the focused native extension with `/opt/bench/rebuild_native.sh`. On A100, the optimized implementation must retain correctness while avoiding the legacy large-batch scaling curve.

