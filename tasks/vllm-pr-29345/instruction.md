Work in `/workspace/repo`.

Optimize `bmm_batch_invariant` so a 3-D batched matrix multiplication is executed by a batch-aware deterministic Triton kernel instead of a Python loop that launches one matrix multiplication per batch item.

The result must remain bitwise invariant between a batched call and concatenated single-batch calls for FP16, BF16, and FP32 inputs. Preserve numerical agreement with `torch.bmm`, `out=` identity and contents, dtype/device validation, and shape errors. The optimized path must materially outperform the legacy per-batch launch pattern on representative A100 shapes.

