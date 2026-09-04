#!/usr/bin/env python3
"""Hidden behavioral verifier for the production multimodal merge primitive."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vllm
from vllm.model_executor.models.utils import _merge_multimodal_embeddings


class RejectExplicitMaskTransfer(TorchDispatchMode):
    """Reject explicit transfer of the original boolean mask to CUDA."""

    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask
        self.cpu_to_cuda_copies: list[str] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        if "_to_copy" in str(func):
            inputs = [x for x in args if isinstance(x, torch.Tensor)]
            outputs = (
                list(result)
                if isinstance(result, (tuple, list))
                else [result]
            )
            # Reject moving the boolean mask itself. A solution may legally
            # derive a smaller CPU index tensor and copy that asynchronously;
            # correctness, sync-debug, and allocator checks judge that design.
            if any(x is self.mask for x in inputs) and any(
                isinstance(x, torch.Tensor) and x.device.type == "cuda"
                for x in outputs
            ):
                self.cpu_to_cuda_copies.append(str(func))
        return result


class RejectCudaSync:
    def __enter__(self) -> None:
        self.previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        torch.cuda.set_sync_debug_mode(self.previous)


def assert_runtime_binding() -> None:
    repo = Path("/workspace/repo").resolve()
    source = Path(vllm.__file__).resolve()
    if repo not in source.parents:
        raise AssertionError(f"candidate source is not active: {source}")
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required")
    merge_source = inspect.getsource(_merge_multimodal_embeddings)
    if "set_sync_debug_mode" in merge_source:
        raise AssertionError("production code must not disable synchronization checks")
    print(f"candidate_source={source}")
    print(f"gpu={torch.cuda.get_device_name(0)}")


def make_case(dtype: torch.dtype, mask_device: str):
    num_tokens, hidden = 8192, 512
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    mask[1::3] = True
    selected = int(mask.sum())
    if mask_device == "cuda":
        mask = mask.cuda()
    inputs = torch.full(
        (num_tokens, hidden), -3.0, dtype=dtype, device="cuda"
    )
    first = torch.full(
        (selected // 2, hidden), 7.0, dtype=torch.float32, device="cuda"
    )
    second = torch.full(
        (selected - selected // 2, hidden), 11.0, dtype=torch.float32, device="cuda"
    )
    return inputs, [first, [second]], mask, selected


def check_merge(dtype: torch.dtype, mask_device: str) -> float:
    inputs, multimodal, mask, selected = make_case(dtype, mask_device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    transfer_guard = RejectExplicitMaskTransfer(mask)
    if mask_device == "cpu":
        with RejectCudaSync(), transfer_guard:
            returned = _merge_multimodal_embeddings(inputs, multimodal, mask)
    else:
        # The upstream no-sync contract is specifically the CPU-mask path.
        # GPU boolean indexing may query its dynamic cardinality; it remains a
        # compatibility/correctness control, not the asynchronous hard gate.
        with transfer_guard:
            returned = _merge_multimodal_embeddings(inputs, multimodal, mask)
    if returned is not inputs:
        raise AssertionError("merge must return the mutated input tensor")
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - before
    target_bytes = inputs.numel() * inputs.element_size()
    ratio = peak_delta / target_bytes
    if transfer_guard.cpu_to_cuda_copies:
        raise AssertionError(
            f"mask was explicitly materialized on CUDA: "
            f"{transfer_guard.cpu_to_cuda_copies}"
        )
    mask_cpu = mask.cpu()
    chosen = inputs[mask_cpu]
    split = selected // 2
    if not torch.equal(chosen[:split], torch.full_like(chosen[:split], 7.0)):
        raise AssertionError("first nested embedding segment is misplaced")
    if not torch.equal(chosen[split:], torch.full_like(chosen[split:], 11.0)):
        raise AssertionError("second nested embedding segment is misplaced")
    if not torch.equal(inputs[~mask_cpu], torch.full_like(inputs[~mask_cpu], -3.0)):
        raise AssertionError("non-placeholder embeddings were modified")
    # Keep a broad regression guard without prescribing PyTorch's internal
    # indexing implementation or the Oracle's exact temporary-allocation curve.
    if mask_device == "cpu" and ratio >= 4.0:
        raise AssertionError(f"temporary CUDA allocation is excessive: {ratio:.3f}")
    return ratio


def check_empty_identity() -> None:
    inputs = torch.randn((7, 13), dtype=torch.float16, device="cuda")
    before = inputs.clone()
    returned = _merge_multimodal_embeddings(
        inputs, [], torch.zeros(7, dtype=torch.bool)
    )
    if returned is not inputs or not torch.equal(inputs, before):
        raise AssertionError("empty multimodal input must be an identity operation")


def expect_cardinality_error(num_embeddings: int, num_placeholders: int) -> None:
    inputs = torch.zeros((9, 16), dtype=torch.bfloat16, device="cuda")
    mask = torch.zeros(9, dtype=torch.bool)
    mask[:num_placeholders] = True
    embeddings = [
        torch.ones((num_embeddings, 16), dtype=torch.float32, device="cuda")
    ]
    try:
        _merge_multimodal_embeddings(inputs, embeddings, mask)
    except ValueError as exc:
        message = str(exc)
        lowered = message.lower()
        if (
            str(num_embeddings) not in message
            or str(num_placeholders) not in message
            or not any(word in lowered for word in ("multimodal", "embedding", "token"))
            or not any(word in lowered for word in ("placeholder", "mask", "expected"))
        ):
            raise AssertionError(f"uninformative cardinality error: {message}") from exc
        return
    raise AssertionError(
        f"cardinality mismatch {num_embeddings}!={num_placeholders} was accepted"
    )


def main() -> int:
    assert_runtime_binding()
    results = {}
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        results[f"cpu_{dtype}"] = check_merge(dtype, "cpu")
    results["cuda_bfloat16"] = check_merge(torch.bfloat16, "cuda")
    check_empty_identity()
    expect_cardinality_error(5, 3)
    expect_cardinality_error(2, 4)
    for name, ratio in results.items():
        print(f"peak_ratio[{name}]={ratio:.3f}")
    print("PASS: production merge is ordered, async, bounded, strict, and CPU-mask native")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
