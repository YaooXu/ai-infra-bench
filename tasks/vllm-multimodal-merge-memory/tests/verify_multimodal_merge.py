#!/usr/bin/env python3
"""Unprivileged observation worker for the trusted verifier supervisor.

The worker imports candidate-controlled vLLM and executes exactly one case.
It never decides the reward and never reports aggregate success.  The trusted
parent owns the case list, the expectations, and the reward file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

# ``-I`` prevents candidate-controlled environment variables from selecting a
# different package, so bind the one intended candidate repository explicitly.
sys.path.insert(0, "/workspace/repo")

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vllm
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import _merge_multimodal_embeddings


RESULT_PREFIX = "AI_INFRA_OBSERVATION="

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class RejectExplicitMaskTransfer(TorchDispatchMode):
    """Record explicit transfer of the original boolean mask to CUDA."""

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
            outputs = list(result) if isinstance(result, (tuple, list)) else [result]
            # Record moving the boolean mask itself.  A solution may legally
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
        self.guard = patch.object(
            torch.cuda,
            "set_sync_debug_mode",
            side_effect=AssertionError(
                "production code must not disable CUDA synchronization checks"
            ),
        )
        self.guard.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.guard.stop()
        torch.cuda.set_sync_debug_mode(self.previous)


def assert_runtime_binding() -> None:
    repo = Path("/workspace/repo").resolve()
    source = Path(vllm.__file__).resolve()
    if repo not in source.parents:
        raise AssertionError(f"candidate source is not active: {source}")
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required")


def make_case(dtype: torch.dtype, mask_device: str):
    num_tokens, hidden = 8192, 512
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    mask[1::3] = True
    selected = int(mask.sum())
    if mask_device == "cuda":
        mask = mask.cuda()
    inputs = torch.full((num_tokens, hidden), -3.0, dtype=dtype, device="cuda")
    first = torch.full(
        (selected // 2, hidden), 7.0, dtype=torch.float32, device="cuda"
    )
    second = torch.full(
        (selected - selected // 2, hidden), 11.0, dtype=torch.float32, device="cuda"
    )
    return inputs, [first, [second]], mask, selected


def observe_merge(dtype_name: str, mask_device: str) -> dict[str, Any]:
    dtype = DTYPES[dtype_name]
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
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - before
    target_bytes = inputs.numel() * inputs.element_size()

    mask_cpu = mask.cpu()
    chosen = inputs[mask_cpu]
    split = selected // 2
    return {
        "returned_input_tensor": returned is inputs,
        "peak_ratio": peak_delta / target_bytes,
        "mask_copied_to_cuda": transfer_guard.cpu_to_cuda_copies,
        "first_segment_placed": bool(
            torch.equal(chosen[:split], torch.full_like(chosen[:split], 7.0))
        ),
        "second_segment_placed": bool(
            torch.equal(chosen[split:], torch.full_like(chosen[split:], 11.0))
        ),
        "text_rows_preserved": bool(
            torch.equal(inputs[~mask_cpu], torch.full_like(inputs[~mask_cpu], -3.0))
        ),
    }


def observe_cardinality(
    mask_device: str, num_embeddings: int, num_placeholders: int
) -> dict[str, Any]:
    num_tokens = max(9, num_placeholders)
    inputs = torch.zeros((num_tokens, 16), dtype=torch.bfloat16, device="cuda")
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    mask[:num_placeholders] = True
    if mask_device == "cuda":
        mask = mask.cuda()
    embeddings = [
        torch.ones((num_embeddings, 16), dtype=torch.float32, device="cuda")
    ]
    try:
        _merge_multimodal_embeddings(inputs, embeddings, mask)
        torch.cuda.synchronize()
    except ValueError as exc:
        return {
            "raised_value_error": True,
            "outcome": "ValueError",
            "message": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - reported verbatim to the parent
        # Any other failure mode (for example an asynchronous device-side
        # assert) is reported as-is; the parent decides whether the declared
        # contract accepts it.
        return {
            "raised_value_error": False,
            "outcome": type(exc).__name__,
            "message": str(exc).splitlines()[0] if str(exc) else "",
        }
    # A silently broadcast assignment lands here: record how many placeholder
    # rows were actually overwritten so the parent can see the duplication.
    mask_cpu = mask.cpu()
    overwritten = int(inputs[mask_cpu].eq(1.0).all(dim=1).sum())
    return {
        "raised_value_error": False,
        "outcome": "accepted",
        "rows_overwritten": overwritten,
        "num_placeholders": num_placeholders,
    }


def observe_empty_identity() -> dict[str, Any]:
    inputs = torch.randn((7, 13), dtype=torch.float16, device="cuda")
    before = inputs.clone()
    returned = _merge_multimodal_embeddings(
        inputs, [], torch.zeros(7, dtype=torch.bool)
    )
    torch.cuda.synchronize()
    return {"identity": returned is inputs and bool(torch.equal(inputs, before))}


def observe_model_interface_path() -> dict[str, Any]:
    """Exercise the production multimodal model interface without model weights."""

    class LanguageModel:
        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.full(
                (len(input_ids), 8), -5.0, dtype=torch.float16, device="cuda"
            )

    class ModelHarness:
        _has_oov_mm_tokens = False
        _embed_text_input_ids = SupportsMultiModal._embed_text_input_ids

        def get_language_model(self) -> LanguageModel:
            return LanguageModel()

    input_ids = torch.arange(12, dtype=torch.int64, device="cuda")
    mask = torch.zeros(12, dtype=torch.bool)
    mask[[1, 4, 9]] = True
    replacements = torch.arange(24, dtype=torch.float32, device="cuda").reshape(3, 8)
    with RejectCudaSync():
        merged = SupportsMultiModal.embed_input_ids(
            ModelHarness(), input_ids, replacements, is_multimodal=mask
        )
    torch.cuda.synchronize()
    return {
        "replacements_placed": bool(
            torch.equal(merged[mask], replacements.to(dtype=merged.dtype))
        ),
        "text_rows_preserved": bool(
            torch.equal(merged[~mask], torch.full_like(merged[~mask], -5.0))
        ),
    }


def dispatch(request: dict[str, Any]) -> Any:
    kind = request["kind"]
    if kind == "merge":
        return observe_merge(request["dtype"], request["mask_device"])
    if kind == "cardinality":
        return observe_cardinality(
            request["mask_device"],
            request["num_embeddings"],
            request["num_placeholders"],
        )
    if kind == "empty_identity":
        return observe_empty_identity()
    if kind == "model_interface_path":
        return observe_model_interface_path()
    raise AssertionError(f"unknown case kind {kind!r}")


def main() -> int:
    request = json.loads(sys.stdin.readline())
    envelope: dict[str, Any] = {
        "case": request.get("case"),
        "nonce": request.get("nonce"),
        "error": None,
        "value": None,
    }
    try:
        assert_runtime_binding()
        envelope["value"] = dispatch(request)
    except BaseException as exc:  # noqa: BLE001 - reported to the trusted parent
        envelope["error"] = f"{type(exc).__name__}: {exc}"
    print(RESULT_PREFIX + json.dumps(envelope))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
