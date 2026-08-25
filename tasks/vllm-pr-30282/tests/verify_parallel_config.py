#!/usr/bin/env python3
"""Real CUDA plus production-consumer contract for modular MoE config ownership."""

from __future__ import annotations

import json
import inspect
import logging
import sys
from types import SimpleNamespace

sys.path.insert(0, "/workspace/repo")

import torch

import vllm.model_executor.layers.fused_moe.fused_moe_modular_method as method_module
import vllm.model_executor.layers.quantization.utils.flashinfer_utils as flashinfer_utils
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp8
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedPrepareAndFinalize,
    BatchedTritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel,
)
from vllm.v1.worker.workspace import init_workspace_manager


EXPERTS = 4
TOKENS = 19
HIDDEN = 80
INTERMEDIATE = 144
TOP_K = 2


def make_components():
    return (
        BatchedPrepareAndFinalize(
            max_num_tokens=TOKENS,
            num_local_experts=EXPERTS,
            num_dispatchers=1,
            rank=0,
        ),
        BatchedTritonExperts(
            max_num_tokens=TOKENS,
            num_dispatchers=1,
            quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        ),
    )


def make_kernel(**kwargs):
    return FusedMoEModularKernel(*make_components(), **kwargs)


def make_inputs():
    torch.manual_seed(82023)
    hidden_states = torch.randn(
        TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16
    )
    w1 = torch.randn(
        EXPERTS,
        2 * INTERMEDIATE,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    logits = torch.randn(TOKENS, EXPERTS, device="cuda")
    weights, ids = torch.topk(logits, TOP_K, dim=-1)
    return hidden_states, w1, w2, torch.softmax(weights, dim=-1), ids


def run_cuda(kernel, inputs):
    hidden_states, w1, w2, weights, ids = inputs
    output = kernel(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=weights,
        topk_ids=ids,
        inplace=False,
        global_num_experts=EXPERTS,
    )
    torch.cuda.synchronize()
    assert output.is_cuda and output.shape == hidden_states.shape
    assert bool(torch.isfinite(output).all()) and output.abs().sum().item() > 0
    return output


def parallel_config(dp_size: int, use_ep: bool) -> FusedMoEParallelConfig:
    return FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=dp_size,
        ep_size=dp_size if use_ep else 1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        use_ep=use_ep,
        all2all_backend="allgather_reducescatter",
    )


def check_production_factory(config: FusedMoEParallelConfig) -> None:
    """Exercise the real FusedMoEModularMethod.make propagation logic."""

    original_make = method_module.FusedMoEModularMethod.make
    original_kernel = method_module.FusedMoEModularKernel
    original_method = method_module.FusedMoEModularMethod
    captured = {}

    def fake_kernel(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "kernel-sentinel"

    def fake_method(old_quant_method, kernel):
        captured["method_kernel"] = kernel
        return "method-sentinel"

    class OldQuantMethod:
        @staticmethod
        def select_gemm_impl(prepare_finalize, moe_layer):
            return "experts-sentinel"

    poison_global_config = object()
    layer = SimpleNamespace(
        moe_parallel_config=config,
        vllm_config=SimpleNamespace(parallel_config=poison_global_config),
        shared_experts_stream=None,
    )
    method_module.FusedMoEModularKernel = fake_kernel
    method_module.FusedMoEModularMethod = fake_method
    try:
        result = original_make(
            layer,
            OldQuantMethod(),
            "prepare-sentinel",
            "shared-sentinel",
        )
    finally:
        method_module.FusedMoEModularKernel = original_kernel
        method_module.FusedMoEModularMethod = original_method

    assert result == "method-sentinel"
    assert captured["method_kernel"] == "kernel-sentinel"
    assert captured["kwargs"].get("moe_parallel_config") is config
    assert "parallel_config" not in captured["kwargs"]


def check_flashinfer_consumer(config: FusedMoEParallelConfig) -> None:
    """Exercise the real FlashInfer wrapper without requiring that backend."""

    original_kernel = flashinfer_utils.mk.FusedMoEModularKernel
    original_prepare = (
        flashinfer_utils.build_flashinfer_fp8_cutlass_moe_prepare_finalize
    )
    original_select = flashinfer_utils.select_cutlass_fp8_gemm_impl
    captured = {}
    sentinel = torch.ones(1, 1, device="cuda")

    class FakeKernel:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def __call__(self, *args, **kwargs):
            captured["call_args"] = args
            captured["call_kwargs"] = kwargs
            return sentinel

    class QuantMethod:
        @staticmethod
        def get_fused_moe_quant_config(layer):
            return "quant-config-sentinel"

    poison_global_config = object()
    layer = SimpleNamespace(
        moe_parallel_config=config,
        vllm_config=SimpleNamespace(parallel_config=poison_global_config),
        quant_method=QuantMethod(),
        w13_weight="w13-sentinel",
        w2_weight="w2-sentinel",
    )
    flashinfer_utils.mk.FusedMoEModularKernel = FakeKernel
    flashinfer_utils.build_flashinfer_fp8_cutlass_moe_prepare_finalize = (
        lambda **kwargs: "prepare-sentinel"
    )
    flashinfer_utils.select_cutlass_fp8_gemm_impl = (
        lambda **kwargs: "experts-sentinel"
    )
    try:
        result = flashinfer_utils.flashinfer_cutlass_moe_fp8(
            torch.ones(1, 1, device="cuda"),
            layer,
            torch.ones(1, 1, device="cuda"),
            torch.zeros(1, 1, device="cuda", dtype=torch.int64),
        )
    finally:
        flashinfer_utils.mk.FusedMoEModularKernel = original_kernel
        flashinfer_utils.build_flashinfer_fp8_cutlass_moe_prepare_finalize = (
            original_prepare
        )
        flashinfer_utils.select_cutlass_fp8_gemm_impl = original_select

    assert result is sentinel
    assert captured["kwargs"].get("moe_parallel_config") is config
    assert "parallel_config" not in captured["kwargs"]
    assert "parallel_config" not in inspect.signature(cutlass_moe_fp8).parameters


def main() -> None:
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (8, 0)
    init_workspace_manager(torch.device("cuda:0"))
    inputs = make_inputs()

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture()
    logging.getLogger().addHandler(handler)
    try:
        compatibility = make_kernel()
        compatibility_output = run_cuda(compatibility, inputs)
    finally:
        logging.getLogger().removeHandler(handler)
    print("CUDA_TRITON_COMPAT_OK", float(compatibility_output.float().norm()))
    assert not compatibility.is_dp_ep
    assert not any("Current vLLM config is not set" in msg for msg in records)

    ordinary_config = parallel_config(1, False)
    ordinary = make_kernel(moe_parallel_config=ordinary_config)
    assert ordinary.moe_parallel_config is ordinary_config
    assert not ordinary.is_dp_ep
    ordinary_output = run_cuda(ordinary, inputs)
    torch.testing.assert_close(ordinary_output, compatibility_output)

    dp_ep_config = parallel_config(2, True)
    dp_ep = make_kernel(moe_parallel_config=dp_ep_config)
    assert dp_ep.moe_parallel_config is dp_ep_config and dp_ep.is_dp_ep
    check_production_factory(dp_ep_config)
    check_flashinfer_consumer(dp_ep_config)

    try:
        make_kernel(parallel_config=SimpleNamespace())
    except TypeError:
        pass
    else:
        raise AssertionError("legacy parallel_config keyword is still accepted")

    print(
        json.dumps(
            {
                "compatibility_norm": float(compatibility_output.float().norm()),
                "consumers": [
                    "FusedMoEModularMethod.make",
                    "flashinfer_cutlass_moe_fp8",
                ],
                "cuda": True,
                "dp_ep": dp_ep.is_dp_ep,
                "gpu": torch.cuda.get_device_name(0),
                "legacy_keyword_rejected": True,
                "ordinary_matches_compatibility": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
