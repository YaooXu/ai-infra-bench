import inspect
import logging
from types import SimpleNamespace
from unittest import mock

import torch
import vllm.config.vllm as vllm_config_mod
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel,
)


def _unset_current_config(monkeypatch):
    monkeypatch.setattr(vllm_config_mod, "_current_vllm_config", None)
    monkeypatch.setattr(
        vllm_config_mod,
        "VllmConfig",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                enable_expert_parallel=True,
            )
        ),
    )


def _make_kernel():
    prepare_finalize = mock.MagicMock()
    prepare_finalize.activation_format = "float16"

    fused_experts = mock.MagicMock()
    fused_experts.activation_formats = ["float16"]
    fused_experts.supports_chunking.return_value = True
    fused_experts.workspace_dtype.return_value = torch.float32
    fused_experts.workspace_shapes.side_effect = (
        lambda m, n, k, top_k, global_experts, local_experts, metadata: (
            (m, n),
            (m, k),
            (m, n),
        )
    )

    kwargs = {}
    if "parallel_config" in inspect.signature(
        FusedMoEModularKernel.__init__
    ).parameters:
        kwargs["parallel_config"] = SimpleNamespace(
            data_parallel_size=2,
            enable_expert_parallel=True,
        )

    return FusedMoEModularKernel(
        prepare_finalize,
        fused_experts,
        None,
        None,
        **kwargs,
    )


def test_profile_buffer_allocation_does_not_read_unset_global_config(
    monkeypatch, caplog
):
    _unset_current_config(monkeypatch)
    kernel = _make_kernel()

    module = "vllm.model_executor.layers.fused_moe.modular_kernel"
    with caplog.at_level(logging.WARNING), mock.patch(
        f"{module}.is_forward_context_available", return_value=True
    ), mock.patch(
        f"{module}.get_forward_context",
        return_value=SimpleNamespace(attn_metadata=None),
    ), mock.patch(
        f"{module}.dbo_current_ubatch_id", return_value=0
    ):
        kernel._allocate_buffers(
            out_dtype=torch.float32,
            device=torch.device("cpu"),
            M_chunk=2,
            M_full=2,
            N=4,
            K=4,
            top_k=2,
            global_num_experts=8,
            local_num_experts=4,
            expert_tokens_meta=None,
        )

    assert "Current vLLM config is not set." not in caplog.text
