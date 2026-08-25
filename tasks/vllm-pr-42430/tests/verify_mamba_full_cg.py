#!/usr/bin/env python3
"""Focused production behavior contract for Mamba one-token FULL-CG rows."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec


class ConcreteMambaBuilder(
    BaseMambaAttentionMetadataBuilder[BaseMambaAttentionMetadata]
):
    metadata_cls = BaseMambaAttentionMetadata


def make_config() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, mamba_cache_mode="all"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=None,
        num_speculative_tokens=0,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        model_config=SimpleNamespace(max_model_len=64),
    )


def make_common_metadata(
    *, seq_len: int, is_prefilling: bool, device: torch.device
) -> CommonAttentionMetadata:
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int32)
    metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.tensor([seq_len - 1], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=seq_len,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32, device=device),
        slot_mapping=torch.zeros((1,), dtype=torch.int64, device=device),
        causal=True,
    )
    return metadata.replace(
        is_prefilling=torch.tensor([is_prefilling], dtype=torch.bool)
    )


def make_builder(device: torch.device) -> ConcreteMambaBuilder:
    spec = MambaSpec(
        block_size=16,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
    )
    return ConcreteMambaBuilder(spec, ["layer0"], make_config(), device)


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA is unavailable", file=sys.stderr)
        return 2

    device = torch.device("cuda")

    # Environment control: an ordinary decode row must traverse the production
    # FULL-CG metadata path and keep CUDA-backed persistent decode state.
    control = make_builder(device).build_for_cudagraph_capture(
        make_common_metadata(seq_len=10, is_prefilling=False, device=device)
    )
    if not (
        control.num_decodes == 1
        and control.num_prefills == 0
        and control.state_indices_tensor_d is not None
        and control.state_indices_tensor_d.is_cuda
    ):
        print(f"FAIL: control decode metadata is invalid: {control}", file=sys.stderr)
        return 2

    # Bug contract: D-side recomputation of N from existing h(N-1) is still
    # scheduler-labelled prefill, but a uniform single-token FULL-CG batch must
    # build decode/update metadata.
    prior_state = make_builder(device).build_for_cudagraph_capture(
        make_common_metadata(seq_len=10, is_prefilling=True, device=device)
    )

    # Guardrail: the very first prompt token has no prior Mamba state and must
    # remain a prefill.
    first_token = make_builder(device).build(
        0, make_common_metadata(seq_len=1, is_prefilling=True, device=device)
    )
    torch.cuda.synchronize()

    print(
        "observed "
        f"prior_state=(decodes={prior_state.num_decodes},prefills={prior_state.num_prefills}) "
        f"first_token=(decodes={first_token.num_decodes},prefills={first_token.num_prefills})"
    )
    if prior_state.num_decodes != 1 or prior_state.num_prefills != 0:
        print(
            "FAIL: a one-token Mamba row with prior state stayed prefill while "
            "the production FULL-CG metadata path is decode-shaped",
            file=sys.stderr,
        )
        return 1
    if first_token.num_decodes != 0 or first_token.num_prefills != 1:
        print("FAIL: a true first-token prompt was reclassified", file=sys.stderr)
        return 3

    props = torch.cuda.get_device_properties(0)
    print(
        "PASS: production Mamba FULL-CG metadata classifies prior-state "
        "single-token rows as decode and preserves first-token prefill"
    )
    print(
        f"gpu={props.name} capability={props.major}.{props.minor} uuid={props.uuid}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
