# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import ctypes
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    SendBlockMeta,
    TransferRegion,
    _align_transfer_regions,
    _expand_transfer_regions,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    MambaSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.request import Request


class RecordingEngine:
    def __init__(self) -> None:
        self.registrations: list[tuple[list[int], list[int]]] = []
        self.transfers: list[tuple[list[int], list[int], list[int]]] = []

    def batch_register_memory(self, addresses, lengths) -> int:
        self.registrations.append((list(addresses), list(lengths)))
        return 0

    def batch_transfer_sync_write(self, _target, src, dst, lengths) -> int:
        self.transfers.append((list(src), list(dst), list(lengths)))
        for src_ptr, dst_ptr, length in zip(src, dst, lengths, strict=True):
            ctypes.memmove(dst_ptr, src_ptr, length)
        return 0


class LocalTopology:
    def __init__(self, *, split: bool = False) -> None:
        self.virtually_split_kv_in_blocks = split
        self.local_replicates_kv_cache = False
        self.is_mamba = False

    def get_transfer_cache_regions(self, cache, _spec):
        return [cache]


def full_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )


def gdn_spec() -> MambaSpec:
    return MambaSpec(
        block_size=4,
        shapes=((6, 3), (1, 2, 2)),
        dtypes=(torch.float16, torch.float16),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )


def mla_spec(*, payload: int = 96) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=payload,
        dtype=torch.uint8,
        page_size_padded=payload,
    )


def hybrid_config(num_blocks: int = 10) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.0.self_attn"], full_spec()),
            KVCacheGroupSpec(["model.layers.1.linear_attn"], gdn_spec()),
        ],
    )


def mla_config(num_blocks: int = 4, payload: int = 96) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.2.self_attn"], mla_spec(payload=payload))
        ],
    )


def sliding_mla_config(num_blocks: int = 4, payload: int = 80) -> KVCacheConfig:
    spec = SlidingWindowMLASpec(
        block_size=1,
        num_kv_heads=1,
        head_size=payload,
        dtype=torch.uint8,
        page_size_padded=payload,
        sliding_window=16,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["model.layers.3.self_attn"], spec)],
    )


def full_attention_config(num_blocks: int = 4) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.0.self_attn"], full_spec())
        ],
    )


def make_request(
    token_count: int,
    *,
    remote_prefill: bool = False,
    remote_decode: bool = False,
    prompt_embeds: bool = False,
) -> Request:
    embeddings = torch.randn(token_count, 8) if prompt_embeds else None
    request = Request(
        request_id=f"request-{token_count}-{remote_prefill}-{remote_decode}",
        prompt_token_ids=None if prompt_embeds else list(range(token_count)),
        prompt_embeds=embeddings,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        block_hasher=None,
    )
    request.kv_transfer_params = {
        "do_remote_prefill": remote_prefill,
        "do_remote_decode": remote_decode,
    }
    return request


def make_scheduler(*, has_mamba: bool, producer: bool):
    scheduler = object.__new__(MooncakeConnectorScheduler)
    scheduler._has_mamba = has_mamba
    scheduler.is_kv_producer = producer
    scheduler.is_kv_consumer = not producer
    return scheduler


def layer_specs(config: KVCacheConfig):
    return {
        layer: group.kv_cache_spec
        for group in config.kv_cache_groups
        for layer in group.layer_names
    }


def make_worker(
    config: KVCacheConfig,
    *,
    split: bool = False,
    engine: RecordingEngine | None = None,
) -> MooncakeConnectorWorker:
    worker = object.__new__(MooncakeConnectorWorker)
    worker.use_mla = any(
        isinstance(group.kv_cache_spec, MLAAttentionSpec)
        for group in config.kv_cache_groups
    )
    worker.kv_cache_config = config
    worker._layer_specs = layer_specs(config)
    worker._layer_group_indices = {
        layer: index
        for index, group in enumerate(config.kv_cache_groups)
        for layer in group.layer_names
    }
    worker.transfer_topo = LocalTopology(split=split)
    worker.engine = engine or RecordingEngine()
    worker.is_kv_consumer = True
    worker.is_kv_producer = False
    worker.tp_rank = 0
    worker.tp_size = 1
    worker._physical_blocks_per_logical_kv_block = 1
    worker.shutdown = lambda: None
    return worker


def hybrid_caches(num_blocks: int = 10):
    fa = torch.zeros((num_blocks, full_spec().page_size_bytes), dtype=torch.uint8)
    conv = torch.zeros((num_blocks, 22), dtype=torch.float16)
    ssm = torch.zeros((num_blocks, 4), dtype=torch.float16)
    return {
        "model.layers.0.self_attn": fa,
        "model.layers.1.linear_attn": (conv, ssm),
    }


def test_regions_keep_address_stride_separate_from_payload() -> None:
    regions = _expand_transfer_regions(
        base_addrs=[0x1000],
        block_lens=[1024],
        kv_block_lens=[96],
        layer_names=["model.layers.2.self_attn"],
        layer_indices=[2],
        is_kv_layout_blocks_first=False,
        group_indices=[3],
    )
    assert regions == [
        TransferRegion("model.layers.2.self_attn", 2, 0x1000, 1024, 96, 3)
    ]


def test_only_full_attention_region_is_virtually_split() -> None:
    worker = make_worker(hybrid_config(), split=True)
    regions = worker._get_transfer_regions(
        base_addrs=[0x1000, 0x2000],
        block_lens=[128, 44],
        kv_block_lens=[64, 44],
        layer_names=["model.layers.0.self_attn", "model.layers.1.linear_attn"],
        layer_indices=[0, 1],
        group_indices=[0, 1],
    )
    assert [(r.group_index, r.base_addr, r.kv_block_len) for r in regions] == [
        (0, 0x1000, 64),
        (0, 0x1040, 64),
        (1, 0x2000, 44),
    ]


def test_region_metadata_lengths_must_align() -> None:
    with pytest.raises(AssertionError):
        _expand_transfer_regions(
            [1, 2], [32], [16], ["layer"], [0], False, [0]
        )


def test_region_group_mismatch_is_rejected() -> None:
    local = [TransferRegion("layer.0", 0, 100, 32, 32, 0)]
    remote = [TransferRegion("layer.0", 0, 200, 32, 32, 1)]
    aligned_local, aligned_remote, error = _align_transfer_regions(local, remote)
    assert aligned_local == [] and aligned_remote == []
    assert error is not None and "group index mismatch" in error


def test_transfer_metadata_round_trip_preserves_layout_fields() -> None:
    metadata = MooncakeXferMetadata(
        remote_hostname="decoder",
        remote_port=9000,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"request": ("transfer", [[1, 2], [3]])},
        kv_caches_base_addr=[1000, 2000],
        block_lens=[1024, 64],
        kv_block_lens=[96, 64],
        registered_layer_names=["layer.0", "layer.1"],
        registered_layer_indices=[0, 1],
        registered_group_indices=[0, 1],
    )
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(metadata), type=MooncakeXferMetadata
    )
    assert restored.kv_block_lens == [96, 64]
    assert restored.registered_group_indices == [0, 1]
    assert restored.req_blocks == metadata.req_blocks


def test_hybrid_remote_prefill_recomputes_final_prompt_token() -> None:
    scheduler = make_scheduler(has_mamba=True, producer=False)
    request = make_request(11, remote_prefill=True)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (10, True)


def test_pure_attention_remote_prefill_keeps_full_prompt() -> None:
    scheduler = make_scheduler(has_mamba=False, producer=False)
    request = make_request(11, remote_prefill=True)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (11, True)


def test_hybrid_remote_decode_truncates_once() -> None:
    scheduler = make_scheduler(has_mamba=True, producer=True)
    request = make_request(9, remote_decode=True)
    original = list(request.prompt_token_ids)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert request.prompt_token_ids == original[:-1]
    assert list(request.all_token_ids) == original[:-1]
    assert request.num_prompt_tokens == 8
    assert request.max_tokens == 1
    assert request.kv_transfer_params["_p_side_truncated"] is True
    scheduler.get_num_new_matched_tokens(request, 0)
    assert request.prompt_token_ids == original[:-1]


def test_hybrid_prompt_embeddings_truncate_once() -> None:
    scheduler = make_scheduler(has_mamba=True, producer=True)
    request = make_request(7, remote_decode=True, prompt_embeds=True)
    scheduler.get_num_new_matched_tokens(request, 0)
    assert request.prompt_embeds.shape[0] == 6
    assert request.num_prompt_tokens == 6
    scheduler.get_num_new_matched_tokens(request, 0)
    assert request.prompt_embeds.shape[0] == 6


def test_pure_attention_remote_decode_does_not_truncate() -> None:
    scheduler = make_scheduler(has_mamba=False, producer=True)
    request = make_request(7, remote_decode=True)
    original = list(request.prompt_token_ids)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert request.prompt_token_ids == original
    assert request.num_prompt_tokens == 7


def test_nixl_remote_prefill_behavior_remains_compatible() -> None:
    scheduler = object.__new__(NixlPullConnectorScheduler)
    scheduler._has_mamba = True
    request = make_request(10, remote_prefill=True)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (9, True)


@pytest.mark.parametrize("factor", [1, 3, 17])
def test_logical_block_expansion_applies_only_to_attention(factor) -> None:
    worker = make_worker(hybrid_config())
    worker._physical_blocks_per_logical_kv_block = factor
    mapped = worker._logical_to_kernel_block_ids([[2], [2]])
    expected_fa = list(range(2 * factor, 3 * factor))
    assert mapped == [expected_fa, [2]]


def test_hybrid_registration_emits_fa_and_gdn_regions() -> None:
    worker = make_worker(hybrid_config())
    caches = hybrid_caches()
    worker.register_kv_caches(caches)
    assert worker.registered_layer_names == [
        "model.layers.0.self_attn",
        "model.layers.1.linear_attn",
    ]
    assert worker.registered_group_indices == [0, 1]
    assert worker.kv_caches_base_addr == [
        caches["model.layers.0.self_attn"].data_ptr(),
        caches["model.layers.1.linear_attn"][0].data_ptr(),
    ]


def test_shared_backing_is_registered_once() -> None:
    config = hybrid_config(num_blocks=4)
    engine = RecordingEngine()
    worker = make_worker(config, engine=engine)
    backing = torch.zeros((4, 128), dtype=torch.uint8)
    fa = backing[:, :64]
    conv = backing[:, 64:96]
    ssm = torch.zeros((4, 4), dtype=torch.float16)
    worker.register_kv_caches(
        {
            "model.layers.0.self_attn": fa,
            "model.layers.1.linear_attn": (conv, ssm),
        }
    )
    assert engine.registrations == [
        ([backing.untyped_storage().data_ptr()], [backing.untyped_storage().nbytes()])
    ]
    assert worker.kv_caches_base_addr == [fa.data_ptr(), conv.data_ptr()]


def test_mla_registration_uses_stride_for_address_and_payload_for_copy() -> None:
    payload = 96
    stride = 1024
    num_blocks = 4
    engine = RecordingEngine()
    worker = make_worker(mla_config(num_blocks, payload), engine=engine)
    backing = torch.zeros(num_blocks * stride, dtype=torch.uint8)
    cache = torch.as_strided(
        backing,
        size=(num_blocks, payload),
        stride=(stride, 1),
    )
    worker.register_kv_caches({"model.layers.2.self_attn": cache})
    assert worker.block_len_per_layer == [stride]
    assert worker.kv_block_len_per_layer == [payload]
    assert engine.registrations == [([backing.data_ptr()], [backing.nbytes])]
    region = worker._get_transfer_regions(
        worker.kv_caches_base_addr,
        worker.block_len_per_layer,
        worker.kv_block_len_per_layer,
        worker.registered_layer_names,
        worker.registered_layer_indices,
        worker.registered_group_indices,
    )[0]
    assert region.base_addr + region.block_len == cache.data_ptr() + stride
    assert region.kv_block_len == payload


def test_sliding_window_mla_uses_payload_not_padded_stride() -> None:
    payload = 80
    stride = 640
    config = sliding_mla_config(payload=payload)
    worker = make_worker(config)
    backing = torch.zeros(config.num_blocks * stride, dtype=torch.uint8)
    cache = torch.as_strided(
        backing, size=(config.num_blocks, payload), stride=(stride, 1)
    )
    worker.register_kv_caches({"model.layers.3.self_attn": cache})
    assert worker.block_len_per_layer == [stride]
    assert worker.kv_block_len_per_layer == [payload]


def test_blocks_first_full_attention_uses_two_half_payload_regions() -> None:
    worker = make_worker(full_attention_config(), split=True)
    cache = torch.zeros((4, full_spec().page_size_bytes), dtype=torch.uint8)
    worker.register_kv_caches({"model.layers.0.self_attn": cache})
    assert worker.block_len_per_layer == [cache.stride(0)]
    assert worker.kv_block_len_per_layer == [cache.stride(0) // 2]
    regions = worker._get_transfer_regions(
        worker.kv_caches_base_addr,
        worker.block_len_per_layer,
        worker.kv_block_len_per_layer,
        worker.registered_layer_names,
        worker.registered_layer_indices,
        worker.registered_group_indices,
    )
    assert len(regions) == 2
    assert [region.kv_block_len for region in regions] == [
        cache.stride(0) // 2,
        cache.stride(0) // 2,
    ]


def test_memory_registration_failure_propagates() -> None:
    class FailingEngine(RecordingEngine):
        def batch_register_memory(self, addresses, lengths) -> int:
            super().batch_register_memory(addresses, lengths)
            return 7

    worker = make_worker(full_attention_config(), engine=FailingEngine())
    cache = torch.zeros((4, full_spec().page_size_bytes), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="registration failed"):
        worker.register_kv_caches({"model.layers.0.self_attn": cache})


def test_unknown_layers_are_not_registered() -> None:
    worker = make_worker(hybrid_config())
    with pytest.raises(RuntimeError, match="No KV cache tensors"):
        worker.register_kv_caches({"model.layers.9.unknown": torch.zeros(4, 16)})


def test_group_identity_and_null_blocks_drive_transfer_addresses() -> None:
    worker = make_worker(hybrid_config())
    send = SendBlockMeta(
        p_req_id="producer",
        transfer_id="transfer",
        local_block_ids=[[10, 11], [NULL_BLOCK_ID, 4]],
        ready=asyncio.Event(),
    )
    metadata = MooncakeXferMetadata(
        remote_hostname="decoder",
        remote_port=9000,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"decoder-request": ("transfer", [[30], [NULL_BLOCK_ID, 7]])},
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
    )
    local = [
        TransferRegion("model.layers.1.linear_attn", 1, 0x5000, 64, 64, 1),
        TransferRegion("model.layers.0.self_attn", 0, 0x1000, 128, 128, 0),
    ]
    remote = [
        TransferRegion("model.layers.1.linear_attn", 1, 0x6000, 64, 64, 1),
        TransferRegion("model.layers.0.self_attn", 0, 0x2000, 128, 128, 0),
    ]
    src, dst, lengths, errors, message = asyncio.run(
        worker._build_transfer_params(
            [("decoder-request", send)], metadata, local, remote
        )
    )
    assert errors == [] and message is None
    assert src == [0x5000 + 4 * 64, 0x1000 + 11 * 128]
    assert dst == [0x6000 + 7 * 64, 0x2000 + 30 * 128]
    assert lengths == [64, 128]


def test_group_count_mismatch_returns_request_error() -> None:
    worker = make_worker(hybrid_config())
    send = SendBlockMeta("producer", "transfer", [[1]], asyncio.Event())
    metadata = MooncakeXferMetadata(
        "decoder", 9000, 1, 0,
        {"request": ("transfer", [[2], [3]])}, [], [], []
    )
    result = asyncio.run(worker._build_transfer_params([("request", send)], metadata, [], []))
    assert result[3] == ["request"]
    assert result[4] == "KV group count mismatch"


def test_empty_mamba_group_does_not_generate_transfer() -> None:
    worker = make_worker(hybrid_config())
    send = SendBlockMeta(
        "producer", "transfer", [[1], [NULL_BLOCK_ID]], asyncio.Event()
    )
    metadata = MooncakeXferMetadata(
        "decoder", 9000, 1, 0,
        {"request": ("transfer", [[2], [NULL_BLOCK_ID]])}, [], [], []
    )
    local = [TransferRegion("model.layers.1.linear_attn", 1, 1000, 64, 64, 1)]
    remote = [TransferRegion("model.layers.1.linear_attn", 1, 2000, 64, 64, 1)]
    src, dst, lengths, errors, _ = asyncio.run(
        worker._build_transfer_params([("request", send)], metadata, local, remote)
    )
    assert src == dst == lengths == []
    assert errors == []
