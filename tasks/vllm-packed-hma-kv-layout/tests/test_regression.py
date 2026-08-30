# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_config_deepseek_v4,
    get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler


def mock_vllm_config():
    config = MagicMock()
    config.cache_config.num_gpu_blocks_override = None
    return config


def full_spec(*, heads: int = 2, head_size: int = 64) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=heads,
        head_size=head_size,
        dtype=torch.float16,
    )


def sliding_spec(*, heads: int = 2, head_size: int = 64) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=16,
        num_kv_heads=heads,
        head_size=head_size,
        dtype=torch.float16,
        sliding_window=128,
    )


def attention_groups(
    *,
    layers_per_group: tuple[int, ...] = (2, 2, 2),
    varied_pages: bool = False,
) -> list[KVCacheGroupSpec]:
    groups = []
    for group_index, layer_count in enumerate(layers_per_group):
        heads = group_index + 1 if varied_pages else 2
        spec = (
            full_spec(heads=heads)
            if group_index == 0
            else sliding_spec(heads=heads)
        )
        groups.append(
            KVCacheGroupSpec(
                [f"group{group_index}.layer{index}" for index in range(layer_count)],
                spec,
            )
        )
    return groups


def make_config(
    groups: list[KVCacheGroupSpec],
    *,
    packed: bool,
    available_memory: int = 64 * 1024 * 1024,
) -> KVCacheConfig:
    had_flag = "VLLM_USE_PACKED_HMA_KV_CACHE" in vars(envs)
    original = vars(envs).get("VLLM_USE_PACKED_HMA_KV_CACHE")
    try:
        envs.VLLM_USE_PACKED_HMA_KV_CACHE = packed
        return get_kv_cache_config_from_groups(
            mock_vllm_config(), groups, available_memory
        )
    finally:
        if had_flag:
            envs.VLLM_USE_PACKED_HMA_KV_CACHE = original
        else:
            del envs.VLLM_USE_PACKED_HMA_KV_CACHE


def page_sizes(groups: list[KVCacheGroupSpec]) -> dict[str, int]:
    return {
        layer: group.kv_cache_spec.page_size_bytes
        for group in groups
        for layer in group.layer_names
    }


def allocate_views(
    config: KVCacheConfig,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    sizes = page_sizes(config.kv_cache_groups)
    packed = bool(config.kv_cache_tensors[0].block_stride)
    backings = (
        [torch.zeros(config.kv_cache_tensors[0].size, dtype=torch.uint8)]
        if packed
        else [
            torch.zeros(descriptor.size, dtype=torch.uint8)
            for descriptor in config.kv_cache_tensors
        ]
    )
    views: dict[str, torch.Tensor] = {}
    for index, descriptor in enumerate(config.kv_cache_tensors):
        backing = backings[0] if packed else backings[index]
        stride = descriptor.block_stride or sizes[descriptor.shared_by[0]]
        for layer in descriptor.shared_by:
            views[layer] = torch.as_strided(
                backing,
                size=(config.num_blocks, sizes[layer]),
                stride=(stride, 1),
                storage_offset=descriptor.offset,
            )
    return backings, views


def canonicalize(config: KVCacheConfig):
    _, views = allocate_views(config)
    captured = []
    worker = object.__new__(OffloadingConnectorWorker)
    worker.spec = SimpleNamespace(kv_cache_config=config)
    worker._register_handlers = captured.append
    worker.register_kv_caches(views)
    assert len(captured) == 1
    return captured[0], views


def deepseek_groups() -> list[KVCacheGroupSpec]:
    def mla(name: str, page: int) -> tuple[str, MLAAttentionSpec]:
        return name, MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            page_size_padded=page,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
            alignment=576,
        )

    first = dict([mla("mla.0", 37440), mla("index.0", 8640)])
    second = dict([mla("swa.0", 37440)])
    return [
        KVCacheGroupSpec(
            list(first), UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=first)
        ),
        KVCacheGroupSpec(
            list(second), UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=second)
        ),
    ]


def test_flag_defaults_to_disabled(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_USE_PACKED_HMA_KV_CACHE", raising=False)
    assert envs.environment_variables["VLLM_USE_PACKED_HMA_KV_CACHE"]() is False


@pytest.mark.parametrize("raw,expected", [("0", False), ("1", True)])
def test_flag_parses_integer_boolean(monkeypatch, raw, expected) -> None:
    monkeypatch.setenv("VLLM_USE_PACKED_HMA_KV_CACHE", raw)
    assert envs.environment_variables["VLLM_USE_PACKED_HMA_KV_CACHE"]() is expected


def test_default_multi_group_layout_is_unchanged() -> None:
    groups = attention_groups()
    config = make_config(groups, packed=False)
    assert len(config.kv_cache_tensors) == 2
    assert all(t.block_stride == 0 and t.offset == 0 for t in config.kv_cache_tensors)
    assert sum(t.size for t in config.kv_cache_tensors) > config.kv_cache_tensors[0].size


@pytest.mark.parametrize("layers_per_group", [(2, 2), (2, 2, 2), (3, 1, 2)])
def test_opt_in_packs_multi_group_attention(layers_per_group) -> None:
    config = make_config(
        attention_groups(layers_per_group=layers_per_group), packed=True
    )
    assert len(config.kv_cache_tensors) == max(layers_per_group)
    assert all(t.block_stride > 0 for t in config.kv_cache_tensors)
    assert len({t.size for t in config.kv_cache_tensors}) == 1
    assert len({t.block_stride for t in config.kv_cache_tensors}) == 1


def test_single_group_remains_unpacked_when_enabled() -> None:
    config = make_config(attention_groups(layers_per_group=(3,)), packed=True)
    assert len(config.kv_cache_tensors) == 3
    assert all(t.block_stride == 0 and t.offset == 0 for t in config.kv_cache_tensors)


def test_packing_does_not_change_usable_block_capacity() -> None:
    groups = attention_groups(layers_per_group=(3, 2, 3))
    memory = 48 * 1024 * 1024
    unpacked = make_config(groups, packed=False, available_memory=memory)
    packed = make_config(groups, packed=True, available_memory=memory)
    assert packed.num_blocks == unpacked.num_blocks


def test_offsets_partition_each_packed_block() -> None:
    groups = attention_groups()
    config = make_config(groups, packed=True)
    sizes = page_sizes(groups)
    intervals = sorted(
        (t.offset, t.offset + sizes[t.shared_by[0]])
        for t in config.kv_cache_tensors
    )
    assert intervals[0][0] == 0
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    assert intervals[-1][1] == config.kv_cache_tensors[0].block_stride


def test_varied_page_sizes_share_one_correct_backing() -> None:
    groups = attention_groups(layers_per_group=(3, 2, 1), varied_pages=True)
    config = make_config(groups, packed=True)
    sizes = page_sizes(groups)
    assert len({t.size for t in config.kv_cache_tensors}) == 1
    for descriptor in config.kv_cache_tensors:
        assert descriptor.offset + sizes[descriptor.shared_by[0]] <= descriptor.block_stride
        assert descriptor.size == descriptor.block_stride * config.num_blocks


def test_all_layers_are_exposed_exactly_once() -> None:
    groups = attention_groups(layers_per_group=(4, 2, 3))
    config = make_config(groups, packed=True)
    actual = [layer for tensor in config.kv_cache_tensors for layer in tensor.shared_by]
    expected = [layer for group in groups for layer in group.layer_names]
    assert sorted(actual) == sorted(expected)
    assert len(actual) == len(set(actual))


def test_packed_views_are_storage_aliases_but_data_independent() -> None:
    config = make_config(attention_groups(), packed=True)
    backings, views = allocate_views(config)
    assert len(backings) == 1
    assert {view.untyped_storage().data_ptr() for view in views.values()} == {
        backings[0].untyped_storage().data_ptr()
    }
    representatives = [views[t.shared_by[0]] for t in config.kv_cache_tensors]
    for index, view in enumerate(representatives, start=1):
        view.fill_(index * 17)
    for index, view in enumerate(representatives, start=1):
        assert torch.all(view == index * 17)


def test_writing_one_block_does_not_touch_other_blocks() -> None:
    config = make_config(attention_groups(), packed=True)
    backings, views = allocate_views(config)
    backings[0].zero_()
    target = views[config.kv_cache_tensors[1].shared_by[0]]
    target[3].fill_(91)
    assert torch.all(target[3] == 91)
    assert torch.all(target[2] == 0)
    assert torch.all(target[4] == 0)
    first_view = views[config.kv_cache_tensors[0].shared_by[0]]
    assert torch.all(first_view[3] == 0)


def test_existing_deepseek_v4_layout_stays_packed_without_flag() -> None:
    had_flag = "VLLM_USE_PACKED_HMA_KV_CACHE" in vars(envs)
    original = vars(envs).get("VLLM_USE_PACKED_HMA_KV_CACHE")
    try:
        envs.VLLM_USE_PACKED_HMA_KV_CACHE = False
        blocks, tensors = _get_kv_cache_config_deepseek_v4(
            mock_vllm_config(), deepseek_groups(), 32 * 1024 * 1024
        )
    finally:
        if had_flag:
            envs.VLLM_USE_PACKED_HMA_KV_CACHE = original
        else:
            del envs.VLLM_USE_PACKED_HMA_KV_CACHE
    assert blocks > 0
    assert tensors
    assert all(t.block_stride > 0 for t in tensors)
    assert len({t.size for t in tensors}) == 1


def test_simple_cpu_capacity_counts_backing_once() -> None:
    config = make_config(attention_groups(), packed=True)
    backing_size = config.kv_cache_tensors[0].size
    cpu = SimpleCPUOffloadScheduler._derive_cpu_config(config, backing_size * 3)
    assert cpu.num_blocks == config.num_blocks * 3


def test_simple_cpu_mirror_preserves_layout_metadata() -> None:
    config = make_config(attention_groups(), packed=True)
    cpu = SimpleCPUOffloadScheduler._derive_cpu_config(
        config, config.kv_cache_tensors[0].size * 2
    )
    assert [t.offset for t in cpu.kv_cache_tensors] == [
        t.offset for t in config.kv_cache_tensors
    ]
    assert [t.block_stride for t in cpu.kv_cache_tensors] == [
        t.block_stride for t in config.kv_cache_tensors
    ]
    assert all(t.size == t.block_stride * cpu.num_blocks for t in cpu.kv_cache_tensors)


def test_simple_cpu_unpacked_capacity_behavior_is_unchanged() -> None:
    config = make_config(attention_groups(), packed=False)
    total_size = sum(t.size for t in config.kv_cache_tensors)
    cpu = SimpleCPUOffloadScheduler._derive_cpu_config(config, total_size * 2)
    assert cpu.num_blocks == config.num_blocks * 2
    assert all(t.block_stride == 0 and t.offset == 0 for t in cpu.kv_cache_tensors)


def test_partially_strided_layout_is_rejected() -> None:
    groups = attention_groups(layers_per_group=(1, 1))
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(size=4096, shared_by=[groups[0].layer_names[0]], block_stride=1024),
            KVCacheTensor(size=4096, shared_by=[groups[1].layer_names[0]]),
        ],
        kv_cache_groups=groups,
    )
    with pytest.raises(AssertionError):
        SimpleCPUOffloadScheduler._derive_cpu_config(config, 8192)


def test_connector_registers_one_canonical_backing() -> None:
    config = make_config(attention_groups(), packed=True)
    canonical, views = canonicalize(config)
    assert len(canonical.tensors) == 1
    tensor = canonical.tensors[0]
    assert tensor.tensor.shape == (
        config.num_blocks,
        config.kv_cache_tensors[0].block_stride,
    )
    assert tensor.page_size_bytes == config.kv_cache_tensors[0].block_stride
    assert tensor.tensor.untyped_storage().data_ptr() in {
        view.untyped_storage().data_ptr() for view in views.values()
    }


def test_connector_gives_every_group_one_full_row_reference() -> None:
    config = make_config(attention_groups(layers_per_group=(3, 2, 1)), packed=True)
    canonical, _ = canonicalize(config)
    assert len(canonical.group_data_refs) == 3
    for refs in canonical.group_data_refs:
        assert len(refs) == 1
        assert refs[0].tensor_idx == 0
        assert refs[0].page_size_bytes == canonical.tensors[0].page_size_bytes


def test_unpacked_connector_registration_is_unchanged() -> None:
    config = make_config(attention_groups(), packed=False)
    canonical, _ = canonicalize(config)
    assert len(canonical.tensors) == len(config.kv_cache_tensors)
    assert all(len(refs) == len(config.kv_cache_tensors) for refs in canonical.group_data_refs)
