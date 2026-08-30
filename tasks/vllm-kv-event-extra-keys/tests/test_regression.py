# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable

import msgspec
import pytest
import torch

import vllm.v1.core.kv_cache_utils as kv_utils
from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    generate_block_hash_extra_keys,
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request


BLOCK_SIZE = 4


@pytest.fixture(autouse=True)
def _stable_hash_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "kv-event-verifier")
    init_none_hash(sha256)


def make_request(
    token_count: int,
    *,
    identifiers: list[tuple[int, int, str]] | None = None,
    cache_salt: str | None = None,
    lora_name: str | None = None,
    prompt_embeds: torch.Tensor | None = None,
    block_size: int = BLOCK_SIZE,
) -> Request:
    features = None
    if identifiers:
        features = [
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=PlaceholderRange(offset=offset, length=length),
                identifier=identifier,
                modality="image",
            )
            for offset, length, identifier in identifiers
        ]
    lora = (
        LoRARequest(
            lora_name=lora_name,
            lora_int_id=17,
            lora_path="/opt/adapters/debug",
        )
        if lora_name
        else None
    )
    return Request(
        request_id=f"req-{token_count}-{cache_salt}-{lora_name}",
        prompt_token_ids=list(range(100, 100 + token_count)),
        prompt_embeds=prompt_embeds,
        mm_features=features,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        lora_request=lora,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def emit(
    request: Request,
    *,
    block_size: int = BLOCK_SIZE,
    null_indices: set[int] | None = None,
    num_cached_blocks: int = 0,
    existing_blocks: list | None = None,
) -> tuple[BlockStored, BlockPool, list]:
    count = request.num_prompt_tokens // block_size
    pool = BlockPool(
        num_gpu_blocks=max(16, count + 2),
        enable_caching=True,
        hash_block_size=block_size,
        enable_kv_cache_events=True,
    )
    if existing_blocks is None:
        null_indices = null_indices or set()
        real_blocks = iter(pool.get_new_blocks(count - len(null_indices)))
        blocks = [
            pool.null_block if index in null_indices else next(real_blocks)
            for index in range(count)
        ]
    else:
        blocks = existing_blocks
    pool.cache_full_blocks(
        request=request,
        blocks=blocks,
        num_cached_blocks=num_cached_blocks,
        num_full_blocks=count,
        block_size=block_size,
        kv_cache_group_id=0,
    )
    events = pool.take_events()
    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    return events[0], pool, blocks


def assert_event_reconstructs_request(
    event: BlockStored,
    request: Request,
    hash_fn: Callable = sha256,
) -> None:
    assert event.extra_keys is not None
    assert len(event.extra_keys) == len(event.block_hashes)
    parent = kv_utils.NONE_HASH
    for index, (external_hash, extra_keys) in enumerate(
        zip(event.block_hashes, event.extra_keys, strict=True)
    ):
        start = index * event.block_size
        end = start + event.block_size
        parent = hash_block_tokens(
            hash_fn,
            parent,
            tuple(request.all_token_ids[start:end]),
            extra_keys,
        )
        assert maybe_convert_block_hash(parent) == external_hash


def test_event_schema_keeps_existing_fields() -> None:
    event = BlockStored(
        block_hashes=[b"h"],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
        block_size=4,
        lora_id=None,
        medium=MEDIUM_GPU,
        lora_name=None,
        extra_keys=[("image-a",)],
    )
    assert event.block_hashes == [b"h"]
    assert event.token_ids == [1, 2, 3, 4]
    assert event.extra_keys == [("image-a",)]


def test_event_schema_keeps_extra_keys_optional() -> None:
    event = BlockStored(
        block_hashes=[b"legacy"],
        parent_block_hash=None,
        token_ids=[5, 6, 7, 8],
        block_size=4,
        lora_id=None,
        medium=MEDIUM_GPU,
        lora_name=None,
    )
    assert event.extra_keys is None


def test_plain_request_retains_legacy_behavior() -> None:
    request = make_request(8)
    event, _, _ = emit(request)
    assert event.block_hashes == [maybe_convert_block_hash(h) for h in request.block_hashes]
    assert event.extra_keys in (None, [None, None])


@pytest.mark.parametrize(
    "identifiers,expected",
    [
        ([(0, 4, "photo-red"), (4, 4, "photo-blue")], [("photo-red",), ("photo-blue",)]),
        ([(1, 2, "inside-first"), (6, 1, "inside-second")], [("inside-first",), ("inside-second",)]),
        ([(2, 4, "crosses-boundary")], [("crosses-boundary",), ("crosses-boundary",)]),
    ],
)
def test_multimodal_material_is_per_block(identifiers, expected) -> None:
    request = make_request(8, identifiers=identifiers)
    event, _, _ = emit(request)
    assert event.extra_keys == expected
    assert_event_reconstructs_request(event, request)


def test_equal_placeholder_tokens_distinguish_images() -> None:
    first = make_request(8, identifiers=[(0, 8, "asset-4c20")])
    second = make_request(8, identifiers=[(0, 8, "asset-9f71")])
    first_event, _, _ = emit(first)
    second_event, _, _ = emit(second)
    assert list(first.all_token_ids) == list(second.all_token_ids)
    assert first_event.extra_keys != second_event.extra_keys
    assert first_event.block_hashes != second_event.block_hashes


def test_cache_salt_only_applies_where_it_participates() -> None:
    request = make_request(12, cache_salt="tenant:42")
    event, _, _ = emit(request)
    assert event.extra_keys == [("tenant:42",), None, None]
    assert_event_reconstructs_request(event, request)


def test_lora_identity_is_available_for_every_block() -> None:
    request = make_request(12, lora_name="adapter/weather-v3")
    event, _, _ = emit(request)
    assert event.extra_keys == [("adapter/weather-v3",)] * 3
    assert event.lora_name == "adapter/weather-v3"
    assert_event_reconstructs_request(event, request)


def test_combined_identity_material_preserves_block_order() -> None:
    request = make_request(
        12,
        identifiers=[(0, 4, "image-a"), (8, 4, "image-b")],
        cache_salt="salt-z",
        lora_name="adapter-z",
    )
    event, _, _ = emit(request)
    assert event.extra_keys == [
        ("adapter-z", "image-a", "salt-z"),
        ("adapter-z",),
        ("adapter-z", "image-b"),
    ]
    assert_event_reconstructs_request(event, request)


def test_prompt_embeddings_use_compact_per_range_fingerprints() -> None:
    embeddings = torch.arange(8 * 96, dtype=torch.float32).reshape(8, 96)
    request = make_request(8, prompt_embeds=embeddings)
    event, _, _ = emit(request)
    assert event.extra_keys is not None
    raw_size = embeddings[:BLOCK_SIZE].numel() * embeddings.element_size()
    fingerprints = [keys[0] for keys in event.extra_keys if keys]
    assert len(fingerprints) == 2
    assert all(isinstance(value, bytes) for value in fingerprints)
    assert all(0 < len(value) <= 64 for value in fingerprints)
    assert all(len(value) < raw_size for value in fingerprints)
    assert fingerprints[0] != fingerprints[1]
    assert_event_reconstructs_request(event, request)


def test_prompt_embedding_fingerprint_is_stable_for_same_range() -> None:
    embeddings = torch.randn(8, 32, generator=torch.Generator().manual_seed(4831))
    request = make_request(8, prompt_embeds=embeddings)
    first, _ = generate_block_hash_extra_keys(request, 0, 4, 0)
    second, _ = generate_block_hash_extra_keys(request, 0, 4, 0)
    assert first == second
    assert first is not None
    assert first[0] != embeddings[:4].contiguous().numpy().tobytes()


def test_prompt_embedding_change_changes_event_identity() -> None:
    first_embeds = torch.zeros(8, 24)
    second_embeds = first_embeds.clone()
    second_embeds[5, 7] = 1.0
    first = make_request(8, prompt_embeds=first_embeds)
    second = make_request(8, prompt_embeds=second_embeds)
    first_event, _, _ = emit(first)
    second_event, _, _ = emit(second)
    assert first_event.extra_keys[0] == second_event.extra_keys[0]
    assert first_event.extra_keys[1] != second_event.extra_keys[1]


def test_null_blocks_do_not_break_event_alignment() -> None:
    request = make_request(
        12,
        identifiers=[(0, 4, "visible-0"), (4, 4, "skipped-1"), (8, 4, "visible-2")],
    )
    event, _, _ = emit(request, null_indices={1})
    assert len(event.block_hashes) == 2
    assert len(event.extra_keys) == len(event.block_hashes)
    assert event.extra_keys[0] == ("visible-0",)
    assert "visible-2" in event.extra_keys[1]


def test_incremental_cache_event_uses_logical_parent_and_new_material() -> None:
    request = make_request(12, lora_name="incremental-adapter")
    pool = BlockPool(
        num_gpu_blocks=16,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
        enable_kv_cache_events=True,
    )
    blocks = pool.get_new_blocks(3)
    pool.cache_full_blocks(request, blocks, 0, 2, BLOCK_SIZE, 0)
    first = pool.take_events()[0]
    pool.cache_full_blocks(request, blocks, 2, 3, BLOCK_SIZE, 0)
    second = pool.take_events()[0]
    assert first.extra_keys == [("incremental-adapter",)] * 2
    assert second.extra_keys == [("incremental-adapter",)]
    assert second.parent_block_hash == maybe_convert_block_hash(request.block_hashes[1])
    assert second.block_hashes == [maybe_convert_block_hash(request.block_hashes[2])]
    reconstructed = hash_block_tokens(
        sha256,
        request.block_hashes[1],
        tuple(request.all_token_ids[8:12]),
        second.extra_keys[0],
    )
    assert maybe_convert_block_hash(reconstructed) == second.block_hashes[0]


def test_msgpack_round_trip_preserves_extra_keys() -> None:
    request = make_request(
        8,
        identifiers=[(0, 4, "frame-a"), (4, 4, "frame-b")],
        cache_salt="router-salt",
    )
    event, _, _ = emit(request)
    batch = KVEventBatch(ts=19.25, events=[event], data_parallel_rank=3)
    encoded = msgspec.msgpack.encode(batch)
    decoded = msgspec.msgpack.decode(encoded, type=KVEventBatch)
    assert decoded.ts == 19.25
    assert decoded.data_parallel_rank == 3
    assert decoded.events[0].extra_keys == event.extra_keys
    assert decoded.events[0].block_hashes == event.block_hashes


def test_mixed_event_batch_keeps_types_and_order() -> None:
    request = make_request(4, identifiers=[(0, 4, "mm-0")])
    stored, _, _ = emit(request)
    removed = BlockRemoved(block_hashes=[stored.block_hashes[0]], medium=MEDIUM_GPU)
    blob = msgspec.msgpack.encode(KVEventBatch(ts=3.0, events=[stored, removed]))
    restored = msgspec.msgpack.decode(blob, type=KVEventBatch)
    assert [type(item) for item in restored.events] == [BlockStored, BlockRemoved]
    assert restored.events[0].extra_keys == [("mm-0",)]


def test_event_hash_accounts_for_routing_material() -> None:
    common = dict(
        block_hashes=[b"same"],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
        block_size=4,
        lora_id=None,
        medium=MEDIUM_GPU,
        lora_name=None,
    )
    first = BlockStored(**common, extra_keys=[("image-a",)])
    second = BlockStored(**common, extra_keys=[("image-b",)])
    assert hash(first) != hash(second)


def test_router_can_reconstruct_hidden_combination() -> None:
    generator = torch.Generator().manual_seed(917_533)
    embeds = torch.randn(16, 40, generator=generator)
    request = make_request(
        16,
        identifiers=[(3, 3, "media-x91"), (10, 5, "media-q07")],
        cache_salt="org/hidden",
        lora_name="finance-reranker",
        prompt_embeds=embeds,
    )
    event, _, _ = emit(request)
    assert len(event.extra_keys) == 4
    assert_event_reconstructs_request(event, request)
    assert all(
        item is None or all(not isinstance(value, memoryview) for value in item)
        for item in event.extra_keys
    )
