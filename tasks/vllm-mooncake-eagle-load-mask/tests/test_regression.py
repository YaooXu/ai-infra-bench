from .test_mooncake_store_coordinator import (
    ExternalCachedBlockPool,
    KVCacheGroupSpec,
    _full,
    _hashes,
    _make_coord,
    _swa,
)


def test_eagle_lookup_adjustment_is_applied_once_to_receive_mask():
    groups = [KVCacheGroupSpec(["full"], _full(16))]
    coordinator = _make_coord(groups, hash_block_size=16, use_eagle=True)
    hashes = _hashes(4)
    cached = ExternalCachedBlockPool({(0, bytes(value)) for value in hashes})

    _, hit_length = coordinator.find_longest_cache_hit(
        hashes, max_length=64, cached_block_pool=cached
    )
    assert hit_length == 48

    masks = coordinator.load_mask(hashes, token_len=hit_length)
    assert masks[0] == [True, True, True]


def test_hybrid_mask_keeps_full_attention_chunks_and_swa_tail():
    groups = [
        KVCacheGroupSpec(["full"], _full(16)),
        KVCacheGroupSpec(["swa"], _swa(16, 32)),
    ]
    coordinator = _make_coord(groups, hash_block_size=16, use_eagle=True)
    hashes = _hashes(4)
    cached = ExternalCachedBlockPool(
        {(group, bytes(value)) for group in (0, 1) for value in hashes}
    )
    _, hit_length = coordinator.find_longest_cache_hit(
        hashes, max_length=64, cached_block_pool=cached
    )

    masks = coordinator.load_mask(hashes, token_len=hit_length)
    assert hit_length == 48
    assert masks[0] == [True, True, True]
    assert masks[1][-2:] == [True, True]


def test_non_eagle_receive_mask_is_unchanged():
    groups = [KVCacheGroupSpec(["full"], _full(16))]
    coordinator = _make_coord(groups, hash_block_size=16, use_eagle=False)
    hashes = _hashes(4)
    cached = ExternalCachedBlockPool({(0, bytes(value)) for value in hashes})
    _, hit_length = coordinator.find_longest_cache_hit(
        hashes, max_length=64, cached_block_pool=cached
    )

    assert hit_length == 64
    assert coordinator.load_mask(hashes, token_len=hit_length)[0] == [True] * 4
