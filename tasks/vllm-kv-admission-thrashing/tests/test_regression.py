from types import SimpleNamespace

import vllm._C  # noqa: F401 -- proves the CPU native extension is wired in.

from vllm.v1.core.kv_cache_manager import KVCacheManager


class Coordinator:
    def __init__(self, required_blocks: int) -> None:
        self.required_blocks = required_blocks
        self.last_kwargs = None

    def get_num_blocks_to_allocate(self, **kwargs) -> int:
        self.last_kwargs = kwargs
        return self.required_blocks


class BlockPool:
    def __init__(self, free_blocks: int) -> None:
        self.free_blocks = free_blocks

    def get_num_free_blocks(self) -> int:
        return self.free_blocks


def test_full_sequence_admission_uses_required_block_count() -> None:
    manager = object.__new__(KVCacheManager)
    manager.max_model_len = 128
    manager.empty_kv_cache_blocks = SimpleNamespace(blocks=[])
    manager.coordinator = Coordinator(required_blocks=3)
    manager.block_pool = BlockPool(free_blocks=3)
    request = SimpleNamespace(
        request_id="request-1",
        num_tokens=96,
        num_computed_tokens=16,
    )

    assert manager.can_fit_full_sequence(request) is True
    assert manager.coordinator.last_kwargs["num_tokens"] == 96
    assert manager.coordinator.last_kwargs["total_computed_tokens"] == 16

    manager.block_pool.free_blocks = 2
    assert manager.can_fit_full_sequence(request) is False
