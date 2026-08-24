#!/usr/bin/env python3
"""Focused GPU behavior contract for model-runner-v2 DCP metadata."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch


def _expected_slots(positions: list[int]) -> list[int]:
    expected: list[int] = []
    for position in positions:
        virtual_block = position // 8
        virtual_offset = position % 8
        is_rank_one = (virtual_offset // 2) % 2 == 1
        if not is_rank_one:
            expected.append(-1)
            continue
        local_offset = (virtual_offset // 4) * 2 + virtual_offset % 2
        expected.append((10 + virtual_block) * 4 + local_offset)
    return expected


def main() -> int:
    try:
        import vllm.v1.worker.gpu.block_table as block_table_module
        from vllm.v1.worker.gpu.attn_utils import prepare_dcp_local_seq_lens
    except ImportError as exc:
        print(
            "FAIL: base GPU model runner v2 has no DCP local-sequence helper; "
            "the real CUDA slot-mapping contract cannot be prepared",
            file=sys.stderr,
        )
        print(f"DETAIL: {exc}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("FAIL: CUDA is unavailable", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    block_table_module.get_dcp_group = lambda: SimpleNamespace(
        world_size=2, rank_in_group=1
    )

    try:
        tables = block_table_module.BlockTables(
            block_sizes=[4],
            max_num_reqs=1,
            max_num_batched_tokens=16,
            max_model_len=16,
            device=device,
            cp_kv_cache_interleave_size=2,
        )
    except TypeError as exc:
        print(
            "FAIL: base BlockTables does not accept DCP KV-cache interleaving",
            file=sys.stderr,
        )
        print(f"DETAIL: {exc}", file=sys.stderr)
        return 1

    assert tables.block_tables[0].gpu.shape == (1, 2)
    tables.append_block_ids(0, ([10, 11],), overwrite=True)
    tables.apply_staged_writes()

    idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)
    query_start = torch.tensor([0, 16], dtype=torch.int32, device=device)
    positions = torch.arange(16, dtype=torch.int64, device=device)
    seq_lens = torch.arange(10, dtype=torch.int32, device=device)
    local_seq_lens = torch.full((12,), -99, dtype=torch.int32, device=device)

    prepare_dcp_local_seq_lens(
        local_seq_lens,
        seq_lens,
        num_reqs=10,
        dcp_size=2,
        dcp_rank=1,
        cp_kv_cache_interleave_size=2,
    )
    slots = tables.compute_slot_mappings(idx_mapping, query_start, positions)
    torch.cuda.synchronize()
    assert local_seq_lens.cpu().tolist() == [0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 0, 0]
    assert slots[0].cpu().tolist() == _expected_slots(list(range(16)))

    # Capture the production slot-mapping kernel over persistent buffers, mutate
    # its inputs, and replay it. The local-sequence helper creates a small rank
    # offset tensor, so production prepares that metadata outside graph capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        slots = tables.compute_slot_mappings(idx_mapping, query_start, positions)

    new_seq_lens = list(reversed(range(10)))
    new_positions = list(reversed(range(16)))
    seq_lens.copy_(torch.tensor(new_seq_lens, dtype=torch.int32, device=device))
    positions.copy_(torch.tensor(new_positions, dtype=torch.int64, device=device))
    prepare_dcp_local_seq_lens(
        local_seq_lens,
        seq_lens,
        num_reqs=10,
        dcp_size=2,
        dcp_rank=1,
        cp_kv_cache_interleave_size=2,
    )
    graph.replay()
    torch.cuda.synchronize()

    assert local_seq_lens.cpu().tolist() == [4, 4, 3, 2, 2, 2, 1, 0, 0, 0, 0, 0]
    assert slots[0].cpu().tolist() == _expected_slots(new_positions)

    props = torch.cuda.get_device_properties(0)
    print(
        "PASS: exact DCP rank-1 virtual-block/interleave mapping ran through "
        "the production Triton kernel and replayed from a CUDA graph"
    )
    print(
        f"gpu={props.name} capability={props.major}.{props.minor} "
        f"uuid={props.uuid} slots={slots[0].cpu().tolist()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
