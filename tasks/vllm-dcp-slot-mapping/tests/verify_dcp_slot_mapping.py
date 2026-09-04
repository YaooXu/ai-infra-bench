#!/usr/bin/env python3
"""Behavioral verifier for production DCP slot mapping.

The verifier deliberately calls only APIs that already existed on the Base
(`BlockTables`, `append_block_ids`, and `compute_slot_mappings`). It discovers
the newly added interleave configuration semantically instead of requiring the
Oracle's parameter or helper names.
"""

from __future__ import annotations

import inspect
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import torch


PAD_SLOT_ID = -1


def expected_slots(
    positions: list[int],
    block_ids: list[int],
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    interleave: int,
) -> list[int]:
    result = []
    virtual_block_size = block_size * dcp_size
    for position in positions:
        virtual_block = position // virtual_block_size
        virtual_offset = position % virtual_block_size
        owner = (virtual_offset // interleave) % dcp_size
        if owner != dcp_rank:
            result.append(PAD_SLOT_ID)
            continue
        local_offset = (
            virtual_offset // (dcp_size * interleave)
        ) * interleave + virtual_offset % interleave
        result.append(block_ids[virtual_block] * block_size + local_offset)
    return result


def configure_group(module, dcp_size: int, dcp_rank: int) -> None:
    import vllm.distributed as distributed

    group = SimpleNamespace(world_size=dcp_size, rank_in_group=dcp_rank)
    distributed.get_dcp_group = lambda: group
    module.get_dcp_group = lambda: group


def construct_tables(module, *, dcp_size: int, dcp_rank: int, interleave: int):
    configure_group(module, dcp_size, dcp_rank)
    kwargs = {
        "block_sizes": [4, 8],
        "max_num_reqs": 2,
        "max_num_batched_tokens": 48,
        "max_model_len": 64,
        "device": torch.device("cuda"),
    }
    # Accept semantically equivalent constructor APIs. The Oracle passes the
    # scalar directly, while an independent implementation may pass a DCP/CP
    # configuration object instead; the verifier does not prescribe its name.
    parameters = inspect.signature(module.BlockTables).parameters
    scalar = [name for name in parameters if "interleave" in name]
    configs = [
        name for name in parameters
        if name not in kwargs and ("dcp" in name or "parallel_config" in name)
    ]
    if scalar:
        kwargs[scalar[0]] = interleave
    elif configs:
        kwargs[configs[0]] = SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            dcp_world_size=dcp_size,
            dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=interleave,
        )
    else:
        raise AssertionError("BlockTables has no path for the configured DCP layout")
    tables = module.BlockTables(**kwargs)
    for group_index, block_size in enumerate(kwargs["block_sizes"]):
        expected_width = ceil(kwargs["max_model_len"] / (block_size * dcp_size))
        actual_width = tables.block_tables[group_index].gpu.shape[1]
        if actual_width != expected_width:
            raise AssertionError(
                f"DCP block-table width is wrong: {actual_width}!={expected_width}"
            )
    return tables


def populate(tables):
    per_request = []
    for req_index, base in enumerate((10, 40)):
        groups = []
        for group_index, table in enumerate(tables.block_tables):
            width = table.gpu.shape[1]
            groups.append([base + group_index * 20 + i for i in range(width)])
        tables.append_block_ids(req_index, tuple(groups), overwrite=True)
        per_request.append(groups)
    tables.apply_staged_writes()
    return per_request


def run_case(module, *, dcp_size: int, dcp_rank: int, interleave: int) -> None:
    tables = construct_tables(
        module, dcp_size=dcp_size, dcp_rank=dcp_rank, interleave=interleave
    )
    block_ids = populate(tables)
    positions = [0, 1, 3, 4, 7, 8, 15, 16, 23, 31, 2, 5, 9, 14, 18, 27]
    split = 10
    slots = tables.compute_slot_mappings(
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        torch.tensor([0, split, len(positions)], dtype=torch.int32, device="cuda"),
        torch.tensor(positions, dtype=torch.int64, device="cuda"),
    )
    torch.cuda.synchronize()
    for group_index, block_size in enumerate((4, 8)):
        expected = expected_slots(
            positions[:split],
            block_ids[0][group_index],
            block_size,
            dcp_size,
            dcp_rank,
            interleave,
        ) + expected_slots(
            positions[split:],
            block_ids[1][group_index],
            block_size,
            dcp_size,
            dcp_rank,
            interleave,
        )
        actual = slots[group_index].cpu().tolist()
        if actual != expected:
            raise AssertionError(
                f"slot mismatch size={dcp_size} rank={dcp_rank} "
                f"interleave={interleave} group={group_index}: "
                f"expected={expected} actual={actual}"
            )


def check_graph_replay(module) -> None:
    dcp_size, dcp_rank, interleave = 2, 1, 2
    tables = construct_tables(
        module, dcp_size=dcp_size, dcp_rank=dcp_rank, interleave=interleave
    )
    block_ids = populate(tables)
    idx = torch.tensor([0], dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
    positions = torch.arange(16, dtype=torch.int64, device="cuda")
    tables.compute_slot_mappings(idx, starts, positions)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        slots = tables.compute_slot_mappings(idx, starts, positions)

    heldout = [15, 0, 14, 1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7]
    positions.copy_(torch.tensor(heldout, dtype=torch.int64, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    expected = expected_slots(
        heldout, block_ids[0][0], 4, dcp_size, dcp_rank, interleave
    )
    actual = slots[0].cpu().tolist()
    if actual != expected:
        raise AssertionError(
            f"CUDA graph replay used stale or incorrect inputs: {actual}!={expected}"
        )


def check_graph_metadata_flow() -> None:
    """Ensure DCP-local lengths reach the attention backend on graph warm-up."""
    import vllm.v1.worker.gpu.block_table as block_table_module
    import vllm.v1.worker.gpu.cudagraph_utils as graph_module
    from vllm.v1.worker.gpu.input_batch import InputBuffers

    dcp_size, dcp_rank, interleave = 2, 1, 2
    configure_group(graph_module, dcp_size, dcp_rank)
    tables = construct_tables(
        block_table_module,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        interleave=interleave,
    )
    populate(tables)
    buffers = InputBuffers(max_num_reqs=2, max_num_tokens=48, device=torch.device("cuda"))
    captured = []

    class CaptureBuilder:
        def build(self, *, common_prefix_len, common_attn_metadata):
            assert common_prefix_len == 0
            captured.append(common_attn_metadata)
            return common_attn_metadata

    cache_group = SimpleNamespace(layer_names=["layer"], kv_cache_spec=object())
    cache_config = SimpleNamespace(kv_cache_groups=[cache_group])
    metadata, slots_by_layer = graph_module.prepare_inputs_to_capture(
        num_reqs=2,
        num_tokens=16,
        input_buffers=buffers,
        block_tables=tables,
        attn_metadata_builders=[CaptureBuilder()],
        max_model_len=64,
        kv_cache_config=cache_config,
    )
    assert "layer" in metadata and "layer" in slots_by_layer
    assert len(captured) == 1
    local_lens = captured[0].dcp_local_seq_lens
    if local_lens is None:
        raise AssertionError("CUDA-graph attention metadata omitted DCP-local lengths")
    torch.cuda.synchronize()
    actual = local_lens.cpu().tolist()
    if actual != [8, 8]:
        raise AssertionError(f"wrong DCP-local graph sequence lengths: {actual}")


def main() -> int:
    import vllm
    import vllm.v1.worker.gpu.block_table as block_table_module

    repo = Path("/workspace/repo").resolve()
    source = Path(vllm.__file__).resolve()
    if repo not in source.parents:
        raise AssertionError(f"candidate source is not active: {source}")
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required")

    for case in ((1, 0, 1), (2, 0, 1), (2, 1, 2), (4, 3, 2)):
        run_case(
            block_table_module,
            dcp_size=case[0],
            dcp_rank=case[1],
            interleave=case[2],
        )
    check_graph_replay(block_table_module)
    check_graph_metadata_flow()
    print(
        "PASS: production slot mapping handles non-DCP, held-out DCP ranks, "
        "interleaving, multiple cache groups, requests, CUDA graph replay, "
        "and graph attention metadata"
    )
    print(f"candidate_source={source} gpu={torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
