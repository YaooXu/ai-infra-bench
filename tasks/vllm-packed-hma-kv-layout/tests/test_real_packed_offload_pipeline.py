# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

from test_regression import attention_groups, canonicalize, make_config


def main() -> None:
    config = make_config(
        attention_groups(layers_per_group=(3, 2, 3), varied_pages=True),
        packed=True,
        available_memory=24 * 1024 * 1024,
    )
    canonical, views = canonicalize(config)
    assert len(canonical.tensors) == 1
    assert len(canonical.group_data_refs) == 3

    cpu_config = SimpleCPUOffloadScheduler._derive_cpu_config(
        config, config.kv_cache_tensors[0].size * 2
    )
    assert cpu_config.num_blocks == config.num_blocks * 2
    assert [t.offset for t in cpu_config.kv_cache_tensors] == [
        t.offset for t in config.kv_cache_tensors
    ]

    for index, descriptor in enumerate(config.kv_cache_tensors, start=1):
        views[descriptor.shared_by[0]].fill_(index * 23)
    canonical_rows = canonical.tensors[0].tensor
    untouched_before = canonical_rows[1].clone()
    selected = torch.tensor([0, 3, 5], dtype=torch.long)
    cpu_rows = canonical_rows.index_select(0, selected).clone()
    canonical_rows.index_fill_(0, selected, 0)
    canonical_rows.index_copy_(0, selected, cpu_rows)
    assert torch.equal(canonical_rows.index_select(0, selected), cpu_rows)
    assert torch.equal(canonical_rows[1], untouched_before)
    assert {v.untyped_storage().data_ptr() for v in views.values()} == {
        canonical_rows.untyped_storage().data_ptr()
    }
    for index, descriptor in enumerate(config.kv_cache_tensors, start=1):
        assert torch.all(views[descriptor.shared_by[0]] == index * 23)
    print(
        "REAL_PACKED_OFFLOAD_PIPELINE_OK "
        f"groups=3 tensors={len(canonical.tensors)} blocks={config.num_blocks} "
        f"cpu_blocks={cpu_config.num_blocks}"
    )


if __name__ == "__main__":
    main()
