# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeXferMetadata,
    SendBlockMeta,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from test_regression import RecordingEngine, hybrid_caches, hybrid_config, make_worker


def regions(worker):
    return worker._get_transfer_regions(
        worker.kv_caches_base_addr,
        worker.block_len_per_layer,
        worker.kv_block_len_per_layer,
        worker.registered_layer_names,
        worker.registered_layer_indices,
        worker.registered_group_indices,
    )


def main() -> None:
    config = hybrid_config(num_blocks=10)
    engine = RecordingEngine()
    producer = make_worker(config, engine=engine)
    consumer = make_worker(config)
    producer_caches = hybrid_caches(10)
    consumer_caches = hybrid_caches(10)
    producer.register_kv_caches(producer_caches)
    consumer.register_kv_caches(consumer_caches)

    producer_fa = producer_caches["model.layers.0.self_attn"]
    producer_gdn = producer_caches["model.layers.1.linear_attn"][0]
    consumer_fa = consumer_caches["model.layers.0.self_attn"]
    consumer_gdn = consumer_caches["model.layers.1.linear_attn"][0]
    producer_fa[1].fill_(11)
    producer_fa[2].fill_(22)
    producer_gdn[3].fill_(33)
    fa_neighbor_before = consumer_fa[4].clone()
    gdn_neighbor_before = consumer_gdn[6].clone()

    send = SendBlockMeta(
        p_req_id="producer-request",
        transfer_id="transfer-real",
        local_block_ids=[[1, 2], [NULL_BLOCK_ID, 3]],
        ready=asyncio.Event(),
    )
    metadata = MooncakeXferMetadata(
        remote_hostname="in-memory",
        remote_port=0,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={
            "decoder-request": (
                "transfer-real",
                [[5, 6], [NULL_BLOCK_ID, 7]],
            )
        },
        kv_caches_base_addr=consumer.kv_caches_base_addr,
        block_lens=consumer.block_len_per_layer,
        kv_block_lens=consumer.kv_block_len_per_layer,
        registered_layer_names=consumer.registered_layer_names,
        registered_layer_indices=consumer.registered_layer_indices,
        registered_group_indices=consumer.registered_group_indices,
    )
    local_regions, remote_regions, error = __import__(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector",
        fromlist=["_align_transfer_regions"],
    )._align_transfer_regions(regions(producer), regions(consumer))
    assert error is None
    src, dst, lengths, errors, message = asyncio.run(
        producer._build_transfer_params(
            [("decoder-request", send)], metadata, local_regions, remote_regions
        )
    )
    assert errors == [] and message is None
    assert engine.batch_transfer_sync_write("in-memory", src, dst, lengths) == 0
    assert torch.equal(consumer_fa[5], producer_fa[1])
    assert torch.equal(consumer_fa[6], producer_fa[2])
    assert torch.equal(consumer_gdn[7], producer_gdn[3])
    assert torch.equal(consumer_fa[4], fa_neighbor_before)
    assert torch.equal(consumer_gdn[6], gdn_neighbor_before)
    assert sum(lengths) == producer_fa[1].nbytes * 2 + producer_gdn[3].nbytes
    print(
        "REAL_INMEMORY_MOONCAKE_PD_OK "
        f"groups=2 descriptors={len(lengths)} bytes={sum(lengths)}"
    )


if __name__ == "__main__":
    main()
