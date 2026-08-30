# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import socket
import time

import msgspec
import zmq

from vllm.distributed.kv_events import KVEventBatch, ZmqEventPublisher
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash

from test_regression import assert_event_reconstructs_request, emit, make_request


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def main() -> None:
    init_none_hash(sha256)
    port = free_tcp_port()
    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"routing")
    subscriber.setsockopt(zmq.RCVTIMEO, 5000)
    subscriber.connect(f"tcp://127.0.0.1:{port}")
    publisher = ZmqEventPublisher(
        data_parallel_rank=0,
        endpoint=f"tcp://*:{port}",
        topic="routing",
    )
    try:
        time.sleep(0.35)
        request = make_request(
            12,
            identifiers=[(0, 4, "camera-a"), (8, 4, "camera-b")],
            cache_salt="production-tenant",
            lora_name="router-adapter",
        )
        event, _, _ = emit(request)
        publisher.publish(KVEventBatch(ts=time.time(), events=[event]))
        topic, sequence, payload = subscriber.recv_multipart()
        assert topic == b"routing"
        assert int.from_bytes(sequence, "big") == 0
        decoded = msgspec.msgpack.decode(payload, type=KVEventBatch)
        assert decoded.data_parallel_rank == 0
        assert len(decoded.events) == 1
        received = decoded.events[0]
        assert received.block_hashes == event.block_hashes
        assert received.extra_keys == event.extra_keys
        assert_event_reconstructs_request(received, request)
        print(
            "REAL_EVENT_TRANSPORT_OK "
            f"blocks={len(received.block_hashes)} extras={len(received.extra_keys)}"
        )
    finally:
        publisher.shutdown()
        subscriber.close(linger=0)


if __name__ == "__main__":
    main()
