#!/usr/bin/env python3
"""Verify blocked requests do not incur an O(queue) scan every idle round."""

from __future__ import annotations

import json

from tests.v1.core.utils import create_requests, create_scheduler, mock_kv
from vllm.v1.request import RequestStatus


REQUEST_COUNT = 24
IDLE_ROUNDS = 5


def main() -> None:
    scheduler = create_scheduler(
        model="/opt/bench/model-fixture",
        max_num_seqs=REQUEST_COUNT,
        use_kv_connector=mock_kv(matched_tokens=8, is_async=True),
        skip_tokenizer_init=True,
    )
    requests = create_requests(num_requests=REQUEST_COUNT, num_tokens=10)
    for request in requests:
        scheduler.add_request(request)

    first = scheduler.schedule()
    assert not first.scheduled_new_reqs
    assert all(
        request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        for request in requests
    )
    assert hasattr(scheduler, "skipped_waiting")
    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == REQUEST_COUNT

    callback_calls = 0
    original_update = scheduler._update_waiting_for_remote_kv

    def counted_update(request):
        nonlocal callback_calls
        callback_calls += 1
        return original_update(request)

    scheduler._update_waiting_for_remote_kv = counted_update
    for _ in range(IDLE_ROUNDS):
        scheduler.schedule()

    assert callback_calls == 0
    assert scheduler.get_request_counts() == (0, REQUEST_COUNT)
    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == REQUEST_COUNT
    print(
        json.dumps(
            {
                "requests": REQUEST_COUNT,
                "idle_rounds": IDLE_ROUNDS,
                "remote_kv_callbacks": callback_calls,
                "blocked_queue_size": len(scheduler.skipped_waiting),
                "ordinary_waiting_queue_size": len(scheduler.waiting),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

