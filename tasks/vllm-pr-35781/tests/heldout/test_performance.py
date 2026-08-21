import time
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest
from tests.v1.core.utils import create_requests, create_scheduler, mock_kv
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test


def _make_blocked_remote_scheduler(num_requests: int):
    scheduler = create_scheduler(
        max_num_seqs=max(16, num_requests),
        use_kv_connector=mock_kv(matched_tokens=8, is_async=True),
    )
    requests = create_requests(num_requests=num_requests, num_tokens=10)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert not output.scheduled_new_reqs
    assert all(
        request.status == RequestStatus.WAITING_FOR_REMOTE_KVS for request in requests
    )
    return scheduler, requests


def test_remote_kv_update_work_is_event_driven():
    """Blocked remote requests must not trigger expensive update work.

    Before the optimization, every scheduler pass called the remote-KV update
    routine for every blocked request. The optimized path first checks the
    completion-event set and calls the update routine only for completed IDs.
    """
    scheduler, _ = _make_blocked_remote_scheduler(96)
    update_spy = Mock(wraps=scheduler._update_waiting_for_remote_kv)
    scheduler._update_waiting_for_remote_kv = update_spy

    for _ in range(8):
        scheduler.schedule()

    assert update_spy.call_count == 0, (
        "remote-KV update work was repeated without a completion event: "
        f"{update_spy.call_count} calls"
    )


def test_blocked_requests_do_not_churn_the_schedulable_queue(monkeypatch):
    """Repeated blocked passes must not mutate the ordinary waiting queue."""
    scheduler, _ = _make_blocked_remote_scheduler(96)
    waiting_queue = scheduler.waiting
    queue_type = type(waiting_queue)
    original_pop: Callable[..., Any] = queue_type.pop_request
    original_prepend: Callable[..., Any] = queue_type.prepend_requests
    main_queue_work = {"pops": 0, "prepended": 0}

    def counted_pop(queue):
        if queue is waiting_queue:
            main_queue_work["pops"] += 1
        return original_pop(queue)

    def counted_prepend(queue, requests):
        if queue is waiting_queue:
            main_queue_work["prepended"] += len(requests)
        return original_prepend(queue, requests)

    monkeypatch.setattr(queue_type, "pop_request", counted_pop)
    monkeypatch.setattr(queue_type, "prepend_requests", counted_prepend)

    for _ in range(8):
        scheduler.schedule()

    assert main_queue_work == {"pops": 0, "prepended": 0}, (
        f"blocked requests still churn the schedulable waiting queue: {main_queue_work}"
    )


def test_blocked_scheduler_cpu_microbenchmark():
    """Paired CPU microbenchmark of the remote-ready hot path.

    The legacy baseline explicitly performs the per-request readiness work.
    Both measurements run in the same process, using the same callback and
    request population, so the ratio is substantially less sensitive to host
    load than an absolute wall-clock limit.
    """
    scheduler, requests = _make_blocked_remote_scheduler(128)
    callback_calls = 0
    checksum = 0

    def readiness_update(_: Request) -> bool:
        nonlocal callback_calls, checksum
        callback_calls += 1
        value = 0x9E3779B1
        for index in range(1200):
            value = ((value << 5) - value + index) & 0xFFFFFFFF
        checksum ^= value
        return False

    scheduler._update_waiting_for_remote_kv = readiness_update
    rounds = 6

    # Warm scheduler data structures before timing.
    scheduler.schedule()
    callback_calls = 0

    started = time.perf_counter_ns()
    for _ in range(rounds):
        scheduler.schedule()
    optimized_ns = time.perf_counter_ns() - started
    optimized_calls = callback_calls

    callback_calls = 0
    started = time.perf_counter_ns()
    for _ in range(rounds):
        for request in requests:
            readiness_update(request)
    legacy_ns = time.perf_counter_ns() - started
    legacy_calls = callback_calls

    # Keep the synthetic work observable and report timings in verifier logs.
    assert checksum >= 0
    print(
        "blocked scheduler microbenchmark: "
        f"optimized={optimized_ns / 1e6:.3f}ms, "
        f"legacy={legacy_ns / 1e6:.3f}ms, "
        f"ratio={optimized_ns / legacy_ns:.3f}, "
        f"optimized_callbacks={optimized_calls}, "
        f"legacy_callbacks={legacy_calls}"
    )

    assert legacy_calls == len(requests) * rounds
    assert optimized_calls == 0
    assert optimized_ns < legacy_ns * 0.50
