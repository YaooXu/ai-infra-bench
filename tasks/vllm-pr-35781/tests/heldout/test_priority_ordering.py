from unittest.mock import Mock

import pytest
from tests.v1.core.test_scheduler import (
    create_requests_with_priority,
    create_scheduler_with_priority,
)
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def test_priority_order_is_preserved_across_blocked_and_ready_requests():
    scheduler = create_scheduler_with_priority(max_num_seqs=8)
    requests = create_requests_with_priority(
        num_requests=4,
        priorities=[3, 0, 2, 1],
        arrival_times=[1.0, 2.0, 3.0, 4.0],
        req_ids=["lowest", "blocked-highest", "low", "high"],
    )
    lowest, blocked_highest, low, high = requests
    blocked_highest.status = RequestStatus.WAITING_FOR_FSM
    blocked_highest.structured_output_request = Mock(grammar=object())

    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert [request.req_id for request in output.scheduled_new_reqs] == [
        blocked_highest.request_id,
        high.request_id,
        low.request_id,
        lowest.request_id,
    ]


def test_blocked_high_priority_request_does_not_starve_ready_work():
    scheduler = create_scheduler_with_priority(max_num_seqs=1)
    blocked, ready = create_requests_with_priority(
        num_requests=2,
        priorities=[0, 1],
        arrival_times=[1.0, 2.0],
        req_ids=["blocked", "ready"],
    )
    blocked.status = RequestStatus.WAITING_FOR_FSM
    blocked.structured_output_request = Mock(grammar=None)
    scheduler.add_request(blocked)
    scheduler.add_request(ready)

    output = scheduler.schedule()
    assert [request.req_id for request in output.scheduled_new_reqs] == [
        ready.request_id
    ]
    assert blocked.status == RequestStatus.WAITING_FOR_FSM

    scheduler.finish_requests(ready.request_id, RequestStatus.FINISHED_ABORTED)
    blocked.structured_output_request = Mock(grammar=object())
    output = scheduler.schedule()
    assert [request.req_id for request in output.scheduled_new_reqs] == [
        blocked.request_id
    ]
