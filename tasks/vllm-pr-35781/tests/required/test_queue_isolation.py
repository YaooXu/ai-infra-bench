from unittest.mock import Mock

import pytest
from tests.v1.core.utils import create_requests, create_scheduler, mock_kv
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def test_async_remote_kv_requests_leave_the_schedulable_waiting_queue():
    scheduler = create_scheduler(
        max_num_seqs=16,
        use_kv_connector=mock_kv(matched_tokens=8, is_async=True),
    )
    requests = create_requests(num_requests=8, num_tokens=10)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert not output.scheduled_new_reqs
    assert not scheduler.waiting
    assert list(scheduler.skipped_waiting) == requests
    assert all(
        request.status == RequestStatus.WAITING_FOR_REMOTE_KVS for request in requests
    )

    for _ in range(3):
        scheduler.schedule()
        assert not scheduler.waiting
        assert list(scheduler.skipped_waiting) == requests


def test_blocked_requests_are_counted_reported_and_cleaned_on_abort():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=3)
    requests[0].status = RequestStatus.WAITING_FOR_FSM
    requests[0].structured_output_request = Mock(grammar=None)
    requests[1].status = RequestStatus.WAITING_FOR_REMOTE_KVS

    for request in requests:
        scheduler.add_request(request)

    assert list(scheduler.skipped_waiting) == requests[:2]
    assert list(scheduler.waiting) == requests[2:]
    assert scheduler.get_request_counts() == (0, 3)
    assert scheduler.get_num_unfinished_requests() == 3

    stats = scheduler.make_stats()
    assert stats is not None
    assert stats.num_waiting_reqs == 3

    aborted_request = requests[0]
    scheduler.finish_requests(
        aborted_request.request_id,
        RequestStatus.FINISHED_ABORTED,
    )
    assert aborted_request.request_id not in scheduler.requests
    assert aborted_request not in scheduler.waiting
    assert aborted_request not in scheduler.skipped_waiting
    assert scheduler.get_request_counts() == (0, 2)
