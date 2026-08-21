from unittest.mock import Mock

import pytest
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def test_fcfs_mixed_blocked_waiting_types_keep_order():
    scheduler = create_scheduler(max_num_batched_tokens=20)
    scheduler._update_waiting_for_remote_kv = Mock()

    def make_request(req_id: str, num_tokens: int = 1):
        return create_requests(
            num_requests=1,
            num_tokens=num_tokens,
            req_ids=[req_id],
        )[0]

    req_fsm = make_request("fsm")
    req_remote = make_request("remote")
    req_stream = make_request("stream")
    req_regular = make_request("regular", 20)
    req_tail = make_request("tail")

    req_fsm.status = RequestStatus.WAITING_FOR_FSM
    req_fsm.structured_output_request = Mock(grammar=None)
    req_remote.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    req_stream.status = RequestStatus.WAITING_FOR_STREAMING_REQ

    for req in (req_fsm, req_remote, req_stream, req_regular, req_tail):
        scheduler.add_request(req)

    scheduler.schedule()
    assert list(scheduler.skipped_waiting) == [req_fsm, req_remote, req_stream]

    scheduler.finish_requests(req_regular.request_id, RequestStatus.FINISHED_ABORTED)
    assert not scheduler.running

    req_fsm.structured_output_request = Mock(grammar=object())
    scheduler.finished_recving_kv_req_ids.add(req_remote.request_id)
    req_stream.status = RequestStatus.WAITING

    output = scheduler.schedule()
    expected_order = [
        req_fsm.request_id,
        req_remote.request_id,
        req_stream.request_id,
        req_tail.request_id,
    ]
    assert [req.req_id for req in output.scheduled_new_reqs] == expected_order
    assert [req.request_id for req in scheduler.running] == expected_order
    scheduler._update_waiting_for_remote_kv.assert_called_once_with(req_remote)
