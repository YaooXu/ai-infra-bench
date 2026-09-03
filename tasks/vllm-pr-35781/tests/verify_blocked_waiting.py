#!/usr/bin/env python3
"""Behavioral and deterministic-work contract for remote-KV waiting."""

from __future__ import annotations

import copy
import json

from tests.v1.core.utils import create_requests, create_scheduler, mock_kv
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
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
    assert all(r.status == RequestStatus.WAITING_FOR_REMOTE_KVS for r in requests)
    assert scheduler.get_request_counts() == (0, REQUEST_COUNT)

    # Count the production readiness callback, not a particular queue or helper
    # layout. Unchanged external state must not cause one callback per blocked
    # request on every idle tick.
    callback_calls = 0
    waiting_pops = 0
    original_update = scheduler._update_waiting_for_remote_kv
    original_pop = scheduler.waiting.pop_request

    def counted_update(request):
        nonlocal callback_calls
        callback_calls += 1
        return original_update(request)

    scheduler._update_waiting_for_remote_kv = counted_update

    def counted_pop():
        nonlocal waiting_pops
        waiting_pops += 1
        return original_pop()

    scheduler.waiting.pop_request = counted_pop
    for _ in range(IDLE_ROUNDS):
        output = scheduler.schedule()
        assert not output.scheduled_new_reqs
        assert scheduler.get_request_counts() == (0, REQUEST_COUNT)

    if callback_calls > 1:
        raise AssertionError(
            "idle remote-KV work still scales with the blocked population: "
            f"callbacks={callback_calls}"
        )
    if waiting_pops > IDLE_ROUNDS:
        raise AssertionError(
            "idle scheduler still churns the ordinary waiting queue: "
            f"pops={waiting_pops}"
        )

    # A real connector completion event must make only the named requests
    # eligible, in their original FCFS order, without losing the rest from
    # unfinished accounting.
    ready_ids = {requests[i].request_id for i in (2, 7, 13)}
    finished = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    finished.kv_connector_output = KVConnectorOutput(finished_recving=ready_ids)
    scheduler.update_from_output(output, finished)
    resumed = scheduler.schedule()
    resumed_ids = [request.req_id for request in resumed.scheduled_new_reqs]
    expected_ids = [r.request_id for r in requests if r.request_id in ready_ids]
    if resumed_ids != expected_ids:
        raise AssertionError(f"completion promotion/order changed: {resumed_ids}")
    running, waiting = scheduler.get_request_counts()
    assert running == len(ready_ids) and waiting == REQUEST_COUNT - len(ready_ids)

    # Abort/removal must also find a blocked request regardless of the internal
    # container chosen by an implementation.
    victim = requests[-1]
    scheduler.finish_requests(victim.request_id, RequestStatus.FINISHED_ABORTED)
    assert victim.status == RequestStatus.FINISHED_ABORTED

    print(json.dumps({
        "requests": REQUEST_COUNT,
        "idle_rounds": IDLE_ROUNDS,
        "remote_kv_callbacks": callback_calls,
        "ordinary_waiting_pops": waiting_pops,
        "resumed": resumed_ids,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
