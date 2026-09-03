from __future__ import annotations

import json

from tests.v1.core.utils import create_requests, create_scheduler, mock_kv
from vllm.v1.request import RequestStatus


request_count = 24
rounds = 5
scheduler = create_scheduler(
    model="/opt/bench/model-fixture",
    max_num_seqs=request_count,
    use_kv_connector=mock_kv(matched_tokens=8, is_async=True),
    skip_tokenizer_init=True,
)
requests = create_requests(num_requests=request_count, num_tokens=10)
for request in requests:
    scheduler.add_request(request)

first = scheduler.schedule()
assert not first.scheduled_new_reqs
assert all(
    request.status == RequestStatus.WAITING_FOR_REMOTE_KVS for request in requests
)

callback_calls = 0
original_update = scheduler._update_waiting_for_remote_kv


def counted_update(request):
    global callback_calls
    callback_calls += 1
    return original_update(request)


scheduler._update_waiting_for_remote_kv = counted_update
for _ in range(rounds):
    scheduler.schedule()

expected_calls = request_count * rounds
evidence = {
    "requests": request_count,
    "idle_rounds": rounds,
    "remote_kv_callbacks": callback_calls,
    "expected_legacy_callbacks": expected_calls,
    "ordinary_waiting_queue_size": len(scheduler.waiting),
    "has_separate_skipped_waiting_queue": hasattr(scheduler, "skipped_waiting"),
    "baseline_bug_reproduced": callback_calls == expected_calls,
}
print(json.dumps(evidence, sort_keys=True))
assert callback_calls == expected_calls
assert len(scheduler.waiting) == request_count
assert not hasattr(scheduler, "skipped_waiting")
