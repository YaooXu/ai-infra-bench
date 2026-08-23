from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus


class RequestDouble:
    def __init__(self, request_id: str, specs: list[tuple[str, int, list[bool]]]):
        self.request_id = request_id
        self.mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=identifier,
                mm_position=PlaceholderRange(offset, len(mask), torch.tensor(mask)),
            )
            for identifier, offset, mask in specs
        ]
        self.has_encoder_inputs = bool(self.mm_features)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        return self.mm_features[input_id].mm_position.length

    def get_num_encoder_embeds(self, input_id: int) -> int:
        value = self.mm_features[input_id].mm_position.get_num_embeds
        return int(value() if callable(value) else value)


class RunningRequestDouble(RequestDouble):
    """Minimum production-Scheduler request surface for a preemption step."""

    def __init__(self, request_id: str, priority: int):
        super().__init__(
            request_id,
            [(request_id, 0, [False, True, False, True, False])],
        )
        self.priority = priority
        self.arrival_time = float(priority)
        self.status = RequestStatus.RUNNING
        self.num_output_placeholders = 0
        self.num_computed_tokens = 0
        self.num_tokens_with_spec = 5
        self.num_prompt_tokens = 5
        self.max_tokens = 1
        self.spec_token_ids = []
        self.lora_request = None
        self.num_preemptions = 0


class KVCacheDouble:
    """Force one real priority preemption while tracking later allocations."""

    def __init__(self):
        self.attempts: dict[str, int] = {}

    def allocate_slots(self, request, *_args, **_kwargs):
        count = self.attempts.get(request.request_id, 0) + 1
        self.attempts[request.request_id] = count
        if request.request_id == "b" and count == 1:
            return None
        return object()

    def free(self, _request):
        return None

    def get_num_common_prefix_blocks(self, _request_id):
        return [0]


def _scheduler(cache_size: int = 100, *, disable_chunking: bool = False):
    return SimpleNamespace(
        ec_connector=None,
        is_encoder_decoder=False,
        encoder_cache_manager=EncoderCacheManager(cache_size),
        scheduler_config=SimpleNamespace(disable_chunked_mm_input=disable_chunking),
    )


def _try(scheduler, request, computed, new, budget):
    return Scheduler._try_schedule_encoder_inputs(
        scheduler,
        request,
        num_computed_tokens=computed,
        num_new_tokens=new,
        encoder_compute_budget=budget,
    )


def test_cache_allocate_budget_free_and_eviction_use_embedding_count():
    request = RequestDouble(
        "r1", [("first", 0, [False] * 92 + [True] * 8)]
    )
    manager = EncoderCacheManager(cache_size=8)
    assert manager.can_allocate(request, 0, 8, 0)
    assert not manager.can_allocate(request, 0, 7, 0)
    manager.allocate(request, 0)
    assert manager.num_free_slots == 0
    manager.free_encoder_input(request, 0)
    assert manager.num_freeable_slots == 8

    replacement = RequestDouble("r2", [("second", 0, [True] * 8)])
    assert manager.can_allocate(replacement, 0, 8, 0)
    manager.allocate(replacement, 0)
    assert "first" not in manager.cached
    assert manager.num_free_slots == 0


def test_cache_hit_shared_hash_and_multiple_items_do_not_double_charge():
    first = RequestDouble(
        "r1",
        [
            ("shared", 0, [True, False, True, False]),
            ("unique", 4, [False, True, False]),
        ],
    )
    second = RequestDouble("r2", [("shared", 0, [True, False, True, False])])
    manager = EncoderCacheManager(cache_size=3)
    manager.allocate(first, 0)
    manager.allocate(first, 1)
    assert manager.num_free_slots == 0
    assert manager.check_and_update_cache(second, 0)
    assert manager.num_free_slots == 0
    manager.free(first)
    assert manager.num_freeable_slots == 1
    manager.free(second)
    assert manager.num_freeable_slots == 3


def test_scheduler_budget_and_multiple_items_use_embedding_count():
    request = RequestDouble(
        "r",
        [
            ("a", 0, [True, False, True, False, False]),
            ("b", 5, [False, True, False, False, False]),
        ],
    )
    scheduled, new_tokens, remaining, external = _try(
        _scheduler(cache_size=3), request, 0, 10, 3
    )
    assert scheduled == [0, 1]
    assert new_tokens == 10
    assert remaining == 0
    assert external == []


def test_chunk_overlap_uses_decoder_positions_and_special_only_chunk_skips_encoder():
    request = RequestDouble(
        "r", [("sparse", 0, [False, False, True, True, False])]
    )
    scheduler = _scheduler(cache_size=2)
    scheduled, new_tokens, remaining, _ = _try(scheduler, request, 0, 2, 2)
    assert scheduled == []
    assert new_tokens == 2
    assert remaining == 2
    assert scheduler.encoder_cache_manager.num_free_slots == 2

    scheduled, new_tokens, remaining, _ = _try(scheduler, request, 2, 2, 2)
    assert scheduled == [0]
    assert new_tokens == 2
    assert remaining == 0


def test_preemption_restores_embedding_budget():
    # Exercise Scheduler.schedule's actual priority-preemption branch. The
    # initial budget admits two E=2 items. Request ``a`` is scheduled and then
    # preempted while admitting ``b``. Its E=2 budget must be restored, which
    # admits exactly ``c`` but not ``d``. Restoring P=5 incorrectly admits d;
    # failing to carry the restored budget forward incorrectly rejects c.
    requests = [
        RunningRequestDouble(name, priority)
        for name, priority in zip("abcd", (10, 1, 2, 3), strict=True)
    ]
    kv_cache = KVCacheDouble()
    scheduler = SimpleNamespace(
        running=requests,
        waiting=create_request_queue(SchedulingPolicy.PRIORITY),
        max_num_scheduled_tokens=20,
        max_num_encoder_input_tokens=4,
        max_model_len=64,
        scheduler_config=SimpleNamespace(
            long_prefill_token_threshold=0,
            disable_chunked_mm_input=False,
        ),
        use_eagle=False,
        is_encoder_decoder=False,
        encoder_cache_manager=EncoderCacheManager(cache_size=16),
        kv_cache_manager=kv_cache,
        policy=SchedulingPolicy.PRIORITY,
        num_lookahead_tokens=0,
        lora_config=None,
        max_num_running_reqs=8,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
        use_v2_model_runner=False,
        use_pp=False,
        connector=None,
        ec_connector=None,
        finished_req_ids=set(),
        prev_step_scheduled_req_ids=set(),
        log_stats=False,
        requests={request.request_id: request for request in requests},
        _try_schedule_encoder_inputs=None,
        _preempt_request=None,
        _make_cached_request_data=lambda *_args, **_kwargs: CachedRequestData.make_empty(),
        _update_after_schedule=lambda *_args, **_kwargs: None,
    )
    scheduler._try_schedule_encoder_inputs = lambda *args, **kwargs: (
        Scheduler._try_schedule_encoder_inputs(scheduler, *args, **kwargs)
    )
    scheduler._preempt_request = lambda *args, **kwargs: Scheduler._preempt_request(
        scheduler, *args, **kwargs
    )

    output = Scheduler.schedule(scheduler)
    assert output.preempted_req_ids == {"a"}
    assert output.scheduled_encoder_inputs == {"b": [0], "c": [0]}
    assert set(output.num_scheduled_tokens) == {"b", "c"}
    assert kv_cache.attempts.get("d", 0) == 0
