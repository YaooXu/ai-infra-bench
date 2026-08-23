from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.scheduler import Scheduler


def test_all_false_extracts_no_decoder_embedding_ranges():
    position = PlaceholderRange(9, 4, torch.zeros(4, dtype=torch.bool))
    assert position.extract_embeds_range() == []


class AllFalseRequest:
    request_id = "all-false"
    has_encoder_inputs = True

    def __init__(self):
        self.mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="nothing",
                mm_position=PlaceholderRange(0, 6, torch.zeros(6, dtype=torch.bool)),
            )
        ]

    def get_num_encoder_tokens(self, _index):
        return 6

    def get_num_encoder_embeds(self, _index):
        value = self.mm_features[0].mm_position.get_num_embeds
        return int(value() if callable(value) else value)


def test_all_false_pipeline_allocates_and_executes_nothing():
    manager = EncoderCacheManager(1)
    scheduler = SimpleNamespace(
        ec_connector=None,
        is_encoder_decoder=False,
        encoder_cache_manager=manager,
        scheduler_config=SimpleNamespace(disable_chunked_mm_input=False),
    )
    request = AllFalseRequest()
    scheduled, num_new_tokens, remaining, external = (
        Scheduler._try_schedule_encoder_inputs(scheduler, request, 0, 6, 1)
    )
    assert scheduled == []
    assert external == []
    assert num_new_tokens == 6
    assert remaining == 1
    assert manager.num_free_slots == 1
    assert manager.cached == {}


def test_text_only_regression_has_no_encoder_work():
    request = SimpleNamespace(has_encoder_inputs=False)
    scheduler = SimpleNamespace()
    result = Scheduler._try_schedule_encoder_inputs(scheduler, request, 0, 16, 8)
    assert result == ([], 16, 8, [])


def test_dense_multimodal_regression_schedules_full_item():
    class DenseRequest:
        request_id = "dense"
        has_encoder_inputs = True

        def __init__(self):
            self.mm_features = [
                MultiModalFeatureSpec(
                    data=None,
                    modality="image",
                    identifier="dense-item",
                    mm_position=PlaceholderRange(0, 4, None),
                )
            ]

        def get_num_encoder_tokens(self, _index):
            return 4

        def get_num_encoder_embeds(self, _index):
            return 4

    scheduler = SimpleNamespace(
        ec_connector=None,
        is_encoder_decoder=False,
        encoder_cache_manager=EncoderCacheManager(4),
        scheduler_config=SimpleNamespace(disable_chunked_mm_input=False),
    )
    result = Scheduler._try_schedule_encoder_inputs(
        scheduler, DenseRequest(), 0, 4, 4
    )
    assert result == ([0], 4, 0, [])
