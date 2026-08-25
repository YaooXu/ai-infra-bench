from __future__ import annotations

from types import SimpleNamespace

import torch

import vllm.v1.worker.gpu_model_runner as runner_module
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class PipelineRequest:
    request_id = "cpu-e2e"
    has_encoder_inputs = True

    def __init__(self, feature: MultiModalFeatureSpec):
        self.mm_features = [feature]

    def get_num_encoder_tokens(self, input_id: int) -> int:
        return self.mm_features[input_id].mm_position.length

    def get_num_encoder_embeds(self, input_id: int) -> int:
        value = self.mm_features[input_id].mm_position.get_num_embeds
        return int(value() if callable(value) else value)


class CpuBoolBuffer:
    def __init__(self, size: int):
        self.cpu = torch.zeros(size, dtype=torch.bool)

    def copy_to_gpu(self, size: int) -> torch.Tensor:
        return self.cpu[:size]


def test_cpu_sparse_pipeline_uses_e_end_to_end(monkeypatch):
    mask = [False, True, False, True, True]
    position = PlaceholderRange(1, len(mask), torch.tensor(mask))
    feature = MultiModalFeatureSpec(
        data=object(),
        modality="image",
        identifier="cpu-item",
        mm_position=position,
    )
    request = PipelineRequest(feature)

    manager = EncoderCacheManager(cache_size=3)
    scheduler = SimpleNamespace(
        ec_connector=None,
        is_encoder_decoder=False,
        encoder_cache_manager=manager,
        scheduler_config=SimpleNamespace(disable_chunked_mm_input=False),
    )
    scheduled, num_new_tokens, remaining, external = (
        Scheduler._try_schedule_encoder_inputs(scheduler, request, 0, 7, 3)
    )
    assert scheduled == [0]
    assert num_new_tokens == 7
    assert remaining == 0
    assert external == []

    manager.allocate(request, 0)
    assert manager.num_free_slots == 0

    compact = torch.tensor([[11.0, 1.0], [22.0, 2.0], [33.0, 3.0]])
    execute_runner = SimpleNamespace(
        encoder_cache={},
        model=SimpleNamespace(embed_multimodal=lambda **_: [compact]),
        device=torch.device("cpu"),
        pin_memory=False,
        is_multimodal_pruning_enabled=False,
        _batch_mm_kwargs_from_scheduler=lambda _: ([object()], [("cpu-item", position)]),
        maybe_save_ec_to_connector=lambda *_: None,
    )
    monkeypatch.setattr(
        runner_module,
        "group_mm_kwargs_by_modality",
        lambda *_args, **_kwargs: iter([("image", 1, {})]),
    )
    outputs = GPUModelRunner._execute_mm_encoder(execute_runner, SimpleNamespace())
    assert len(outputs) == 1
    assert torch.equal(outputs[0], compact)
    assert execute_runner.encoder_cache["cpu-item"].shape == (3, 2)

    gather_runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=[request.request_id]),
        requests={
            request.request_id: SimpleNamespace(
                num_computed_tokens=0,
                mm_features=[feature],
                mrope_positions=None,
            )
        },
        encoder_cache=execute_runner.encoder_cache,
        is_mm_embed=CpuBoolBuffer(7),
        is_multimodal_pruning_enabled=False,
        uses_mrope=False,
        uses_xdrope=False,
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=7,
        num_scheduled_tokens={request.request_id: 7},
    )
    mm_embeds, is_embed = GPUModelRunner._gather_mm_embeddings(
        gather_runner, scheduler_output
    )
    assert len(mm_embeds) == 1
    assert torch.equal(mm_embeds[0], compact)
    assert is_embed.tolist() == [False, False, True, False, True, True, False]

    decoder = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    original = decoder.clone()
    decoder[is_embed] = torch.cat(mm_embeds)
    assert torch.equal(decoder[is_embed], compact)
    assert torch.equal(decoder[~is_embed], original[~is_embed])
