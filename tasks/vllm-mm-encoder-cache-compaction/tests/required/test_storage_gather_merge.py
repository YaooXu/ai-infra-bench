from __future__ import annotations

from types import SimpleNamespace

import torch

import vllm.v1.worker.gpu_model_runner as runner_module
from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class BoolBuffer:
    def __init__(self, size: int):
        self.cpu = torch.zeros(size, dtype=torch.bool)

    def copy_to_gpu(self, size: int) -> torch.Tensor:
        # Preserve the production-facing method while exercising the same
        # boolean-mask semantics without assigning a physical GPU.
        return self.cpu[:size]


def _feature(identifier: str, offset: int, mask: list[bool]):
    return SimpleNamespace(
        identifier=identifier,
        data=object(),
        mm_position=PlaceholderRange(offset, len(mask), torch.tensor(mask)),
    )


def _gather_runner(features, cache, total_tokens: int, computed: int = 0):
    request_id = "request"
    return SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=[request_id]),
        requests={
            request_id: SimpleNamespace(
                num_computed_tokens=computed,
                mm_features=features,
                mrope_positions=None,
            )
        },
        encoder_cache=cache,
        is_mm_embed=BoolBuffer(total_tokens),
        is_multimodal_pruning_enabled=False,
        uses_mrope=False,
        uses_xdrope=False,
    )


def _scheduler_output(num_tokens: int):
    return SimpleNamespace(
        total_num_scheduled_tokens=num_tokens,
        num_scheduled_tokens={"request": num_tokens},
    )


def test_encoder_cache_physically_stores_only_embedding_rows(monkeypatch):
    mask = [False, True, False, True, True]
    position = PlaceholderRange(0, len(mask), torch.tensor(mask))
    compact = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    runner = SimpleNamespace(
        encoder_cache={},
        model=SimpleNamespace(embed_multimodal=lambda **_: [compact]),
        device=torch.device("cpu"),
        pin_memory=False,
        is_multimodal_pruning_enabled=False,
        _batch_mm_kwargs_from_scheduler=lambda _: ([object()], [("item", position)]),
        maybe_save_ec_to_connector=lambda *_: None,
    )
    monkeypatch.setattr(
        runner_module,
        "group_mm_kwargs_by_modality",
        lambda *_args, **_kwargs: iter([("image", 1, {})]),
    )
    outputs = GPUModelRunner._execute_mm_encoder(runner, SimpleNamespace())
    assert outputs[0].shape[0] == 3
    assert runner.encoder_cache["item"].shape == (3, 4)
    assert torch.equal(runner.encoder_cache["item"], compact)


def test_sparse_compact_rows_merge_into_correct_decoder_positions():
    mask = [False, True, False, True, True]
    feature = _feature("item", 0, mask)
    compact = torch.tensor([[11.0, 1.0], [22.0, 2.0], [33.0, 3.0]])
    runner = _gather_runner([feature], {"item": compact}, len(mask))
    mm_embeds, is_embed = GPUModelRunner._gather_mm_embeddings(
        runner, _scheduler_output(len(mask))
    )
    assert len(mm_embeds) == 1
    assert torch.equal(mm_embeds[0], compact)
    assert is_embed.tolist() == mask

    decoder = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    original = decoder.clone()
    decoder[is_embed] = torch.cat(mm_embeds)
    assert torch.equal(decoder[is_embed], compact)
    assert torch.equal(decoder[~is_embed], original[~is_embed])


def test_chunked_and_multiple_item_merge_preserves_non_embedding_positions():
    first = _feature("a", 0, [False, False, True, True, False])
    second = _feature("b", 5, [True, False, True])
    cache = {
        "a": torch.tensor([[1.0], [2.0]]),
        "b": torch.tensor([[3.0], [4.0]]),
    }
    special_runner = _gather_runner([first, second], cache, 2, computed=0)
    embeds, mask = GPUModelRunner._gather_mm_embeddings(
        special_runner, _scheduler_output(2)
    )
    assert embeds == []
    assert not mask.any()

    full_runner = _gather_runner([first, second], cache, 8, computed=0)
    embeds, mask = GPUModelRunner._gather_mm_embeddings(
        full_runner, _scheduler_output(8)
    )
    assert [item.flatten().tolist() for item in embeds] == [[1.0, 2.0], [3.0, 4.0]]
    assert mask.tolist() == [False, False, True, True, False, True, False, True]
