from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch

import vllm.multimodal.registry as registry_module
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.example_connector import ECExampleConnector
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.multimodal.registry import MultiModalRegistry


def test_profiler_and_dummy_cache_use_embedding_count(monkeypatch):
    calls: list[str] = []

    class ProfilerDouble:
        def __init__(self, _processor):
            pass

        def get_mm_limits(self):
            return {"image": 1}

        def get_mm_max_tokens(self, _seq_len, _limits):
            calls.append("embedding")
            return {"image": 3}

        def get_mm_max_contiguous_tokens(self, _seq_len, _limits):
            calls.append("placeholder")
            return {"image": 9}

    monkeypatch.setattr(registry_module, "MultiModalProfiler", ProfilerDouble)
    registry = MultiModalRegistry.__new__(MultiModalRegistry)
    registry.create_processor = lambda *_args, **_kwargs: object()
    model_config = SimpleNamespace(is_multimodal_model=True, max_model_len=128)
    result = MultiModalRegistry.get_max_tokens_per_item_by_modality(
        registry, model_config
    )
    assert result == {"image": 3}
    assert calls == ["embedding"]


class ConnectorRequest:
    def __init__(self):
        mask = torch.tensor([False, True, False, True, True])
        self.mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="sparse",
                mm_position=PlaceholderRange(0, 5, mask),
            )
        ]

    def get_num_encoder_tokens(self, _index):
        return 5

    def get_num_encoder_embeds(self, _index):
        value = self.mm_features[0].mm_position.get_num_embeds
        return int(value() if callable(value) else value)


def test_external_connector_accounts_and_transfers_embedding_rows(tmp_path):
    config = Mock()
    config.ec_transfer_config.get_from_extra_config.return_value = str(tmp_path)
    config.ec_transfer_config.is_ec_producer = False
    connector = ECExampleConnector(config, ECConnectorRole.SCHEDULER)
    connector.update_state_after_alloc(ConnectorRequest(), 0)
    assert connector._mm_datas_need_loads == {"sparse": 3}
