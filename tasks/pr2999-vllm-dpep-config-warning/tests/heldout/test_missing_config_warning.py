import logging
from types import SimpleNamespace

import vllm.config.vllm as vllm_config_mod


def test_direct_unset_config_access_still_warns(monkeypatch, caplog):
    monkeypatch.setattr(vllm_config_mod, "_current_vllm_config", None)
    monkeypatch.setattr(
        vllm_config_mod,
        "VllmConfig",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                enable_expert_parallel=True,
            )
        ),
    )

    with caplog.at_level(logging.WARNING):
        config = vllm_config_mod.get_current_vllm_config()

    assert config is not None
    assert "Current vLLM config is not set." in caplog.text
