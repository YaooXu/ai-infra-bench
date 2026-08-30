from types import SimpleNamespace
from unittest.mock import call, patch

import huggingface_hub
import pytest

from vllm.config import ModelConfig
from vllm.engine.arg_utils import EngineArgs


@pytest.mark.parametrize("uri", ["s3://bucket/model", "gs://bucket/model", "az://container/model"])
def test_offline_cloud_model_bypasses_huggingface(uri, monkeypatch):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    with patch("vllm.engine.arg_utils.get_model_path", side_effect=AssertionError("HF called")) as resolve:
        args = EngineArgs(model=uri)

    assert args.model == uri
    resolve.assert_not_called()


@pytest.mark.parametrize("uri", ["s3://bucket/tokenizer", "gs://bucket/tokenizer", "az://container/tokenizer"])
def test_offline_cloud_tokenizer_bypasses_huggingface(uri, monkeypatch):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)

    def resolve(value, revision):
        if value == uri:
            raise AssertionError("cloud tokenizer sent to Hugging Face")
        return "/cached/model"

    with patch("vllm.engine.arg_utils.get_model_path", side_effect=resolve) as get_path:
        args = EngineArgs(model="org/model", tokenizer=uri)

    assert args.model == "/cached/model"
    assert args.tokenizer == uri
    assert get_path.call_args_list == [call("org/model", None)]


@pytest.mark.parametrize("model", ["org/model", "/models/local"])
def test_offline_regular_model_paths_still_resolve(model, monkeypatch):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    with patch("vllm.engine.arg_utils.get_model_path", return_value="/cached/result") as resolve:
        args = EngineArgs(model=model)

    assert args.model == "/cached/result"
    resolve.assert_called_once_with(model, None)


@pytest.mark.parametrize(
    "model_uri,tokenizer_uri",
    [
        ("s3://bucket/model/", "s3://bucket/tokenizer/"),
        ("gs://bucket/model/", "az://container/tokenizer/"),
        ("az://container/model/", "gs://bucket/tokenizer/"),
    ],
)
def test_distinct_cloud_model_and_tokenizer_are_pulled_independently(
    model_uri,
    tokenizer_uri,
):
    config = SimpleNamespace(model_weights=None)
    with patch(
        "vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files"
    ) as pull:
        ModelConfig.maybe_pull_model_tokenizer_for_runai(
            config,
            model_uri,
            tokenizer_uri,
        )

    assert pull.call_count == 2
    assert pull.call_args_list[0].args[0] == model_uri
    assert pull.call_args_list[1].args[0] == tokenizer_uri
    assert config.model != config.tokenizer


@pytest.mark.parametrize("uri", ["s3://bucket/shared/", "gs://bucket/shared/", "az://container/shared/"])
def test_shared_cloud_model_and_tokenizer_use_one_cache_directory(uri):
    config = SimpleNamespace(model_weights=None)
    with patch(
        "vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files"
    ) as pull:
        ModelConfig.maybe_pull_model_tokenizer_for_runai(config, uri, uri)

    assert pull.call_count == 2
    assert [entry.args[0] for entry in pull.call_args_list] == [uri, uri]
    assert config.model == config.tokenizer


def test_offline_none_tokenizer_does_not_add_resolution(monkeypatch):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    with patch("vllm.engine.arg_utils.get_model_path", return_value="/cached/model") as resolve:
        args = EngineArgs(model="org/model", tokenizer=None)

    assert args.model == "/cached/model"
    assert args.tokenizer is None
    resolve.assert_called_once()
