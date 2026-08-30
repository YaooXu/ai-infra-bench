from unittest.mock import patch

from transformers import PretrainedConfig

from vllm.transformers_utils import config as config_module


class FlakyParser:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise FileNotFoundError("config.json temporarily hidden by cache refresh")
        config = PretrainedConfig(architectures=["BertModel"])
        return {}, config


def test_get_config_retries_transient_parser_failure(tmp_path) -> None:
    parser = FlakyParser()
    (tmp_path / "config.json").write_text('{"model_type":"bert"}')
    with (
        patch.object(config_module, "get_config_parser", return_value=parser),
        patch("vllm.transformers_utils.repo_utils.time.sleep", return_value=None),
    ):
        config = config_module.get_config(
            tmp_path,
            trust_remote_code=False,
            config_format="hf",
        )

    assert parser.calls == 2
    assert config.architectures == ["BertModel"]
