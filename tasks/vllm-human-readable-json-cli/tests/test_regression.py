import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.fixture(scope="module")
def parser():
    return EngineArgs.add_cli_args(FlexibleArgumentParser())


def _kv_config(parser, value):
    args = parser.parse_args(
        [
            "--kv-transfer-config",
            '{"kv_connector":"OffloadingConnector",'
            '"kv_connector_extra_config":{"cpu_bytes_to_use":'
            + value
            + '},"kv_role":"kv_both"}',
        ]
    )
    return args.kv_transfer_config


@pytest.mark.parametrize(
    "literal,expected",
    [
        ("80m", 80_000_000),
        ("1k", 1_000),
        ("5m", 5_000_000),
        ("2t", 2_000_000_000_000),
        ("1K", 2**10),
        ("2M", 2 * 2**20),
        ("1G", 2**30),
        ("1.5g", 1_500_000_000),
    ],
)
def test_json_dataclass_accepts_human_readable_numbers(parser, literal, expected):
    config = _kv_config(parser, literal)

    assert config.kv_connector_extra_config["cpu_bytes_to_use"] == expected


def test_json_nested_and_multiple_numbers(parser):
    config = _kv_config(parser, '{"host":1k,"device":2M,"ratio":1.25g}')

    assert config.kv_connector_extra_config["cpu_bytes_to_use"] == {
        "host": 1_000,
        "device": 2 * 2**20,
        "ratio": 1_250_000_000,
    }


def test_quoted_suffix_text_is_not_converted(parser):
    config = _kv_config(parser, '"80m"')

    assert config.kv_connector_extra_config["cpu_bytes_to_use"] == "80m"


@pytest.mark.parametrize(
    "literal,expected",
    [("1k", 1_000), ("1K", 2**10), ("1.5g", 1_500_000_000), ("2M", 2 * 2**20)],
)
def test_dotted_dataclass_override_accepts_human_readable_numbers(
    parser,
    literal,
    expected,
):
    args = parser.parse_args(
        ["--compilation-config.max_cudagraph_capture_size", literal]
    )

    assert args.compilation_config.max_cudagraph_capture_size == expected


@pytest.mark.parametrize("literal", ["1k1", "1000meters", "1.5G"])
def test_malformed_or_partial_tokens_remain_errors(parser, literal):
    with pytest.raises(SystemExit):
        _kv_config(parser, literal)


@pytest.mark.parametrize("value,expected", [("auto", -1), ("-1", -1)])
def test_existing_auto_and_minus_one_parsing_is_unchanged(parser, value, expected):
    args = parser.parse_args(["--max-model-len", value])

    assert args.max_model_len == expected
