from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser


def test_anyof_object_value_is_json_decoded() -> None:
    parser = object.__new__(Qwen3CoderToolParser)
    schema = {
        "payload": {
            "anyOf": [
                {"type": "object"},
                {"type": "null"},
            ]
        }
    }

    converted = parser._convert_param_value(
        '{"name":"codex","count":2}', "payload", schema, "submit"
    )

    assert converted == {"name": "codex", "count": 2}
    assert isinstance(converted, dict)
