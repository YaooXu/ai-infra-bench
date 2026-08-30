import json
from dataclasses import dataclass, field

import pytest
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_output_item import McpCall
from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    Role,
    load_harmony_encoding,
)

from vllm.entrypoints.openai.chat_completion.stream_harmony import (
    TokenState,
    extract_harmony_streaming_delta,
)
from vllm.entrypoints.openai.responses.harmony import harmony_to_response_output
from vllm.tool_parsers.openai_tool_parser import OpenAIToolParser


@dataclass
class _PreviousMessage:
    channel: str | None = None
    recipient: str | None = None


@dataclass
class _StreamParser:
    messages: list[_PreviousMessage] = field(default_factory=list)


@pytest.fixture(scope="module")
def harmony_encoding():
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


@pytest.fixture(scope="module")
def chat_parser():
    return OpenAIToolParser(object())


def _chat_result(chat_parser, harmony_encoding, recipient, channel="commentary"):
    message = (
        Message.from_role_and_content(Role.ASSISTANT, '{"query":"weather"}')
        .with_channel(channel)
        .with_recipient(recipient)
        .with_content_type("json")
    )
    token_ids = harmony_encoding.render_conversation_for_completion(
        Conversation.from_messages(
            [Message.from_role_and_content(Role.USER, "Use the declared tool."), message]
        ),
        Role.ASSISTANT,
    )
    return chat_parser.extract_tool_calls("", request=None, token_ids=token_ids)


@pytest.mark.parametrize(
    "recipient,channel,expected_name",
    [
        ("webSearch", "commentary", "webSearch"),
        ("webSearch", "analysis", "webSearch"),
        ("webSearch", "comment", "webSearch"),
        ("math.sum", "analysis", "math.sum"),
        ("functions.webSearch", "commentary", "webSearch"),
    ],
    ids=["bare-commentary", "bare-analysis", "bare-comment", "dotted", "prefixed"],
)
def test_chat_nonstream_exposes_function_call(
    chat_parser,
    harmony_encoding,
    recipient,
    channel,
    expected_name,
):
    result = _chat_result(chat_parser, harmony_encoding, recipient, channel)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == expected_name
    assert json.loads(result.tool_calls[0].function.arguments) == {"query": "weather"}


@pytest.mark.parametrize(
    "recipient",
    ["assistant", "python", "container", "browser.search"],
)
def test_chat_nonstream_does_not_promote_builtin_or_assistant(
    chat_parser,
    harmony_encoding,
    recipient,
):
    result = _chat_result(chat_parser, harmony_encoding, recipient)

    assert result.tools_called is False
    assert result.tool_calls == []


@pytest.mark.parametrize(
    "recipient,channel,expected_name",
    [
        ("webSearch", "commentary", "webSearch"),
        ("webSearch", "analysis", "webSearch"),
        ("webSearch", "comment", "webSearch"),
        ("functions.webSearch", "comment", "webSearch"),
    ],
    ids=["bare-commentary", "bare-analysis", "bare-comment", "prefixed-comment"],
)
def test_chat_streaming_exposes_function_call(recipient, channel, expected_name):
    delta, streamed = extract_harmony_streaming_delta(
        harmony_parser=_StreamParser(),
        token_states=[TokenState(channel=channel, recipient=recipient, text="")],
        prev_recipient=None,
        include_reasoning=False,
    )

    assert streamed is True
    assert delta is not None
    assert len(delta.tool_calls) == 1
    assert delta.tool_calls[0].function.name == expected_name


@pytest.mark.parametrize("recipient", ["assistant", "browser.search"])
def test_chat_streaming_rejects_non_function_recipients(recipient):
    delta, streamed = extract_harmony_streaming_delta(
        harmony_parser=_StreamParser(),
        token_states=[
            TokenState(channel="commentary", recipient=recipient, text="ignored")
        ],
        prev_recipient=None,
        include_reasoning=False,
    )

    assert delta is None
    assert streamed is False


@pytest.mark.parametrize(
    "recipient,channel",
    [
        ("webSearch", "commentary"),
        ("webSearch", "analysis"),
        ("webSearch", "comment"),
        ("math.sum", "comment"),
    ],
    ids=["commentary", "analysis", "comment", "dotted-comment"],
)
def test_responses_api_routes_declared_bare_function(recipient, channel):
    message = (
        Message.from_role_and_content(Role.ASSISTANT, '{"query":"weather"}')
        .with_channel(channel)
        .with_recipient(recipient)
    )

    items = harmony_to_response_output(message, frozenset({recipient}))

    assert len(items) == 1
    assert isinstance(items[0], ResponseFunctionToolCall)
    assert items[0].name == recipient


def test_responses_api_keeps_unknown_bare_recipient_as_mcp():
    message = (
        Message.from_role_and_content(Role.ASSISTANT, "{}")
        .with_channel("commentary")
        .with_recipient("otherServer.search")
    )

    items = harmony_to_response_output(message, frozenset({"webSearch"}))

    assert len(items) == 1
    assert isinstance(items[0], McpCall)


def test_responses_api_empty_tool_list_does_not_invent_function():
    message = (
        Message.from_role_and_content(Role.ASSISTANT, "{}")
        .with_channel("commentary")
        .with_recipient("webSearch")
    )

    items = harmony_to_response_output(message, frozenset())

    assert len(items) == 1
    assert isinstance(items[0], McpCall)


def test_responses_api_prefixed_name_remains_function_with_empty_tool_list():
    message = (
        Message.from_role_and_content(Role.ASSISTANT, "{}")
        .with_channel("analysis")
        .with_recipient("functions.webSearch")
    )

    items = harmony_to_response_output(message, frozenset())

    assert len(items) == 1
    assert isinstance(items[0], ResponseFunctionToolCall)
    assert items[0].name == "webSearch"


@pytest.mark.parametrize("recipient", ["python", "container", "browser.search"])
def test_responses_api_never_promotes_builtin_recipients(recipient):
    message = (
        Message.from_role_and_content(Role.ASSISTANT, "{}")
        .with_channel("comment")
        .with_recipient(recipient)
    )

    items = harmony_to_response_output(message, frozenset({recipient}))

    assert not any(isinstance(item, ResponseFunctionToolCall) for item in items)


def test_responses_api_ignores_non_assistant_recipient():
    message = (
        Message.from_role_and_content(Role.USER, "{}")
        .with_channel("commentary")
        .with_recipient("functions.webSearch")
    )

    assert harmony_to_response_output(message, frozenset({"webSearch"})) == []
