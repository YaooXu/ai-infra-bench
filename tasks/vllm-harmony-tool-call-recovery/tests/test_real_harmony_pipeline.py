#!/usr/bin/env python3
"""Drive one encoded Harmony call through the real API parsing paths."""

from __future__ import annotations

from dataclasses import dataclass, field

from openai.types.responses import ResponseFunctionToolCall
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
from vllm.entrypoints.openai.parser.harmony_utils import parse_output_into_messages
from vllm.entrypoints.openai.responses.harmony import harmony_to_response_output
from vllm.tool_parsers.openai_tool_parser import OpenAIToolParser


@dataclass
class _ParserView:
    messages: list = field(default_factory=list)


def _run_channel(encoding, channel):
    tool_message = (
        Message.from_role_and_content(Role.ASSISTANT, '{"query":"weather Paris"}')
        .with_channel(channel)
        .with_recipient("webSearch")
        .with_content_type("json")
    )
    conversation = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.USER, "Find the weather in Paris."),
            tool_message,
        ]
    )
    token_ids = encoding.render_conversation_for_completion(
        conversation,
        Role.ASSISTANT,
    )
    parsed = parse_output_into_messages(token_ids)
    parsed_tool = next(
        message for message in parsed.messages if message.recipient == "webSearch"
    )

    chat_result = OpenAIToolParser(object()).extract_tool_calls(
        "",
        request=None,
        token_ids=token_ids,
    )
    assert chat_result.tools_called is True
    assert chat_result.tool_calls[0].function.name == "webSearch"

    response_items = harmony_to_response_output(
        parsed_tool,
        frozenset({"webSearch"}),
    )
    assert len(response_items) == 1
    assert isinstance(response_items[0], ResponseFunctionToolCall)
    assert response_items[0].name == "webSearch"

    delta, streamed = extract_harmony_streaming_delta(
        harmony_parser=_ParserView(),
        token_states=[TokenState(channel=channel, recipient="webSearch", text="")],
        prev_recipient=None,
        include_reasoning=False,
    )
    assert streamed is True
    assert delta is not None
    assert delta.tool_calls[0].function.name == "webSearch"
    return {
        "channel": channel,
        "encoded_tokens": len(token_ids),
        "chat_tool_calls": len(chat_result.tool_calls),
        "response_items": len(response_items),
        "streamed": streamed,
    }


def main() -> int:
    try:
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        cases = [_run_channel(encoding, channel) for channel in ("analysis", "comment")]
        print(
            {
                "entrypoint": "encoded Harmony output to Chat and Responses parsers",
                "cases": cases,
                "vocabulary": "o200k_base",
            },
            flush=True,
        )
        return 0
    except Exception as exc:
        lines = str(exc).splitlines()
        print(
            {
                "error": type(exc).__name__,
                "message": lines[0] if lines else "no exception message",
            },
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
