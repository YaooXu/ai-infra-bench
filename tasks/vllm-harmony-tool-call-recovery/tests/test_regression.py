import asyncio
import json

import pytest

import harmony_api_probe as probe
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


MODEL = "openai/gpt-oss-120b"
ARGUMENTS = {"service": "checkout-api", "symptom": "elevated 502s"}
ARGUMENTS_JSON = json.dumps(ARGUMENTS, separators=(",", ":"))


def _tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Search approved operational data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "symptom": {"type": "string"},
                },
                "required": ["service", "symptom"],
            },
        },
    }


def _response_tool(name):
    tool = _tool(name)["function"]
    return {"type": "function", **tool}


def _chat_request(stream, declared_names=("webSearch",)):
    return ChatCompletionRequest.model_validate(
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Find the checkout incident runbook.",
                }
            ],
            "stream": stream,
            "tool_choice": "auto",
            "tools": [_tool(name) for name in declared_names],
        }
    )


def _responses_request(stream, declared_names=("webSearch",)):
    return ResponsesRequest.model_validate(
        {
            "model": MODEL,
            "input": "Find the checkout incident runbook.",
            "stream": stream,
            "store": False,
            "tool_choice": "auto",
            "tools": [_response_tool(name) for name in declared_names],
        }
    )


def _configure_output(monkeypatch, recipient, channel):
    raw = (
        "<|channel|>analysis<|message|>I should search the runbooks."
        f"<|end|><|start|>assistant to={recipient}<|channel|>{channel}"
        f"<|constrain|>json<|message|>{ARGUMENTS_JSON}<|call|>"
    )
    monkeypatch.setattr(probe, "ARGUMENTS", ARGUMENTS)
    monkeypatch.setattr(probe, "ARGUMENTS_JSON", ARGUMENTS_JSON)
    monkeypatch.setattr(probe, "RAW_HARMONY", raw)


def _chat_stream_calls(chunks):
    calls = {}
    for chunk in chunks:
        for call in chunk["choices"][0]["delta"].get("tool_calls") or []:
            item = calls.setdefault(
                call["index"], {"id": None, "name": None, "arguments": ""}
            )
            if call.get("id"):
                item["id"] = call["id"]
            function = call.get("function") or {}
            if function.get("name"):
                item["name"] = function["name"]
            item["arguments"] += function.get("arguments") or ""
    return [calls[index] for index in sorted(calls)]


def _assert_chat_call(result, stream, expected_name):
    if stream:
        calls = _chat_stream_calls(result["chunks"])
    else:
        calls = result["choices"][0]["message"].get("tool_calls") or []
        calls = [
            {
                "id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            }
            for call in calls
        ]
    assert len(calls) == 1
    assert calls[0]["id"]
    assert calls[0]["name"] == expected_name
    assert json.loads(calls[0]["arguments"]) == ARGUMENTS


def _responses_completed_output(events):
    completed = next(event for event in events if event["type"] == "response.completed")
    return completed["response"]["output"]


def _assert_responses_call(result, stream, expected_name):
    if stream:
        events = result["events"]
        added = [
            event["item"]
            for event in events
            if event["type"] == "response.output_item.added"
            and event["item"]["type"] == "function_call"
        ]
        assert len(added) == 1
        assert added[0]["id"]
        assert added[0]["call_id"]
        assert added[0]["name"] == expected_name
        arguments = "".join(
            event["delta"]
            for event in events
            if event["type"] == "response.function_call_arguments.delta"
        )
        assert json.loads(arguments) == ARGUMENTS
        done = [
            event
            for event in events
            if event["type"] == "response.function_call_arguments.done"
        ]
        assert len(done) == 1
        assert done[0]["name"] == expected_name
        assert json.loads(done[0]["arguments"]) == ARGUMENTS
        output = _responses_completed_output(events)
    else:
        output = result["output"]
    calls = [item for item in output if item["type"] == "function_call"]
    assert len(calls) == 1
    assert calls[0]["name"] == expected_name
    assert json.loads(calls[0]["arguments"]) == ARGUMENTS
    assert not any(item["type"] == "mcp_call" for item in output)


@pytest.mark.parametrize(
    "recipient,channel,expected_name",
    [
        ("webSearch", "commentary", "webSearch"),
        ("searchIncidentRunbooks", "analysis", "searchIncidentRunbooks"),
        ("lookupTicket", "comment", "lookupTicket"),
        ("ops.search", "analysis", "ops.search"),
        ("functions.webSearch", "commentary", "webSearch"),
        ("functions.searchIncidentRunbooks", "comment", "searchIncidentRunbooks"),
    ],
)
def test_chat_nonstream_declared_function(
    monkeypatch, recipient, channel, expected_name
):
    _configure_output(monkeypatch, recipient, channel)
    declared = recipient.removeprefix("functions.")
    result = asyncio.run(probe.run_chat(_chat_request(False, (declared,))))
    _assert_chat_call(result, False, expected_name)


@pytest.mark.parametrize(
    "recipient,channel,expected_name",
    [
        ("webSearch", "commentary", "webSearch"),
        ("searchIncidentRunbooks", "analysis", "searchIncidentRunbooks"),
        ("lookupTicket", "comment", "lookupTicket"),
        ("ops.search", "comment", "ops.search"),
        ("functions.webSearch", "comment", "webSearch"),
    ],
)
def test_chat_stream_declared_function(monkeypatch, recipient, channel, expected_name):
    _configure_output(monkeypatch, recipient, channel)
    declared = recipient.removeprefix("functions.")
    result = asyncio.run(probe.run_chat(_chat_request(True, (declared,))))
    _assert_chat_call(result, True, expected_name)


@pytest.mark.parametrize("recipient", ["assistant", "python", "container", "browser.search"])
def test_chat_nonstream_preserves_nonfunction(monkeypatch, recipient):
    _configure_output(monkeypatch, recipient, "commentary")
    result = asyncio.run(probe.run_chat(_chat_request(False)))
    calls = result["choices"][0]["message"].get("tool_calls") or []
    assert calls == []


@pytest.mark.parametrize("recipient", ["assistant", "python", "container", "browser.search"])
def test_chat_stream_preserves_nonfunction(monkeypatch, recipient):
    _configure_output(monkeypatch, recipient, "commentary")
    result = asyncio.run(probe.run_chat(_chat_request(True)))
    assert _chat_stream_calls(result["chunks"]) == []


@pytest.mark.parametrize(
    "recipient,channel",
    [
        ("webSearch", "commentary"),
        ("searchIncidentRunbooks", "analysis"),
        ("lookupTicket", "comment"),
        ("ops.search", "comment"),
        ("inventory.lookup", "analysis"),
    ],
)
def test_responses_nonstream_declared_function(monkeypatch, recipient, channel):
    _configure_output(monkeypatch, recipient, channel)
    request = _responses_request(False, (recipient,))
    result = asyncio.run(probe.run_responses(request))
    _assert_responses_call(result, False, recipient)


@pytest.mark.parametrize(
    "recipient,channel",
    [
        ("webSearch", "commentary"),
        ("searchIncidentRunbooks", "analysis"),
        ("lookupTicket", "comment"),
        ("ops.search", "comment"),
    ],
)
def test_responses_stream_declared_function(monkeypatch, recipient, channel):
    _configure_output(monkeypatch, recipient, channel)
    request = _responses_request(True, (recipient,))
    result = asyncio.run(probe.run_responses(request))
    _assert_responses_call(result, True, recipient)


@pytest.mark.parametrize(
    "recipient,server_label,name",
    [
        ("otherServer.search", "otherServer", "search"),
        ("filesystem", "filesystem", "filesystem"),
    ],
)
def test_responses_nonstream_preserves_mcp(
    monkeypatch, recipient, server_label, name
):
    _configure_output(monkeypatch, recipient, "commentary")
    result = asyncio.run(probe.run_responses(_responses_request(False)))
    items = [item for item in result["output"] if item["type"] == "mcp_call"]
    assert len(items) == 1
    assert items[0]["server_label"] == server_label
    assert items[0]["name"] == name
    assert json.loads(items[0]["arguments"]) == ARGUMENTS


@pytest.mark.parametrize(
    "recipient,expected_name",
    [("otherServer.search", "otherServer.search"), ("filesystem", "filesystem")],
)
def test_responses_stream_preserves_mcp(monkeypatch, recipient, expected_name):
    _configure_output(monkeypatch, recipient, "commentary")
    result = asyncio.run(probe.run_responses(_responses_request(True)))
    events = result["events"]
    added = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.added"
        and event["item"]["type"] == "mcp_call"
    ]
    assert len(added) == 1
    assert added[0]["name"] == expected_name
    arguments = "".join(
        event["delta"]
        for event in events
        if event["type"] == "response.mcp_call_arguments.delta"
    )
    assert json.loads(arguments) == ARGUMENTS
    assert not any(
        event["type"] == "response.function_call_arguments.delta"
        for event in events
    )


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_responses_empty_tools_does_not_invent_bare_function(monkeypatch, stream):
    _configure_output(monkeypatch, "webSearch", "commentary")
    result = asyncio.run(probe.run_responses(_responses_request(stream, ())))
    output = (
        _responses_completed_output(result["events"])
        if stream
        else result["output"]
    )
    assert not any(item["type"] == "function_call" for item in output)
    assert any(item["type"] == "mcp_call" for item in output)


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_responses_prefixed_function_preserves_legacy_behavior(monkeypatch, stream):
    _configure_output(monkeypatch, "functions.webSearch", "analysis")
    result = asyncio.run(probe.run_responses(_responses_request(stream, ())))
    _assert_responses_call(result, stream, "webSearch")


@pytest.mark.parametrize(
    "recipient,expected_type",
    [
        ("python", "reasoning"),
        ("container", "reasoning"),
        ("browser.search", "web_search_call"),
    ],
)
def test_responses_nonstream_preserves_builtin(
    monkeypatch, recipient, expected_type
):
    _configure_output(monkeypatch, recipient, "commentary")
    result = asyncio.run(
        probe.run_responses(_responses_request(False, (recipient,)))
    )
    assert any(item["type"] == expected_type for item in result["output"])
    assert not any(item["type"] == "function_call" for item in result["output"])


@pytest.mark.parametrize(
    "recipient,expected_type",
    [("python", "code_interpreter_call"), ("browser.search", "mcp_call")],
)
def test_responses_stream_preserves_builtin(monkeypatch, recipient, expected_type):
    _configure_output(monkeypatch, recipient, "comment")
    result = asyncio.run(
        probe.run_responses(_responses_request(True, (recipient,)))
    )
    added_types = [
        event["item"]["type"]
        for event in result["events"]
        if event["type"] == "response.output_item.added"
    ]
    assert expected_type in added_types
    assert "function_call" not in added_types
