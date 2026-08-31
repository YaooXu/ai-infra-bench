from __future__ import annotations

import json
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    OpenAIServingModels,
)
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_encoding,
    render_for_completion,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.tool_parsers import ToolParserManager


MODEL_NAME = "openai/gpt-oss-120b"
TOOL_NAME = "searchIncidentRunbooks"
ARGUMENTS = {"service": "checkout-api", "symptom": "elevated 502s"}
ARGUMENTS_JSON = json.dumps(ARGUMENTS, separators=(",", ":"))
RAW_HARMONY = (
    "<|channel|>analysis<|message|>I should check the incident runbooks."
    "<|end|><|start|>assistant to=searchIncidentRunbooks"
    "<|channel|>comment<|constrain|>json<|message|>"
    f"{ARGUMENTS_JSON}<|call|>"
)


def chat_request_body(stream: bool) -> dict:
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Checkout API is returning elevated 502s. Find the current "
                    "incident runbook before answering."
                ),
            }
        ],
        "stream": stream,
        "tool_choice": "auto",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": "Search the current incident runbooks.",
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
        ],
    }


def responses_request_body(stream: bool) -> dict:
    return {
        "model": MODEL_NAME,
        "input": (
            "Checkout API is returning elevated 502s. Find the current incident "
            "runbook before answering."
        ),
        "stream": stream,
        "store": False,
        "tool_choice": "auto",
        "tools": [
            {
                "type": "function",
                "name": TOOL_NAME,
                "description": "Search the current incident runbooks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},
                        "symptom": {"type": "string"},
                    },
                    "required": ["service", "symptom"],
                },
            }
        ],
    }


def _completion(
    token_ids,
    *,
    request_id,
    finished,
    include_prompt,
    prompt_token_ids,
):
    return RequestOutput(
        request_id=request_id,
        prompt=[],
        prompt_token_ids=prompt_token_ids if include_prompt else None,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=token_ids,
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason="stop" if finished else None,
                stop_reason=200012 if finished else None,
            )
        ],
        finished=finished,
        num_cached_tokens=0 if include_prompt else None,
    )


def _probe_engine():
    engine = MagicMock()
    engine.errored = False
    engine.model_config.max_model_len = 4096
    engine.model_config.get_diff_sampling_param.return_value = {}
    engine.input_processor = MagicMock()
    engine.renderer = MagicMock()
    return engine


def _build_chat_serving():
    engine = _probe_engine()
    models = OpenAIServingModels(
        engine,
        [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)],
    )
    render = OpenAIServingRender(
        model_config=engine.model_config,
        renderer=engine.renderer,
        model_registry=models.registry,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    serving = OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        openai_serving_render=render,
        chat_template=None,
        chat_template_content_format="auto",
        request_logger=None,
    )
    serving.use_harmony = True
    serving.tool_parser = ToolParserManager.get_tool_parser("openai")
    return serving


async def run_chat(request: ChatCompletionRequest) -> dict:
    serving = _build_chat_serving()
    token_ids = get_encoding().encode(RAW_HARMONY, allowed_special="all")
    input_messages, _ = serving.openai_serving_render._make_request_with_harmony(
        request
    )
    prompt_token_ids = render_for_completion(input_messages)

    async def result_generator():
        if request.stream:
            for index, token_id in enumerate(token_ids):
                yield _completion(
                    [token_id],
                    request_id=request.request_id,
                    finished=False,
                    include_prompt=index == 0,
                    prompt_token_ids=prompt_token_ids,
                )
            yield _completion(
                [],
                request_id=request.request_id,
                finished=True,
                include_prompt=False,
                prompt_token_ids=prompt_token_ids,
            )
        else:
            yield _completion(
                token_ids,
                request_id=request.request_id,
                finished=True,
                include_prompt=True,
                prompt_token_ids=prompt_token_ids,
            )

    generator = (
        serving.chat_completion_stream_generator
        if request.stream
        else serving.chat_completion_full_generator
    )
    result = generator(
        request=request,
        result_generator=result_generator(),
        request_id=request.request_id,
        model_name=request.model,
        conversation=[],
        tokenizer=MagicMock(),
        request_metadata=RequestResponseMetadata(
            request_id=request.request_id, model_name=request.model
        ),
    )

    if not request.stream:
        response = await result
        return response.model_dump(mode="json", by_alias=True)

    chunks = []
    async for event in result:
        for line in event.splitlines():
            if line.startswith("data: {"):
                chunks.append(json.loads(line[6:]))
    return {"chunks": chunks}


def _build_responses_serving(stream: bool):
    engine = _probe_engine()
    token_ids = get_encoding().encode(RAW_HARMONY, allowed_special="all")

    async def generate(engine_input, _sampling_params, request_id, **_kwargs):
        if stream:
            for index, token_id in enumerate(token_ids):
                yield RequestOutput(
                    request_id=request_id,
                    prompt=[],
                    prompt_token_ids=(
                        engine_input["prompt_token_ids"] if index == 0 else None
                    ),
                    prompt_logprobs=None,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text="",
                            token_ids=[token_id],
                            cumulative_logprob=0.0,
                            logprobs=None,
                            finish_reason=(
                                "stop" if index == len(token_ids) - 1 else None
                            ),
                            stop_reason=(
                                200012 if index == len(token_ids) - 1 else None
                            ),
                        )
                    ],
                    finished=index == len(token_ids) - 1,
                    num_cached_tokens=0 if index == 0 else None,
                )
        else:
            yield RequestOutput(
                request_id=request_id,
                prompt=[],
                prompt_token_ids=engine_input["prompt_token_ids"],
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="",
                        token_ids=token_ids,
                        cumulative_logprob=0.0,
                        logprobs=None,
                        finish_reason="stop",
                        stop_reason=200012,
                    )
                ],
                finished=True,
                num_cached_tokens=0,
            )

    engine.generate = generate
    models = OpenAIServingModels(
        engine,
        [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)],
    )
    serving = OpenAIServingResponses(
        engine_client=engine,
        models=models,
        openai_serving_render=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    serving.use_harmony = True
    serving.renderer = engine.renderer
    return serving


async def run_responses(request: ResponsesRequest) -> dict:
    serving = _build_responses_serving(bool(request.stream))
    result = await serving.create_responses(request)
    if not request.stream:
        return result.model_dump(mode="json", by_alias=True)

    events = []
    async for event in result:
        events.append(event.model_dump(mode="json", by_alias=True))
    return {"events": events}


def _chat_stream_call(chunks: list[dict]) -> dict:
    calls: dict[int, dict] = {}
    for chunk in chunks:
        for call in chunk["choices"][0]["delta"].get("tool_calls") or []:
            item = calls.setdefault(
                call["index"], {"name": None, "arguments": "", "id": None}
            )
            if call.get("id"):
                item["id"] = call["id"]
            function = call.get("function") or {}
            if function.get("name"):
                item["name"] = function["name"]
            item["arguments"] += function.get("arguments") or ""
    assert len(calls) == 1
    return calls[0]


def assert_contract(results: dict) -> None:
    chat_full = results["chat_nonstream"]
    chat_calls = chat_full["choices"][0]["message"].get("tool_calls") or []
    assert len(chat_calls) == 1
    assert chat_calls[0]["function"]["name"] == TOOL_NAME
    assert json.loads(chat_calls[0]["function"]["arguments"]) == ARGUMENTS

    chat_stream_call = _chat_stream_call(results["chat_stream"]["chunks"])
    assert chat_stream_call["id"]
    assert chat_stream_call["name"] == TOOL_NAME
    assert json.loads(chat_stream_call["arguments"]) == ARGUMENTS

    response_full = results["responses_nonstream"]
    response_calls = [
        item for item in response_full["output"] if item["type"] == "function_call"
    ]
    assert len(response_calls) == 1
    assert response_calls[0]["name"] == TOOL_NAME
    assert json.loads(response_calls[0]["arguments"]) == ARGUMENTS
    assert not any(item["type"] == "mcp_call" for item in response_full["output"])

    events = results["responses_stream"]["events"]
    added = [
        event
        for event in events
        if event["type"] == "response.output_item.added"
        and event["item"]["type"] == "function_call"
    ]
    assert len(added) == 1
    assert added[0]["item"]["name"] == TOOL_NAME
    argument_text = "".join(
        event["delta"]
        for event in events
        if event["type"] == "response.function_call_arguments.delta"
    )
    assert json.loads(argument_text) == ARGUMENTS
    done = [
        event
        for event in events
        if event["type"] == "response.function_call_arguments.done"
    ]
    assert len(done) == 1
    assert done[0]["name"] == TOOL_NAME
    assert json.loads(done[0]["arguments"]) == ARGUMENTS
    completed = next(event for event in events if event["type"] == "response.completed")
    final_calls = [
        item
        for item in completed["response"]["output"]
        if item["type"] == "function_call"
    ]
    assert len(final_calls) == 1
    assert final_calls[0]["name"] == TOOL_NAME


def summarize(results: dict) -> dict:
    chat_full_calls = (
        results["chat_nonstream"]["choices"][0]["message"].get("tool_calls") or []
    )
    chat_stream_calls = [
        call
        for chunk in results["chat_stream"]["chunks"]
        for call in (chunk["choices"][0]["delta"].get("tool_calls") or [])
    ]
    response_full_types = [
        item["type"] for item in results["responses_nonstream"]["output"]
    ]
    response_stream_types = [
        event["type"] for event in results["responses_stream"]["events"]
    ]
    return {
        "chat_nonstream_tool_calls": len(chat_full_calls),
        "chat_stream_tool_deltas": len(chat_stream_calls),
        "responses_nonstream_output_types": response_full_types,
        "responses_stream_has_function_delta": (
            "response.function_call_arguments.delta" in response_stream_types
        ),
        "responses_stream_has_function_done": (
            "response.function_call_arguments.done" in response_stream_types
        ),
    }
