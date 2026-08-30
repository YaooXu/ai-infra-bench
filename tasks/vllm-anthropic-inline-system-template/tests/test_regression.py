import pytest

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat


RESTRICTIVE_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'system' and not loop.first %}"
    "{{- raise_exception('System message must be first') }}"
    "{%- endif %}"
    "{{- message.role }}:{{ message.content }};"
    "{%- endfor %}"
)
PERMISSIVE_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- message.role }}:{{ message.content }};"
    "{%- endfor %}"
)


def _request():
    return AnthropicMessagesRequest(
        model="test-model",
        max_tokens=32,
        system="top|",
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "mid|"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "last|"},
        ],
    )


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        (RESTRICTIVE_TEMPLATE, True),
        (PERMISSIVE_TEMPLATE, False),
        (None, True),
    ],
)
def test_template_capability_is_detected_by_rendering(template, expected):
    assert AnthropicServingMessages._detect_merge_inline_system(template) is expected


def test_conversion_merges_only_when_the_template_requires_it():
    merged = AnthropicServingMessages._convert_anthropic_to_openai_request(
        _request(), merge_inline_system=True
    )
    assert [message["role"] for message in merged.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert merged.messages[0]["content"] == "top|mid|last|"

    preserved = AnthropicServingMessages._convert_anthropic_to_openai_request(
        _request(), merge_inline_system=False
    )
    assert [message["role"] for message in preserved.messages] == [
        "system",
        "user",
        "system",
        "assistant",
        "system",
    ]


@pytest.mark.parametrize(
    ("template", "expected"),
    [(RESTRICTIVE_TEMPLATE, True), (PERMISSIVE_TEMPLATE, False)],
)
def test_service_initialization_records_template_capability(
    monkeypatch, template, expected
):
    monkeypatch.setattr(OpenAIServingChat, "__init__", lambda self, *args, **kwargs: None)
    serving = AnthropicServingMessages(
        None,
        None,
        "assistant",
        openai_serving_render=None,
        request_logger=None,
        chat_template=template,
        chat_template_content_format="auto",
    )
    assert serving._merge_inline_system is expected


@pytest.mark.asyncio
async def test_generation_endpoint_uses_detected_mode():
    serving = AnthropicServingMessages.__new__(AnthropicServingMessages)
    serving._merge_inline_system = True
    captured = []
    sentinel_request = object()

    def convert(request, *, merge_inline_system=False):
        captured.append(merge_inline_system)
        return sentinel_request

    async def create_chat_completion(request, raw_request):
        assert request is sentinel_request
        return object()

    serving._convert_anthropic_to_openai_request = convert
    serving.create_chat_completion = create_chat_completion
    serving.message_stream_converter = lambda _generator: "stream-result"

    result = await AnthropicServingMessages.create_messages(serving, _request())
    assert result == "stream-result"
    assert captured == [True]


@pytest.mark.asyncio
async def test_count_tokens_endpoint_uses_detected_mode():
    serving = AnthropicServingMessages.__new__(AnthropicServingMessages)
    serving._merge_inline_system = True
    captured = []
    sentinel_request = object()

    def convert(request, *, merge_inline_system=False):
        captured.append(merge_inline_system)
        return sentinel_request

    async def render_chat_request(request):
        assert request is sentinel_request
        return None, [{"prompt_token_ids": [1, 2, 3]}]

    serving._convert_anthropic_to_openai_request = convert
    serving.render_chat_request = render_chat_request

    result = await AnthropicServingMessages.count_tokens(serving, _request())
    assert result.input_tokens == 3
    assert captured == [True]
