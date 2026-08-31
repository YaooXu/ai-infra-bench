#!/usr/bin/env python3
from __future__ import annotations

import asyncio

from harmony_api_probe import (
    assert_contract,
    chat_request_body,
    responses_request_body,
    run_chat,
    run_responses,
    summarize,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


async def main_async() -> int:
    results = {
        "chat_nonstream": await run_chat(
            ChatCompletionRequest.model_validate(chat_request_body(False))
        ),
        "chat_stream": await run_chat(
            ChatCompletionRequest.model_validate(chat_request_body(True))
        ),
        "responses_nonstream": await run_responses(
            ResponsesRequest.model_validate(responses_request_body(False))
        ),
        "responses_stream": await run_responses(
            ResponsesRequest.model_validate(responses_request_body(True))
        ),
    }
    print(summarize(results), flush=True)
    assert_contract(results)
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except Exception as exc:
        print(
            {
                "error": type(exc).__name__,
                "message": str(exc).splitlines()[0] if str(exc) else "no message",
            },
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
