#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

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


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def _body(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        return json.loads(self.rfile.read(length))

    def _json(self, payload: dict):
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _sse(self, items, *, responses_api):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for item in items:
            if responses_api:
                self.wfile.write(f"event: {item['type']}\n".encode())
            self.wfile.write(
                f"data: {json.dumps(item, separators=(',', ':'))}\n\n".encode()
            )
            self.wfile.flush()
        if not responses_api:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    def do_POST(self):
        body = self._body()
        if self.path == "/v1/chat/completions":
            request = ChatCompletionRequest.model_validate(body)
            result = asyncio.run(run_chat(request))
            if request.stream:
                self._sse(result["chunks"], responses_api=False)
            else:
                self._json(result)
            return
        if self.path == "/v1/responses":
            request = ResponsesRequest.model_validate(body)
            result = asyncio.run(run_responses(request))
            if request.stream:
                self._sse(result["events"], responses_api=True)
            else:
                self._json(result)
            return
        self.send_error(404)


def _read_sse(url, body):
    items = []
    with httpx.stream("POST", url, json=body, timeout=30) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: {"):
                items.append(json.loads(line[6:]))
    return items


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        chat_full_response = httpx.post(
            f"{root}/v1/chat/completions",
            json=chat_request_body(False),
            timeout=30,
        )
        assert chat_full_response.status_code == 200
        response_full_response = httpx.post(
            f"{root}/v1/responses",
            json=responses_request_body(False),
            timeout=30,
        )
        assert response_full_response.status_code == 200
        results = {
            "chat_nonstream": chat_full_response.json(),
            "chat_stream": {
                "chunks": _read_sse(
                    f"{root}/v1/chat/completions", chat_request_body(True)
                )
            },
            "responses_nonstream": response_full_response.json(),
            "responses_stream": {
                "events": _read_sse(
                    f"{root}/v1/responses", responses_request_body(True)
                )
            },
        }
        print(summarize(results), flush=True)
        assert_contract(results)
        return 0
    except Exception as exc:
        print(
            {
                "error": type(exc).__name__,
                "message": str(exc).splitlines()[0] if str(exc) else "no message",
            },
            flush=True,
        )
        return 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
