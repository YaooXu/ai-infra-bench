#!/usr/bin/env python3
"""Send an HTTP Responses payload through the real request schema and mapper."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


app = FastAPI()


@app.post("/v1/responses")
def create_response(request: ResponsesRequest):
    params = request.to_sampling_params(
        default_max_tokens=256,
        default_sampling_params={"repetition_penalty": 1.05},
    )
    return {
        "stop": params.stop,
        "seed": params.seed,
        "repetition_penalty": params.repetition_penalty,
        "ignore_eos": params.ignore_eos,
        "extra_args": params.extra_args,
    }


def main() -> int:
    try:
        payload = {
            "model": "test-model",
            "input": "Write a short sentence",
            "stop": ["END", "STOP"],
            "seed": 42,
            "repetition_penalty": 1.2,
            "ignore_eos": True,
            "vllm_xargs": {"custom": "value", "scale": 0.75},
        }
        response = TestClient(app).post("/v1/responses", json=payload)
        assert response.status_code == 200
        assert response.json() == {
            "stop": ["END", "STOP"],
            "seed": 42,
            "repetition_penalty": 1.2,
            "ignore_eos": True,
            "extra_args": {"custom": "value", "scale": 0.75},
        }
        inherited = TestClient(app).post(
            "/v1/responses",
            json={"model": "test-model", "input": "Hello"},
        )
        assert inherited.status_code == 200
        assert inherited.json()["repetition_penalty"] == 1.05
        print(
            {
                "entrypoint": "HTTP POST /v1/responses",
                "status": response.status_code,
                "mapped": response.json(),
                "inherited_repetition_penalty": 1.05,
            },
            flush=True,
        )
        return 0
    except Exception as exc:
        lines = str(exc).splitlines()
        print({"error": type(exc).__name__, "message": lines[0] if lines else "no exception message"}, flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
