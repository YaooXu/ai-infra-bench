#!/usr/bin/env python3
"""Small deterministic reproduction for completed-request state retention."""

import gc
import sys
import weakref
from pathlib import Path

import vllm
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request


REQUEST_COUNT = 16


class _LargeFeature:
    def __init__(self, size_mib: int = 2):
        self.data = bytearray(size_mib * 1024 * 1024)


def _create_completed_request(index: int):
    feature = _LargeFeature()
    hash_call_lengths: list[int] = []

    def block_hasher(request: Request):
        hash_call_lengths.append(request.num_tokens)
        return []

    request = Request(
        request_id=f"completed-{index}",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        mm_features=[feature],
        block_hasher=block_hasher,
    )
    if hash_call_lengths != [4]:
        raise AssertionError("request creation did not invoke the block hasher")
    return weakref.ref(request), weakref.ref(feature)


def main() -> int:
    source_path = Path(vllm.__file__).resolve()
    repo_root = Path("/workspace/repo").resolve()
    if repo_root not in source_path.parents:
        print(f"FAIL: vLLM imported outside candidate worktree: {source_path}")
        return 2

    request_refs = []
    feature_refs = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for index in range(REQUEST_COUNT):
            request_ref, feature_ref = _create_completed_request(index)
            request_refs.append(request_ref)
            feature_refs.append(feature_ref)

        retained_requests = sum(ref() is not None for ref in request_refs)
        retained_features = sum(ref() is not None for ref in feature_refs)
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()

    print(f"candidate source: {source_path}")
    print(
        f"completed requests still retained: "
        f"{retained_requests}/{REQUEST_COUNT}"
    )
    print(
        f"multimodal payloads still retained: "
        f"{retained_features}/{REQUEST_COUNT}"
    )
    if retained_requests or retained_features:
        print("FAIL: completed request state was not released promptly")
        return 1
    print("PASS: completed request state was released promptly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
