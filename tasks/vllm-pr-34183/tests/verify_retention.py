#!/usr/bin/env python3
"""Behavioral contract for request lifetime and prefix-cache hashing."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import vllm
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, StreamingUpdate


class LargeFeature:
    def __init__(self, marker: int):
        self.marker = marker
        self.data = bytearray(256 * 1024)


def make_request(index: int, calls: list[tuple[int, int]]):
    feature = LargeFeature(index)

    def block_hasher(request: Request):
        calls.append((index, request.num_tokens))
        # The concrete BlockHash type is irrelevant to Request's lifecycle;
        # unique sentinels let us prove that results are still appended.
        return [(index, request.num_tokens)]

    request = Request(
        request_id=f"heldout-{index}",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        eos_token_id=0,
        mm_features=[feature],
        block_hasher=block_hasher,
        resumable=True,
    )
    return request, feature


def check_hash_lifecycle() -> None:
    calls: list[tuple[int, int]] = []
    request, feature = make_request(100, calls)
    if request.mm_features != [feature]:
        raise AssertionError("multimodal payload was discarded while request is live")
    if calls != [(100, 4)] or len(request.block_hashes) != 1:
        raise AssertionError(f"initial block hashing changed: calls={calls}")

    request.append_output_token_ids([7, 8])
    if calls[-1] != (100, 6) or len(request.block_hashes) != 2:
        raise AssertionError(f"append-token hashing changed: calls={calls}")

    request.num_computed_tokens = request.num_tokens
    fake_scheduler = SimpleNamespace(
        num_waiting_for_streaming_input=0,
        log_stats=False,
    )
    update = StreamingUpdate(
        mm_features=None,
        prompt_token_ids=[9, 10, 11],
        max_tokens=8,
        arrival_time=request.arrival_time + 1,
        sampling_params=request.sampling_params,
    )
    Scheduler._update_request_as_session(fake_scheduler, request, update)
    if request.mm_features != [feature]:
        raise AssertionError("streaming continuation discarded a live payload")
    if calls[-1] != (100, 9) or len(request.block_hashes) != 3:
        raise AssertionError(f"streaming-session hashing changed: calls={calls}")


def check_prompt_release() -> None:
    calls: list[tuple[int, int]] = []
    request_refs = []
    feature_refs = []
    for index in range(11):
        request, feature = make_request(index, calls)
        request.append_output_token_ids(index + 20)
        request_refs.append(weakref.ref(request))
        feature_refs.append(weakref.ref(feature))
    del request, feature

    retained_requests = sum(ref() is not None for ref in request_refs)
    retained_features = sum(ref() is not None for ref in feature_refs)
    if retained_requests or retained_features:
        raise AssertionError(
            "completed state requires cyclic GC: "
            f"requests={retained_requests}/11 payloads={retained_features}/11"
        )
    if len(calls) != 22:
        raise AssertionError(f"hash callbacks were suppressed: {len(calls)}!=22")


def main() -> int:
    repo = Path("/workspace/repo").resolve()
    source = Path(vllm.__file__).resolve()
    if repo not in source.parents:
        raise AssertionError(f"candidate source is not active: {source}")

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        check_hash_lifecycle()
        check_prompt_release()
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()

    print(
        "PASS: requests and multimodal payloads release promptly while initial, "
        "append-token, and streaming-session hashing remain active"
    )
    print(f"candidate_source={source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
