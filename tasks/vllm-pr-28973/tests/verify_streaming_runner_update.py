#!/usr/bin/env python3
"""Behavior-only streaming-session continuation contract."""

import json
import random
import string
import sys
import traceback
from types import SimpleNamespace

import torch

sys.path.insert(0, "/workspace/repo")

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.worker.gpu_input_batch import CachedRequestState
import vllm.v1.worker.gpu_model_runner as runner_module
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class PersistentBatchSpy:
    def __init__(self, requests):
        self.requests = dict(requests)
        self.req_id_to_index = {req_id: i for i, req_id in enumerate(requests)}
        self.removed = []
        self.added = []
        self.vocab_size = 32000

    def remove_request(self, req_id):
        self.removed.append(req_id)
        self.req_id_to_index.pop(req_id, None)
        self.requests.pop(req_id, None)

    def add_request(self, request):
        assert request.req_id not in self.req_id_to_index, (
            "request was re-added without removing its stale batch row"
        )
        self.added.append(request.req_id)
        self.req_id_to_index[request.req_id] = (
            max(self.req_id_to_index.values(), default=-1) + 1
        )
        self.requests[request.req_id] = request

    def update_req_spec_token_ids(self, request, scheduled_spec_tokens):
        pass

    def condense(self):
        pass

    def refresh_metadata(self):
        pass


def cached(req_id, prompt, outputs, *, marker=None):
    state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=[] if marker is None else [marker],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
        generator=None,
        block_ids=([4],),
        num_computed_tokens=len(prompt),
        output_token_ids=list(outputs),
    )
    return state


def continuation(
    req_id,
    prompt,
    *,
    marker,
    temperature,
    blocks,
    computed,
    prompt_embeds=None,
    pooling_params=None,
):
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=None if prompt is None else list(prompt),
        prompt_embeds=prompt_embeds,
        mm_features=[marker],
        sampling_params=SamplingParams(temperature=temperature, max_tokens=50),
        pooling_params=pooling_params,
        block_ids=blocks,
        num_computed_tokens=computed,
        lora_request=None,
    )


def scheduler_output(records, scheduled_ids):
    cached_data = CachedRequestData(
        req_ids=[],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[],
        num_computed_tokens=[],
        num_output_tokens=[],
    )
    return SimpleNamespace(
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        num_scheduled_tokens={req_id: 1 for req_id in scheduled_ids},
        scheduled_cached_reqs=cached_data,
        scheduled_new_reqs=list(records),
        scheduled_spec_decode_tokens={},
    )


def make_runner(states):
    runner = object.__new__(GPUModelRunner)
    runner.requests = dict(states)
    runner.num_prompt_logprobs = {}
    runner.encoder_cache = {}
    runner.input_batch = PersistentBatchSpy(states)
    runner.is_pooling_model = False
    runner.device = "cpu"
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.use_async_scheduling = False
    runner._get_valid_sampled_token_count = lambda: None
    runner._may_reorder_batch = lambda output: None
    return runner


def apply_update(runner, records):
    previous = runner_module.get_pp_group
    runner_module.get_pp_group = lambda: SimpleNamespace(is_last_rank=True)
    try:
        scheduled_ids = set(runner.requests) | {record.req_id for record in records}
        GPUModelRunner._update_states(
            runner,
            scheduler_output(records, scheduled_ids),
        )
    finally:
        runner_module.get_pp_group = previous


def random_id(rng, prefix):
    return prefix + "-" + "".join(rng.choice(string.ascii_lowercase) for _ in range(12))


def check_repeated_continuation():
    rng = random.Random(28973)
    req_id = random_id(rng, "session")
    original = cached(req_id, [1, 2, 3], [10])
    runner = make_runner({req_id: original})
    mrope_refreshes = []
    runner._init_mrope_positions = lambda state: mrope_refreshes.append(state.req_id)
    prompts = (
        [1, 2, 3, 10, 4],
        [1, 2, 3, 10, 4, 11, 12],
        [1, 2, 3, 10, 4, 11, 12, 13],
    )
    absorbed_outputs = ([10], [11, 12], [13])
    for step, (prompt, absorbed) in enumerate(zip(prompts, absorbed_outputs), 1):
        runner.uses_mrope = step == 3
        state = runner.requests[req_id]
        state.output_token_ids[:] = list(absorbed)
        marker = object()
        pooling_marker = object()
        blocks = ([4, 5 + step],)
        apply_update(
            runner,
            [continuation(req_id, prompt, marker=marker, temperature=0.1 * step,
                          blocks=blocks, computed=len(prompt) - 1,
                          pooling_params=pooling_marker)],
        )
        assert runner.requests[req_id] is original
        assert runner.input_batch.requests[req_id] is original
        assert original.prompt_token_ids == list(prompt)
        assert original.num_prompt_tokens == len(prompt)
        assert original.output_token_ids == []
        assert original.mm_features == [marker]
        assert original.block_ids is blocks
        assert original.pooling_params is pooling_marker
        assert original.sampling_params.temperature == 0.1 * step
        assert original.num_computed_tokens == len(prompt) - 1
    assert mrope_refreshes == [req_id]
    assert runner.input_batch.removed.count(req_id) == 3
    assert runner.input_batch.added.count(req_id) == 3
    return {"continuations": 3, "mrope_refreshes": 1}


def check_partial_absorption():
    # A continuation where the new prompt absorbs only SOME of the prior
    # output tokens. The prompt grows by one token (the first output token,
    # 10), so a naive "drop the first `tokens_absorbed` outputs" update would
    # leave the un-absorbed tail [11, 12] behind. The contract is that a
    # continuation restarts with an empty output list regardless of how many
    # prior outputs were folded into the new prompt, because every prior
    # output token is now represented inside prompt_token_ids.
    rng = random.Random(28977)
    req_id = random_id(rng, "session-partial")
    original = cached(req_id, [1, 2, 3], [10, 11, 12])
    runner = make_runner({req_id: original})

    new_prompt = [1, 2, 3, 10]
    marker = object()
    blocks = ([13, 14],)
    apply_update(
        runner,
        [continuation(req_id, new_prompt, marker=marker, temperature=0.5,
                      blocks=blocks, computed=len(new_prompt) - 1)],
    )
    assert runner.requests[req_id] is original
    assert runner.input_batch.requests[req_id] is original
    assert original.prompt_token_ids == new_prompt
    assert original.num_prompt_tokens == len(new_prompt)
    # Only token 10 was absorbed (tokens_absorbed == 1), but the whole prior
    # output must be discarded: leaving [11, 12] is incorrect.
    assert original.output_token_ids == [], (
        f"continuation must clear all prior output tokens; "
        f"found leftover {original.output_token_ids!r}"
    )
    assert original.block_ids is blocks
    assert runner.input_batch.removed.count(req_id) == 1
    assert runner.input_batch.added.count(req_id) == 1
    return {"tokens_absorbed": 1, "prior_output_len": 3, "output_cleared": True}


def check_interleaved_sessions():
    rng = random.Random(28974)
    a_id = random_id(rng, "session-a")
    b_id = random_id(rng, "session-b")
    original_a = cached(a_id, [1, 2, 3], [10])
    original_b = cached(b_id, [7, 8], [20, 21])
    runner = make_runner({a_id: original_a, b_id: original_b})

    a_snapshot = list(original_a.prompt_token_ids)
    b_marker = object()
    b_blocks = ([9, 10],)
    apply_update(
        runner,
        [continuation(b_id, [7, 8, 20, 21, 22], marker=b_marker,
                      temperature=0.7, blocks=b_blocks, computed=4)],
    )
    assert runner.requests[b_id] is original_b
    assert original_b.prompt_token_ids == [7, 8, 20, 21, 22]
    assert original_b.output_token_ids == []
    assert original_b.mm_features == [b_marker]
    assert original_b.block_ids is b_blocks
    assert original_a.prompt_token_ids == a_snapshot
    return {"sessions": 2, "unrelated_session_unchanged": True}


def check_prompt_embeddings():
    rng = random.Random(28975)
    req_id = random_id(rng, "session-embeds")
    original = cached(req_id, [7, 8], [20])
    runner = make_runner({req_id: original})

    embeds = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    apply_update(
        runner,
        [
            continuation(
                req_id,
                None,
                marker=object(),
                temperature=0.8,
                blocks=([9, 10, 11],),
                computed=2,
                prompt_embeds=embeds,
            )
        ],
    )
    assert runner.requests[req_id] is original
    assert original.prompt_token_ids is None
    assert original.prompt_embeds is embeds
    assert original.num_prompt_tokens == 3
    assert original.output_token_ids == []
    return {"representation": "prompt_embeddings", "rows": 3}


def check_ordinary_new_request():
    rng = random.Random(28976)
    existing_id = random_id(rng, "existing")
    existing = cached(existing_id, [1, 2], [3])
    runner = make_runner({existing_id: existing})
    snapshot = list(existing.prompt_token_ids)

    new_id = random_id(rng, "ordinary")
    apply_update(
        runner,
        [
            continuation(
                new_id,
                [31, 32, 33],
                marker=object(),
                temperature=0.4,
                blocks=([12],),
                computed=0,
            )
        ],
    )
    assert new_id in runner.requests
    assert runner.input_batch.removed.count(new_id) == 0
    assert runner.input_batch.added.count(new_id) == 1
    assert existing.prompt_token_ids == snapshot
    return {"new_request_added_once": True}


def main():
    checks = {
        "repeated_continuation": check_repeated_continuation,
        "partial_absorption": check_partial_absorption,
        "interleaved_sessions": check_interleaved_sessions,
        "prompt_embeddings": check_prompt_embeddings,
        "ordinary_new_request": check_ordinary_new_request,
    }
    passed = {}
    failures = {}
    for name, check in checks.items():
        try:
            passed[name] = check()
        except Exception as exc:
            failures[name] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
    print(
        json.dumps(
            {
                "contract_device": "cpu",
                "private_helper_names_scored": False,
                "stages": passed,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if failures:
        raise AssertionError(
            f"streaming continuation stages failed: {sorted(failures)}"
        )
    print("STREAMING_CONTINUATION_VERIFIER=PASS")


if __name__ == "__main__":
    main()
