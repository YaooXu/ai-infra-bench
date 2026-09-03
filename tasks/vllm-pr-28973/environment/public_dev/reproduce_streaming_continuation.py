#!/usr/bin/env python3
"""Public symptom reproduction for repeated streaming continuation."""

from types import SimpleNamespace

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.worker.gpu_input_batch import CachedRequestState
import vllm.v1.worker.gpu_model_runner as runner_module
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class PersistentBatch:
    def __init__(self, request):
        self.requests = {request.req_id: request}
        self.req_id_to_index = {request.req_id: 0}
        self.vocab_size = 32000

    def remove_request(self, req_id):
        self.requests.pop(req_id, None)
        self.req_id_to_index.pop(req_id, None)

    def add_request(self, request):
        if request.req_id in self.req_id_to_index:
            raise AssertionError("a stale persistent-batch row survived")
        self.requests[request.req_id] = request
        self.req_id_to_index[request.req_id] = 0

    def update_req_spec_token_ids(self, request, scheduled_spec_tokens):
        pass

    def condense(self):
        pass

    def refresh_metadata(self):
        pass


def scheduler_output(request):
    cached = CachedRequestData(
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
        num_scheduled_tokens={request.req_id: 1},
        scheduled_cached_reqs=cached,
        scheduled_new_reqs=[request],
        scheduled_spec_decode_tokens={},
    )


def continuation(req_id, prompt, marker, block_ids):
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        prompt_embeds=None,
        mm_features=[marker],
        sampling_params=SamplingParams(temperature=0.2, max_tokens=16),
        pooling_params=None,
        block_ids=block_ids,
        num_computed_tokens=len(prompt) - 1,
        lora_request=None,
    )


def main():
    req_id = "public-streaming-session"
    state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=16),
        generator=None,
        block_ids=([4],),
        num_computed_tokens=3,
        output_token_ids=[10],
    )
    original = state
    runner = object.__new__(GPUModelRunner)
    runner.requests = {req_id: state}
    runner.num_prompt_logprobs = {}
    runner.encoder_cache = {}
    runner.input_batch = PersistentBatch(state)
    runner.is_pooling_model = False
    runner.device = "cpu"
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.use_async_scheduling = False
    runner._get_valid_sampled_token_count = lambda: None
    runner._may_reorder_batch = lambda output: None

    prompts = (
        [1, 2, 3, 10, 4],
        [1, 2, 3, 10, 4, 11, 5],
        [1, 2, 3, 10, 4, 11, 5, 12, 6],
    )
    absorbed = ([10], [11], [12])
    previous_pp_group = runner_module.get_pp_group
    runner_module.get_pp_group = lambda: SimpleNamespace(is_last_rank=True)
    try:
        for step, (prompt, output_tokens) in enumerate(
            zip(prompts, absorbed, strict=True), 1
        ):
            original.output_token_ids[:] = output_tokens
            marker = object()
            blocks = ([4, 4 + step],)
            update = continuation(req_id, prompt, marker, blocks)
            GPUModelRunner._update_states(runner, scheduler_output(update))
            assert runner.requests[req_id] is original
            assert runner.input_batch.requests[req_id] is original
            assert original.prompt_token_ids == prompt
            assert original.output_token_ids == []
            assert original.mm_features == [marker]
            assert original.block_ids is blocks
    finally:
        runner_module.get_pp_group = previous_pp_group

    print("STREAMING_CONTINUATION_REPRO=PASS continuations=3")


if __name__ == "__main__":
    main()
