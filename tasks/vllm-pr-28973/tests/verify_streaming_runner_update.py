#!/usr/bin/env python3
import sys
from types import SimpleNamespace

sys.path.insert(0, "/workspace/repo")

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.worker.gpu_input_batch import CachedRequestState
import vllm.v1.worker.gpu_model_runner as runner_module
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class PersistentBatchSpy:
    def __init__(self, req_id, request):
        self.req_id_to_index = {req_id: 0}
        self.requests = {req_id: request}
        self.removed = []
        self.vocab_size = 32000

    def remove_request(self, req_id):
        self.removed.append(req_id)
        self.req_id_to_index.pop(req_id, None)
        self.requests.pop(req_id, None)

    def add_request(self, request):
        assert request.req_id not in self.req_id_to_index, (
            "continuation was re-added without removing its stale batch row"
        )
        self.req_id_to_index[request.req_id] = 0
        self.requests[request.req_id] = request

    def update_req_spec_token_ids(self, request, scheduled_spec_tokens):
        pass

    def condense(self):
        pass

    def refresh_metadata(self):
        pass


def main():
    print("contract_device=cpu")

    req_id = "streaming-session"
    original_params = SamplingParams(temperature=0.0, max_tokens=8)
    existing = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=original_params,
        generator=None,
        block_ids=([4],),
        num_computed_tokens=4,
        output_token_ids=[10, 11],
    )
    batch = PersistentBatchSpy(req_id, existing)

    runner = object.__new__(GPUModelRunner)
    runner.requests = {req_id: existing}
    runner.num_prompt_logprobs = {}
    runner.encoder_cache = {}
    runner.input_batch = batch
    runner.is_pooling_model = False
    runner.device = "cpu"
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.use_async_scheduling = False
    runner._get_valid_sampled_token_count = lambda: None
    runner._may_reorder_batch = lambda scheduler_output: None

    mm_marker = object()
    new_params = SamplingParams(temperature=0.8, max_tokens=50)
    new_blocks = ([4, 5],)
    continuation = NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3, 10, 4, 5],
        prompt_embeds=None,
        mm_features=[mm_marker],
        sampling_params=new_params,
        pooling_params=None,
        block_ids=new_blocks,
        num_computed_tokens=4,
        lora_request=None,
    )
    cached = CachedRequestData(
        req_ids=[],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[],
        num_computed_tokens=[],
        num_output_tokens=[],
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        num_scheduled_tokens={req_id: 2},
        scheduled_cached_reqs=cached,
        scheduled_new_reqs=[continuation],
        scheduled_spec_decode_tokens={},
    )

    previous_get_pp_group = runner_module.get_pp_group
    runner_module.get_pp_group = lambda: SimpleNamespace(is_last_rank=True)
    try:
        GPUModelRunner._update_states(runner, scheduler_output)
    except Exception as exc:
        print(
            "FAIL: production runner rejected session continuation:",
            type(exc).__name__,
            exc,
        )
        raise SystemExit(1)
    finally:
        runner_module.get_pp_group = previous_get_pp_group

    assert runner.requests[req_id] is existing
    assert batch.requests[req_id] is existing
    assert batch.removed == [req_id]
    assert existing.prompt_token_ids == [1, 2, 3, 10, 4, 5]
    assert existing.prompt_embeds is None
    assert existing.mm_features == [mm_marker]
    assert existing.sampling_params is new_params
    assert existing.pooling_params is None
    assert existing.block_ids is new_blocks
    assert existing.num_computed_tokens == 4
    assert existing.num_prompt_tokens == 6
    assert existing.output_token_ids == []
    print("PASS: production runner updated the streaming session in place")


if __name__ == "__main__":
    main()
