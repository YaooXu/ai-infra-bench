import json

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

NUM_SPEC = 5


def _tiny_model(tmp_path):
    model_dir = tmp_path / "tiny-opt"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "hidden_size": 64,
                "ffn_dim": 256,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "vocab_size": 256,
                "max_position_embeddings": 256,
                "word_embed_proj_dim": 64,
                "do_layer_norm_before": True,
                "torch_dtype": "float16",
            }
        )
    )
    return str(model_dir)


def _scheduler_and_request(tmp_path):
    scheduler = create_scheduler(
        model=_tiny_model(tmp_path),
        async_scheduling=True,
        skip_tokenizer_init=True,
        max_model_len=256,
        max_num_batched_tokens=256,
    )
    # The helper's generic NGram config is rejected with async scheduling at
    # this base. The test injects the model-runner frame directly, so size the
    # scheduler's metrics vector to that frame without changing production code.
    scheduler.num_spec_tokens = NUM_SPEC
    request = create_requests(num_requests=1, max_tokens=20)[0]
    request.num_computed_tokens = request.num_tokens
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    request.status = RequestStatus.RUNNING
    return scheduler, request


def _rejected_spec_frame(request_id):
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request_id: NUM_SPEC + 1},
        total_num_scheduled_tokens=NUM_SPEC + 1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={request_id: [10] * NUM_SPEC},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[request_id],
        req_id_to_index={request_id: 0},
        sampled_token_ids=[[999]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    return scheduler_output, model_runner_output


def test_discarded_spec_frame_preserves_resumed_counters(tmp_path):
    scheduler, request = _scheduler_and_request(tmp_path)
    request.num_output_placeholders = 1
    request.async_tokens_to_discard = NUM_SPEC
    computed_before = request.num_computed_tokens
    scheduler_output, model_runner_output = _rejected_spec_frame(request.request_id)

    scheduler.update_from_output(scheduler_output, model_runner_output)

    assert request.num_output_placeholders == 1
    assert request.num_computed_tokens == computed_before
    assert request.async_tokens_to_discard == NUM_SPEC - 1
    assert request.status == RequestStatus.RUNNING


def test_nonstale_spec_frame_keeps_rejection_accounting(tmp_path):
    scheduler, request = _scheduler_and_request(tmp_path)
    request.num_output_placeholders = NUM_SPEC + 1
    request.async_tokens_to_discard = 0
    scheduler_output, model_runner_output = _rejected_spec_frame(request.request_id)

    scheduler.update_from_output(scheduler_output, model_runner_output)

    assert request.num_output_placeholders == 0
    assert request.async_tokens_to_discard == 0
    assert request.status == RequestStatus.RUNNING
