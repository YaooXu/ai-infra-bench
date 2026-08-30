import copy
import json
from functools import partial
from unittest.mock import Mock

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import RequestStatus

from . import utils
from .utils import create_request, create_scheduler, create_vllm_config

MATCHED_TOKENS = 37
PROMPT_TOKENS = 70


def _scheduler_waiting_on_partial_block(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        utils,
        "ModelConfig",
        partial(utils.ModelConfig, skip_tokenizer_init=True),
    )
    scheduler = create_scheduler(
        create_vllm_config(model=str(model_dir), max_model_len=256)
    )
    request = create_request(num_tokens=PROMPT_TOKENS)
    scheduler.add_request(request)

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (
        MATCHED_TOKENS,
        True,
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()
    return scheduler, request, scheduler_output


def test_async_wait_records_exact_partial_block_hit(tmp_path, monkeypatch):
    _scheduler, request, scheduler_output = _scheduler_waiting_on_partial_block(
        tmp_path, monkeypatch
    )

    assert not scheduler_output.scheduled_new_reqs
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == MATCHED_TOKENS


def test_async_completion_does_not_round_hit_to_allocated_blocks(tmp_path, monkeypatch):
    scheduler, request, scheduler_output = _scheduler_waiting_on_partial_block(
        tmp_path, monkeypatch
    )
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    waiting_output = scheduler.schedule()
    model_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_output.kv_connector_output = KVConnectorOutput(
        finished_recving={request.request_id}
    )
    scheduler.update_from_output(waiting_output, model_output)

    runnable_output = scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert runnable_output.scheduled_new_reqs[0].num_computed_tokens == MATCHED_TOKENS
    assert runnable_output.num_scheduled_tokens[request.request_id] == (
        PROMPT_TOKENS - MATCHED_TOKENS
    )
