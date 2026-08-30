import json

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request

from . import test_scheduler as helpers
from .test_scheduler import (
    BLOCK_SIZE,
    _alloc_and_register,
    _allocate_gpu_blocks,
    _flush_old_blocks_to_lru_head,
    make_request,
    make_scheduler,
    make_scheduler_output,
    simulate_load_completion,
    simulate_store_completion,
)


def _use_offline_model_config(tmp_path, monkeypatch):
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
                "max_position_embeddings": 10000,
                "word_embed_proj_dim": 64,
                "do_layer_norm_before": True,
                "torch_dtype": "float16",
            }
        )
    )
    original = helpers.ModelConfig

    def offline_model_config(*args, **kwargs):
        kwargs["model"] = str(model_dir)
        kwargs["skip_tokenizer_init"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(helpers, "ModelConfig", offline_model_config)


def test_reset_waits_for_pending_eager_store(tmp_path, monkeypatch):
    _use_offline_model_config(tmp_path, monkeypatch)
    fixture = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    scheduler = fixture.scheduler
    gpu_pool = fixture.gpu_block_pool
    request = make_request(num_blocks=2)
    kv_blocks = _alloc_and_register(fixture, request, 2)
    scheduler.update_state_after_alloc(request, kv_blocks, num_external_tokens=0)
    block_ids = kv_blocks.get_block_ids()
    output = make_scheduler_output(
        {request.request_id: 2 * BLOCK_SIZE},
        new_reqs={request.request_id: block_ids},
    )
    metadata = scheduler.build_connector_meta(output)
    gpu_pool.free_blocks(gpu_pool.blocks[bid] for bid in block_ids[0])

    assert scheduler.reset() is False
    assert not scheduler._store_event_to_blocks
    assert len(scheduler._abandoned_store_event_to_blocks) == 1
    assert gpu_pool.num_gpu_blocks - gpu_pool.get_num_free_blocks() > 1

    simulate_store_completion(scheduler, metadata.store_event)
    assert not scheduler._abandoned_store_event_to_blocks
    assert gpu_pool.num_gpu_blocks - gpu_pool.get_num_free_blocks() == 1
    assert scheduler.reset() is True


def test_reset_waits_for_pending_lazy_store_and_clears_cpu_cache(tmp_path, monkeypatch):
    _use_offline_model_config(tmp_path, monkeypatch)
    fixture = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=8, lazy=True)
    scheduler = fixture.scheduler
    gpu_pool = fixture.gpu_block_pool
    request = make_request(num_blocks=2)
    gpu_blocks = _allocate_gpu_blocks(gpu_pool, request, 2, group_id=0)
    gpu_pool.free_blocks(gpu_blocks)
    fillers = _flush_old_blocks_to_lru_head(gpu_pool, num_filler_blocks=5)
    metadata = scheduler.build_connector_meta(make_scheduler_output({}))
    gpu_pool.free_blocks(fillers)

    assert scheduler.reset() is False
    assert len(scheduler._abandoned_store_event_to_blocks) == 1
    assert scheduler._cursor is None
    simulate_store_completion(scheduler, metadata.store_event)
    assert scheduler.reset() is True

    later = Request(
        request_id="after-reset",
        prompt_token_ids=request.prompt_token_ids,
        sampling_params=request.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=request._block_hasher,
    )
    hit_tokens, _ = scheduler.get_num_new_matched_tokens(later, 0)
    assert hit_tokens == 0


def test_reset_waits_for_pending_load(tmp_path, monkeypatch):
    _use_offline_model_config(tmp_path, monkeypatch)
    fixture = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    scheduler = fixture.scheduler
    gpu_pool = fixture.gpu_block_pool
    request = make_request(num_blocks=2)
    kv_blocks = _alloc_and_register(fixture, request, 2)
    scheduler.update_state_after_alloc(request, kv_blocks, num_external_tokens=0)
    block_ids = kv_blocks.get_block_ids()
    metadata = scheduler.build_connector_meta(
        make_scheduler_output(
            {request.request_id: 2 * BLOCK_SIZE},
            new_reqs={request.request_id: block_ids},
        )
    )
    simulate_store_completion(scheduler, metadata.store_event)

    loading = Request(
        request_id="load-reset",
        prompt_token_ids=request.prompt_token_ids,
        sampling_params=request.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=request._block_hasher,
    )
    hit_tokens, _ = scheduler.get_num_new_matched_tokens(loading, 0)
    assert hit_tokens > 0
    gpu_blocks = gpu_pool.get_new_blocks(2)
    loading_blocks = KVCacheBlocks(blocks=(gpu_blocks,))
    scheduler.update_state_after_alloc(
        loading, loading_blocks, num_external_tokens=hit_tokens
    )
    loading_ids = loading_blocks.get_block_ids()
    load_metadata = scheduler.build_connector_meta(
        make_scheduler_output(
            {loading.request_id: 1},
            new_reqs={loading.request_id: loading_ids},
        )
    )
    gpu_pool.free_blocks(gpu_pool.blocks[bid] for bid in block_ids[0])
    gpu_pool.free_blocks(gpu_pool.blocks[bid] for bid in loading_ids[0])

    assert scheduler.reset() is False
    assert len(scheduler._abandoned_reqs_to_load) == 1
    assert gpu_pool.num_gpu_blocks - gpu_pool.get_num_free_blocks() > 1
    simulate_load_completion(scheduler, {loading.request_id})
    assert not scheduler._abandoned_reqs_to_load
    assert scheduler.reset() is True
    assert gpu_pool.num_gpu_blocks - gpu_pool.get_num_free_blocks() == 1
    assert load_metadata.load_event >= 0
