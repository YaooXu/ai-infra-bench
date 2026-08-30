from vllm.v1.kv_offload.file_mapper import FileMapper


def _mapper(*, parallel_agnostic: bool, tp_size: int = 1, rank: int = 0):
    return FileMapper(
        root_dir="/tmp/persistent-kv",
        model_name="layout-compatible-model",
        tokens_per_hash=16,
        blocks_per_file=1,
        tp_size=tp_size,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        rank=rank,
        dtype="float16",
        kv_cache_groups=[{"tokens_per_block": 16, "layer_names": ["layer0"]}],
        parallel_agnostic=parallel_agnostic,
    )


def test_layout_compatibility_separates_persistent_namespaces():
    legacy_agnostic = _mapper(parallel_agnostic=True)
    layout_specific = _mapper(parallel_agnostic=False)

    assert legacy_agnostic.base_path != layout_specific.base_path
    assert "parallel_agnostic" not in legacy_agnostic.get_run_config()
    assert layout_specific.get_run_config()["parallel_agnostic"] is False


def test_genuinely_agnostic_parallel_configs_still_share_namespace():
    single_rank = _mapper(parallel_agnostic=True, tp_size=1, rank=0)
    tensor_parallel = _mapper(parallel_agnostic=True, tp_size=4, rank=3)

    assert single_rank.base_path == tensor_parallel.base_path
    assert single_rank.rank == tensor_parallel.rank == 0
