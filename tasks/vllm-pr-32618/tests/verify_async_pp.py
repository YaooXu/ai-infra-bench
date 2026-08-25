#!/usr/bin/env python3
"""Focused two-rank NCCL contract for async-scheduling PP sampled tokens."""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

import vllm.v1.worker.gpu_model_runner as runner_module
from vllm.config import ParallelConfig, SchedulerConfig, VllmConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


METHODS = (
    "_pp_broadcast_prev_sampled_token_ids",
    "_pp_receive_prev_sampled_token_ids_to_input_batch",
)


def check_api() -> int:
    # This is a real config construction, matching the upstream regression
    # contract that async scheduling and PP are a permitted combination.
    cfg = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_model_len=8192,
            is_encoder_decoder=False,
            async_scheduling=True,
        ),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=2,
            distributed_executor_backend="mp",
            nnodes=1,
        ),
    )
    assert cfg.scheduler_config.async_scheduling is True

    missing = [name for name in METHODS if not hasattr(GPUModelRunner, name)]
    if missing:
        print(
            "FAIL: base GPUModelRunner has no direct GPU sampled-token "
            f"broadcast/receive contract for async pipeline parallelism: {missing}",
            file=sys.stderr,
        )
        return 1
    print("api_preflight=PASS async_pp_config=true methods=present")
    return 0


def run_rank() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise AssertionError(f"expected two ranks, got {world_size}")

    pp = SimpleNamespace(
        rank=rank,
        world_size=world_size,
        last_rank=world_size - 1,
        is_last_rank=rank == world_size - 1,
        device_group=dist.group.WORLD,
    )
    runner_module.get_pp_group = lambda: pp
    runner = object.__new__(GPUModelRunner)

    if pp.is_last_rank:
        sampled = torch.tensor([[101], [202]], dtype=torch.int32, device="cuda")
        runner._pp_broadcast_prev_sampled_token_ids(sampled)
        assert sampled.cpu().tolist() == [[101], [202]]
    else:
        runner.device = torch.device("cuda", local_rank)
        runner.input_batch = SimpleNamespace(
            num_reqs=2,
            req_ids=["keep", "discard"],
        )
        runner.discard_request_mask = SimpleNamespace(
            np=np.array([False, True], dtype=np.bool_)
        )
        keep_state = SimpleNamespace(output_token_ids=[7])
        discard_state = SimpleNamespace(output_token_ids=[9])
        runner.requests = {"keep": keep_state, "discard": discard_state}

        runner._pp_receive_prev_sampled_token_ids_to_input_batch()

        assert runner.input_batch.prev_sampled_token_ids.cpu().tolist() == [
            [101],
            [202],
        ]
        assert runner.input_batch.prev_req_id_to_index == {"keep": 0}
        assert keep_state.output_token_ids == [7, -1]
        assert discard_state.output_token_ids == [9]
        print(
            "rank0_received tokens=[[101], [202]] mapping={'keep': 0} "
            "keep_output=[7, -1] discard_output=[9]"
        )

    marker = torch.ones((), dtype=torch.int32, device="cuda")
    dist.all_reduce(marker)
    assert marker.item() == world_size
    dist.barrier()
    if rank == 0:
        props = torch.cuda.get_device_properties(local_rank)
        print(
            "PASS: production async-PP methods broadcast sampled tokens over "
            "two-rank NCCL and rebuilt receiver request state"
        )
        print(
            f"rank0_gpu={props.name} capability={props.major}.{props.minor} "
            f"uuid={props.uuid} world_size={world_size}"
        )
    dist.destroy_process_group()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-api", action="store_true")
    args = parser.parse_args()
    if args.check_api:
        return check_api()
    return run_rank()


if __name__ == "__main__":
    raise SystemExit(main())
