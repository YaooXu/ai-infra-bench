from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedPrepareAndFinalize,
    BatchedTritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel,
)
from vllm.v1.worker.workspace import init_workspace_manager


EXPERTS = 4
TOKENS = 16
HIDDEN = 64
INTERMEDIATE = 128
TOP_K = 2


def make_components():
    prepare_finalize = BatchedPrepareAndFinalize(
        max_num_tokens=TOKENS,
        num_local_experts=EXPERTS,
        num_dispatchers=1,
        rank=0,
    )
    fused_experts = BatchedTritonExperts(
        max_num_tokens=TOKENS,
        num_dispatchers=1,
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    return prepare_finalize, fused_experts


def make_kernel(**kwargs):
    return FusedMoEModularKernel(*make_components(), **kwargs)


def make_inputs():
    torch.manual_seed(30282)
    hidden_states = torch.randn(
        TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16
    )
    w1 = torch.randn(
        EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.bfloat16
    )
    w2 = torch.randn(
        EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", dtype=torch.bfloat16
    )
    router_logits = torch.randn(TOKENS, EXPERTS, device="cuda")
    topk_weights, topk_ids = torch.topk(router_logits, TOP_K, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    return hidden_states, w1, w2, topk_weights, topk_ids


def run_cuda(kernel, inputs):
    hidden_states, w1, w2, topk_weights, topk_ids = inputs
    output = kernel(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        global_num_experts=EXPERTS,
    )
    torch.cuda.synchronize()
    if output.device.type != "cuda" or output.shape != hidden_states.shape:
        raise AssertionError("modular MoE did not return the expected CUDA tensor")
    if not torch.isfinite(output).all() or output.abs().sum().item() == 0:
        raise AssertionError("modular MoE produced an invalid output")
    return output


def parallel_config(dp_size: int, use_ep: bool) -> FusedMoEParallelConfig:
    ep_size = dp_size if use_ep else 1
    return FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=dp_size,
        ep_size=ep_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        use_ep=use_ep,
        all2all_backend="allgather_reducescatter",
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required for this Dev workload")

    init_workspace_manager(torch.device("cuda:0"))
    inputs = make_inputs()

    # The no-config compatibility entry must remain usable, without consulting
    # a process-global vLLM configuration. Execute the real batched Triton MoE.
    compatibility_kernel = make_kernel()
    if compatibility_kernel.is_dp_ep:
        raise AssertionError("no-config compatibility entry was treated as DP+EP")
    compatibility_output = run_cuda(compatibility_kernel, inputs)
    print("CUDA_TRITON_COMPAT_OK", float(compatibility_output.float().norm()))

    ordinary_config = parallel_config(dp_size=1, use_ep=False)
    try:
        ordinary_kernel = make_kernel(moe_parallel_config=ordinary_config)
    except TypeError as exc:
        print(
            "FAIL: modular kernel does not accept explicit MoE parallel config "
            f"after CUDA Triton execution ({exc})"
        )
        raise SystemExit(1)

    if ordinary_kernel.moe_parallel_config is not ordinary_config:
        raise AssertionError("explicit MoE parallel config identity was not preserved")
    if ordinary_kernel.is_dp_ep:
        raise AssertionError("ordinary single-rank config was treated as DP+EP")
    ordinary_output = run_cuda(ordinary_kernel, inputs)
    torch.testing.assert_close(ordinary_output, compatibility_output)

    dp_ep_config = parallel_config(dp_size=2, use_ep=True)
    dp_ep_kernel = make_kernel(moe_parallel_config=dp_ep_config)
    if dp_ep_kernel.moe_parallel_config is not dp_ep_config:
        raise AssertionError("DP+EP config identity was not preserved")
    if not dp_ep_kernel.is_dp_ep:
        raise AssertionError("explicit DP+EP config was not recognized")

    old_parallel_config = SimpleNamespace(
        data_parallel_size=1,
        enable_expert_parallel=False,
    )
    try:
        make_kernel(parallel_config=old_parallel_config)
    except TypeError:
        pass
    else:
        raise AssertionError("legacy ParallelConfig keyword remained accepted")

    print("PASS: CUDA Triton MoE and parallel-config contract are correct")


if __name__ == "__main__":
    main()
