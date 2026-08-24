#!/usr/bin/env python3
"""Real accelerator eligibility gate for the DeepGEMM SM100 task."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import torch
import vllm
import vllm._C as native_vllm


def main() -> int:
    if not torch.cuda.is_available():
        print("BLOCKED: CUDA is unavailable; PR 20087 requires SM100/B200.")
        return 2

    probe = torch.zeros(16, device="cuda")
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    deep_gemm_spec = importlib.util.find_spec("deep_gemm")

    root = pathlib.Path("/workspace/repo").resolve()
    source = pathlib.Path(vllm.__file__).resolve()
    native = pathlib.Path(native_vllm.__file__).resolve()
    assert source.is_relative_to(root), source
    assert native.is_relative_to(root), native
    assert probe.device.type == "cuda" and probe.sum().item() == 0

    print(f"device={props.name}")
    print(f"capability={capability[0]}.{capability[1]}")
    print(f"uuid={props.uuid}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"candidate_source={source}")
    print(f"candidate_native={native}")
    print(f"deep_gemm_available={deep_gemm_spec is not None}")
    print(f"uid={os.getuid()} target_device={os.environ.get('VLLM_TARGET_DEVICE')}")

    if capability[0] != 10:
        print(
            "BLOCKED: observed a real CUDA device, but PR 20087 targets "
            "DeepGEMM v2 block-FP8 kernels on SM100/B200; executing an SM90 "
            "or fallback kernel would not validate the requested feature."
        )
        return 2

    if deep_gemm_spec is None:
        print(
            "BLOCKED: SM100 is present but the separately pinned DeepGEMM v2 "
            "dependency is not installed."
        )
        return 3

    print("ELIGIBLE: SM100 and DeepGEMM are available for the full verifier.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
