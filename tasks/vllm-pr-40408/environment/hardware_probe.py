#!/usr/bin/env python3
"""Report whether the visible GPU can execute this PR's FP8 Cutlass path."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import socket

import torch
import vllm
from vllm.platforms import current_platform


def main() -> None:
    assert torch.cuda.is_available(), "the eligibility probe requires a visible CUDA GPU"
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    minimum = (8, 9)
    source_root = pathlib.Path("/app").resolve()
    native_spec = importlib.util.find_spec("vllm._C")
    assert native_spec and native_spec.origin
    native_path = pathlib.Path(native_spec.origin).resolve()
    python_path = pathlib.Path(vllm.__file__).resolve()

    # A runtime started with --network none returns ENETUNREACH (101).
    sock = socket.socket()
    sock.settimeout(1.0)
    try:
        network_connect_ex = sock.connect_ex(("1.1.1.1", 443))
    finally:
        sock.close()

    blocked = capability < minimum or not current_platform.supports_fp8()
    assert python_path.is_relative_to(pathlib.Path("/usr/local/lib/python3.12/dist-packages"))
    assert not native_path.is_relative_to(source_root)

    print(
        json.dumps(
            {
                "candidate_native_exact_base_bound": False,
                "candidate_source": str(source_root),
                "compute_capability": list(capability),
                "container_device_index": device,
                "cuda_available": True,
                "hardware_blocked": blocked,
                "minimum_required_compute_capability": list(minimum),
                "name": properties.name,
                "native_extension": str(native_path),
                "native_extension_role": "release-image hardware probe only; not candidate code",
                "network_connect_ex": network_connect_ex,
                "python_source_role": "release-image hardware probe only; not candidate code",
                "release_python_source": str(python_path),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "uuid": str(properties.uuid),
                "vllm": vllm.__version__,
                "vllm_supports_fp8": current_platform.supports_fp8(),
            },
            sort_keys=True,
        )
    )

    assert network_connect_ex != 0
    assert blocked, "eligible hardware found: rebuild exact base native extensions there"


if __name__ == "__main__":
    main()
