from __future__ import annotations

import json
import os
import subprocess

import torch


probe = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,uuid",
        "--format=csv,noheader",
    ],
    check=True,
    capture_output=True,
    text=True,
)
visible = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
assert len(visible) == 1, visible
assert "A100" in visible[0], visible
assert torch.__version__.endswith("+cpu")
assert not torch.cuda.is_available()

print(
    json.dumps(
        {
            "container_visible_gpus": visible,
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "declared_workload_accelerator": "cpu",
        },
        sort_keys=True,
    )
)
