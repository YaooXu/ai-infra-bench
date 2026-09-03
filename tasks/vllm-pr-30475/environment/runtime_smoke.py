from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
import vllm


root = Path("/app")
source = Path(vllm.__file__).resolve()
native_spec = importlib.util.find_spec("vllm._C")
assert source.is_relative_to(root), source
assert native_spec is not None and native_spec.origin is not None
native = Path(native_spec.origin).resolve()
assert native.is_relative_to(root), native
assert torch.__version__.startswith("2.9."), torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert torch.cuda.is_available()

print(
    json.dumps(
        {
            "candidate_source": str(source),
            "native_extension": str(native),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        sort_keys=True,
    )
)
