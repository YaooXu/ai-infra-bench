from __future__ import annotations

import importlib.util
import json
import platform
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
assert torch.__version__ == "2.10.0+cpu", torch.__version__
assert torch.version.cuda is None
assert not torch.cuda.is_available()

print(
    json.dumps(
        {
            "candidate_source": str(source),
            "native_extension": str(native),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "workload_accelerator": "cpu",
            "torch_cuda_available": torch.cuda.is_available(),
        },
        sort_keys=True,
    )
)
