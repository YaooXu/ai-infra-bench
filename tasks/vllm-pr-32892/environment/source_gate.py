#!/usr/bin/env python3
"""Lock the exact source state around PR 32892's native kernel."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys


EXPECTED = {
    "CMakeLists.txt": "4b0ea617c2e85e74c753b8a4451f2bc0caeb00b354d832c207dff84fbd38cacb",
    "requirements/cuda.txt": "0b5b8a88c16ef7d371f30b6db61de17b49c251cda4b785213ecfc2b935f9c1c7",
    "csrc/moe/moe_permute_unpermute_op.cu": "dfd8d82095c87a020ec099a31aecb18319c88271c010fdd69952424ece2d2fa1",
    "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.h": "36f006c95cb687fcd557ea5e9cd7038619c6e5a1f10b09fd9ece9282e3c7ff24",
    "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl": "72d01a06ad565226ba680dbbb1fcd20bd3177fb3b9fa4386ebdd98a4d1906e16",
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/app").resolve()
    measured = {name: sha256(root / name) for name in EXPECTED}
    assert measured == EXPECTED, (measured, EXPECTED)

    op_source = (root / "csrc/moe/moe_permute_unpermute_op.cu").read_text()
    kernel_source = (
        root
        / "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl"
    ).read_text()
    assert "moe_permute kernels require at least CUDA 12.0" in op_source
    assert "extern __shared__ int64_t smem_expert_first_token_offset[]" in kernel_source
    assert "aligned_expert_first_token_offset" not in kernel_source

    print(
        json.dumps(
            {
                "base_sha": "dc917cceb877dfd13f98c538c4c96158047d98bd",
                "canonical_git_tree": "89beecb205e031cbb82e2eea9d2cd0f350135b8c",
                "cuda_minimum": "12.0",
                "exact_file_hashes": True,
                "pre_pr_linear_expert_scan_present": True,
                "structural_base_confirmed": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
