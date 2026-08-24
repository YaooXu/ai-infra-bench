#!/usr/bin/env python3
"""Verify the exact pre-PR source state without loading candidate extensions."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys


EXPECTED = {
    "CMakeLists.txt": "cc1be76eea3464d3e845cbd766da595aeeaa6c32b9f7b416e1b04cc0a9a53c6a",
    "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x.cu": "25f058e6e97166d76bc21b6255832963e2e868c7289c38b54bac33beec6a3f46",
    "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x_sm89_fp8_dispatch.cuh": "be16bd91a20f2197bd31fd6a6670345d58fc05e4367bd5bc51c1cc54b1faf63a",
    "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh": "1c3f5b3e0eb1fed4565f1c54bea1e7364a1464a8ddf9bffb4942a2ac2d5815c3",
    "tests/utils.py": "02f513915fe29dc9a0b1aa32c0a2d4c60b4af455cfeb5e4caaa9e7eaed2a39b8",
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
    assert not (root / ".git").exists()

    test_utils = (root / "tests/utils.py").read_text()
    assert "compute capability 8.9+" in test_utils
    assert not (root / "tests/v1/determinism/test_cutlass_batch_invariance.py").exists()

    native_files = [
        root / "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x.cu",
        root
        / "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x_sm89_fp8_dispatch.cuh",
        root
        / "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh",
    ]
    assert all("batch_invariant" not in path.read_text() for path in native_files)

    print(
        json.dumps(
            {
                "base_sha": "ea0e501bb18c12b80acc05ff8c7f013db515ba80",
                "exact_file_hashes": True,
                "git_metadata_absent": True,
                "minimum_fp8_compute_capability": [8, 9],
                "new_regression_test_absent": True,
                "pre_pr_batch_invariant_dispatch_absent": True,
                "structural_base_confirmed": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
