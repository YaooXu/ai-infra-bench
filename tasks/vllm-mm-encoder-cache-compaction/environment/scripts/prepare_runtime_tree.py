#!/usr/bin/env python3
"""Materialize the candidate tree from the verified base source and its wheel.

The vLLM wheel is compiled from the exact hash-verified source during the same
Docker build.  Only generated/native files listed by the committed runtime
contract and generated distribution metadata are copied back.  Candidate
Python therefore always resolves from the editable ``/app`` source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(wheel: Path, destination: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe wheel path: {name!r}")
        archive.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    wheels = sorted(args.wheel_dir.glob("vllm-*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected one source-built vLLM wheel, found {wheels}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    binding = json.loads(args.source_binding.read_text(encoding="utf-8"))

    if args.output.exists() or args.metadata_output.exists():
        raise FileExistsError("runtime output path already exists")
    shutil.copytree(args.source_root, args.output, symlinks=False)
    args.metadata_output.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="vllm-wheel-unpack-") as temporary:
        unpacked = Path(temporary)
        safe_extract(wheels[0], unpacked)
        wheel_shared_objects = {
            path.relative_to(unpacked).as_posix()
            for path in unpacked.rglob("*.so")
            if path.is_file()
        }
        expected_shared_objects = {
            str(item["path"])
            for item in contract["artifacts"]
            if str(item["path"]).endswith(".so")
        }
        missing_shared_objects = expected_shared_objects - wheel_shared_objects
        if missing_shared_objects:
            raise ValueError(
                "source-built wheel is missing required shared objects: "
                f"{sorted(missing_shared_objects)}"
            )
        # A full source build may produce optional extension modules that this
        # A100 task never imports. They remain in the builder stage and are
        # explicitly recorded; only the committed runtime contract is copied.
        excluded_shared_objects = sorted(
            wheel_shared_objects - expected_shared_objects
        )

        generated = []
        for artifact in contract["artifacts"]:
            relative = Path(str(artifact["path"]))
            source = unpacked / relative
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(f"source-built artifact is absent: {relative}")
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            generated.append(
                {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(destination),
                    "size_bytes": destination.stat().st_size,
                }
            )

        dist_info = sorted(unpacked.glob("vllm-*.dist-info"))
        if len(dist_info) != 1:
            raise ValueError(f"expected one vLLM dist-info directory, found {dist_info}")
        shutil.copytree(dist_info[0], args.metadata_output / dist_info[0].name)

    manifest = {
        "schema_version": "1.0",
        "source": "wheel-built-from-hash-verified-base-archive-in-same-image-build",
        "source_wheel_sha256": sha256_file(wheels[0]),
        "base_commit": binding["base_commit"],
        "native_input_tree_sha256": binding["base_native_input_tree_sha256"],
        "target_cuda_arch": contract["target_cuda_arch"],
        "excluded_builder_shared_objects": excluded_shared_objects,
        "artifacts": generated,
    }
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "wheel": wheels[0].name,
                "generated_artifacts": len(generated),
                "native_input_tree_sha256": binding["base_native_input_tree_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
