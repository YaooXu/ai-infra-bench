#!/usr/bin/env python3
"""Prepare a clean candidate tree and a separately verified runtime overlay.

The public base archive is materialized once as an immutable pristine tree.  A
different copy is used for compilation.  This program deliberately copies only
the pristine tree into ``runtime_output``; generated wheel files are written to
``overlay_output`` and are applied only after the synthetic source commit has
been created in the final image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


CHUNK_SIZE = 8 * 1024 * 1024
TREE_ALGORITHM = "sha256(path\\0file_sha256\\0size_bytes\\0executable_int\\n)"
FORBIDDEN_DIRECTORY_NAMES = {
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
}
FORBIDDEN_FILE_SUFFIXES = {".pyc", ".pyo"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> Path:
    posix = PurePosixPath(value)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise ValueError(f"unsafe manifest path: {value!r}")
    return Path(*posix.parts)


def safe_extract(wheel: Path, destination: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            safe_relative(name)
        archive.extractall(destination)


def regular_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"source directory is absent: {root}")
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is forbidden in candidate source: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    return files


def load_source_manifest(path: Path) -> tuple[dict, dict[str, dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "2.0":
        raise ValueError("source manifest schema mismatch")
    if manifest.get("tree_hash_algorithm") != TREE_ALGORITHM:
        raise ValueError("source manifest tree algorithm mismatch")
    expected: dict[str, dict] = {}
    for artifact in manifest.get("artifacts", []):
        relative = safe_relative(str(artifact["path"])).as_posix()
        if relative in expected:
            raise ValueError(f"duplicate source path: {relative}")
        if not isinstance(artifact.get("executable"), bool):
            raise ValueError(f"missing executable contract: {relative}")
        expected[relative] = artifact
    if len(expected) != int(manifest.get("artifact_count", -1)):
        raise ValueError("source artifact count mismatch")
    aggregate = hashlib.sha256()
    for relative, artifact in sorted(expected.items()):
        aggregate.update(
            (
                f"{relative}\0{artifact['sha256']}\0{artifact['size_bytes']}\0"
                f"{int(artifact['executable'])}\n"
            ).encode()
        )
    if aggregate.hexdigest() != manifest.get("tree_sha256"):
        raise ValueError("source manifest semantic tree hash mismatch")
    return manifest, expected


def verify_file(path: Path, artifact: dict, relative: str) -> None:
    if path.stat().st_size != int(artifact["size_bytes"]):
        raise ValueError(f"source size mismatch: {relative}")
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"source SHA-256 mismatch: {relative}")
    executable = bool(path.stat().st_mode & stat.S_IXUSR)
    if executable != bool(artifact["executable"]):
        raise ValueError(f"source executable-bit mismatch: {relative}")


def verify_source_tree(
    root: Path, expected: dict[str, dict], *, allow_additions: bool
) -> None:
    actual = regular_files(root)
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"source files are missing: {missing[:20]}")
    if not allow_additions:
        extra = sorted(set(actual) - set(expected))
        if extra:
            raise ValueError(f"pristine source has extra files: {extra[:20]}")
    for relative, artifact in expected.items():
        verify_file(actual[relative], artifact, relative)


def forbidden_entries(root: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(
            part in FORBIDDEN_DIRECTORY_NAMES or part.endswith(".egg-info")
            for part in relative.parts
        ) or (path.is_file() and path.suffix in FORBIDDEN_FILE_SUFFIXES):
            found.append(relative.as_posix())
    return found


def require_distinct_paths(paths: dict[str, Path]) -> dict[str, Path]:
    resolved = {
        name: path.resolve(strict=name in {"pristine_source", "build_source"})
        for name, path in paths.items()
    }
    items = list(resolved.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    f"build path isolation failed: {left_name}={left} "
                    f"overlaps {right_name}={right}"
                )
    return resolved


def normalize_source_modes(root: Path, expected: dict[str, dict]) -> None:
    os.chmod(root, 0o755)
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            os.chmod(path, 0o755)
    for relative, artifact in expected.items():
        os.chmod(
            root / safe_relative(relative),
            0o755 if artifact["executable"] else 0o644,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pristine-source", type=Path, required=True)
    parser.add_argument("--build-source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-binding", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--overlay-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--git-exclude-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    resolved = require_distinct_paths(
        {
            "pristine_source": args.pristine_source,
            "build_source": args.build_source,
            "runtime_output": args.runtime_output,
            "overlay_output": args.overlay_output,
            "metadata_output": args.metadata_output,
        }
    )
    pristine_source = resolved["pristine_source"]
    build_source = resolved["build_source"]
    runtime_output = resolved["runtime_output"]
    overlay_output = resolved["overlay_output"]
    metadata_output = resolved["metadata_output"]

    for output in (runtime_output, overlay_output, metadata_output):
        if output.exists():
            raise FileExistsError(f"runtime preparation output already exists: {output}")
    if not (build_source / "build").is_dir():
        raise ValueError("build_source has no build directory; wheel provenance is unclear")

    source_manifest, expected_source = load_source_manifest(args.source_manifest)
    verify_source_tree(pristine_source, expected_source, allow_additions=False)
    verify_source_tree(build_source, expected_source, allow_additions=True)
    polluted_pristine = forbidden_entries(pristine_source)
    if polluted_pristine:
        raise ValueError(f"pristine source is polluted: {polluted_pristine[:20]}")

    wheels = sorted(args.wheel_dir.glob("vllm-*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected one source-built vLLM wheel, found {wheels}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    binding = json.loads(args.source_binding.read_text(encoding="utf-8"))
    if binding.get("base_commit") != source_manifest.get("base_commit"):
        raise ValueError("source and native binding commits differ")
    if contract.get("base_commit") != source_manifest.get("base_commit"):
        raise ValueError("source and runtime contract commits differ")

    shutil.copytree(pristine_source, runtime_output, symlinks=False)
    normalize_source_modes(runtime_output, expected_source)
    verify_source_tree(runtime_output, expected_source, allow_additions=False)
    polluted_runtime = forbidden_entries(runtime_output)
    if polluted_runtime:
        raise ValueError(f"runtime source is polluted: {polluted_runtime[:20]}")

    overlay_output.mkdir(parents=True)
    metadata_output.mkdir(parents=True)
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
        excluded_shared_objects = sorted(
            wheel_shared_objects - expected_shared_objects
        )

        generated: list[dict[str, object]] = []
        generated_paths: set[str] = set()
        for artifact in contract["artifacts"]:
            relative = safe_relative(str(artifact["path"]))
            relative_text = relative.as_posix()
            if relative_text in generated_paths:
                raise ValueError(f"duplicate runtime artifact: {relative_text}")
            if relative_text in expected_source:
                raise ValueError(
                    f"runtime artifact would overwrite pristine source: {relative_text}"
                )
            source = unpacked / relative
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(
                    f"source-built artifact is absent: {relative_text}"
                )
            destination = overlay_output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            generated.append(
                {
                    "path": relative_text,
                    "sha256": sha256_file(destination),
                    "size_bytes": destination.stat().st_size,
                }
            )
            generated_paths.add(relative_text)

        if len(generated_paths) != int(contract.get("artifact_count", -1)):
            raise ValueError("runtime artifact contract count mismatch")
        dist_info = sorted(unpacked.glob("vllm-*.dist-info"))
        if len(dist_info) != 1:
            raise ValueError(f"expected one vLLM dist-info directory, found {dist_info}")
        shutil.copytree(dist_info[0], metadata_output / dist_info[0].name)

    actual_overlay = regular_files(overlay_output)
    if set(actual_overlay) != generated_paths:
        raise ValueError("runtime overlay contains undeclared files")
    for artifact in generated:
        path = actual_overlay[str(artifact["path"])]
        if path.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"runtime overlay size mismatch: {artifact['path']}")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"runtime overlay hash mismatch: {artifact['path']}")
    polluted_overlay = forbidden_entries(overlay_output)
    if polluted_overlay:
        raise ValueError(f"runtime overlay is polluted: {polluted_overlay[:20]}")

    git_excludes = "".join(f"/{path}\n" for path in sorted(generated_paths))
    args.git_exclude_output.parent.mkdir(parents=True, exist_ok=True)
    args.git_exclude_output.write_text(git_excludes, encoding="utf-8")

    manifest = {
        "schema_version": "2.0",
        "source": "wheel-built-from-distinct-hash-verified-build-copy",
        "source_wheel_sha256": sha256_file(wheels[0]),
        "base_commit": binding["base_commit"],
        "pristine_source_tree_sha256": source_manifest["tree_sha256"],
        "native_input_tree_sha256": binding["base_native_input_tree_sha256"],
        "target_cuda_arch": contract["target_cuda_arch"],
        "excluded_builder_shared_objects": excluded_shared_objects,
        "artifacts": generated,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "wheel": wheels[0].name,
                "pristine_files": len(expected_source),
                "pristine_tree_sha256": source_manifest["tree_sha256"],
                "generated_artifacts": len(generated),
                "native_input_tree_sha256": binding[
                    "base_native_input_tree_sha256"
                ],
                "runtime_build_outputs_present": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
