#!/usr/bin/env python3
"""Verify public-build declarations, fetched source and generated runtime.

The committed task contains neither a wheelhouse nor precompiled vLLM native
objects. Python distributions are fetched under pip hash-checking mode; the
base source and native dependencies are strict public-archive locks; vLLM
extensions are compiled from those inputs during the image build.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import stat
from typing import Any


CHUNK_SIZE = 8 * 1024 * 1024
TREE_ALGORITHM = "sha256(path\\0file_sha256\\0size_bytes\\0executable_int\\n)"
REQUIREMENT_RE = re.compile(
    r"^([A-Za-z0-9_.-]+)==([^\s;]+)\s+--hash=sha256:([0-9a-f]{64})$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe manifest path: {value!r}")
    return path


def regular_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"source directory is absent: {root}")
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is forbidden by this source lock: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    return files


def expected_source_tree(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "2.0":
        raise ValueError("sources.lock.json schema mismatch")
    if manifest.get("tree_hash_algorithm") != TREE_ALGORITHM:
        raise ValueError("unknown base source tree hash algorithm")
    if manifest.get("mode_policy") != (
        "canonical-git-100755-if-executable-else-100644"
    ):
        raise ValueError("unknown base source mode policy")
    if manifest.get("symlink_count") != 0:
        raise ValueError("base source lock unexpectedly permits symlinks")
    archive = manifest.get("archive")
    if not isinstance(archive, dict):
        raise ValueError("base source archive declaration is absent")
    expected_url = (
        "https://codeload.github.com/vllm-project/vllm/tar.gz/"
        + str(manifest["base_commit"])
    )
    if archive.get("url") != expected_url:
        raise ValueError("base source URL is not commit-bound")
    if not re.fullmatch(r"[0-9a-f]{64}", str(archive.get("sha256", ""))):
        raise ValueError("invalid base archive SHA-256")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("git_tree_sha1", ""))):
        raise ValueError("invalid base Git tree identity")

    expected: dict[str, dict[str, Any]] = {}
    for artifact in manifest.get("artifacts", []):
        relative = safe_relative(str(artifact["path"])).as_posix()
        if relative in expected:
            raise ValueError(f"duplicate base source path: {relative}")
        if not isinstance(artifact.get("executable"), bool):
            raise ValueError(f"missing executable contract: {relative}")
        expected[relative] = artifact
    if len(expected) != int(manifest.get("artifact_count", -1)):
        raise ValueError("base source artifact count mismatch")
    aggregate = hashlib.sha256()
    for relative, artifact in sorted(expected.items()):
        aggregate.update(
            (
                f"{relative}\0{artifact['sha256']}\0{artifact['size_bytes']}\0"
                f"{int(artifact['executable'])}\n"
            ).encode()
        )
    if aggregate.hexdigest() != manifest["tree_sha256"]:
        raise ValueError("base source manifest semantic hash mismatch")
    return expected


def verify_file(path: Path, artifact: dict[str, Any], relative: str) -> None:
    if path.stat().st_size != int(artifact["size_bytes"]):
        raise ValueError(f"size mismatch: {relative}")
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {relative}")
    expected_executable = bool(artifact["executable"])
    actual_executable = bool(path.stat().st_mode & stat.S_IXUSR)
    if actual_executable != expected_executable:
        raise ValueError(f"executable-bit mismatch: {relative}")


def verify_tree(root: Path, manifest_path: Path, *, allow_additions: bool) -> int:
    expected = expected_source_tree(manifest_path)
    actual = regular_files(root)
    if set(expected) - set(actual):
        missing = sorted(set(expected) - set(actual))
        raise ValueError(f"base source files are missing: {missing[:20]}")
    if not allow_additions and set(actual) != set(expected):
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"base source tree has extra files: {extra[:20]}")
    for relative, artifact in expected.items():
        verify_file(actual[relative], artifact, relative)
    return len(expected)


def parse_requirements_lock(path: Path) -> dict[str, dict[str, str]]:
    requirements: dict[str, dict[str, str]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = REQUIREMENT_RE.fullmatch(line)
        if not match:
            raise ValueError(
                f"requirements.lock line {line_number} is not an exact hash lock: {raw!r}"
            )
        name, version, digest = match.groups()
        canonical = canonical_name(name)
        if canonical in requirements:
            raise ValueError(f"duplicate locked distribution: {canonical}")
        requirements[canonical] = {
            "name": name,
            "version": version,
            "sha256": digest,
        }
    if not requirements:
        raise ValueError("requirements.lock is empty")
    return requirements


def verify_public_wheel_lock(lock_root: Path) -> dict[str, Any]:
    locked = parse_requirements_lock(lock_root / "requirements.lock")
    manifest = load_json(lock_root / "wheelhouse-manifest.json")
    if manifest.get("distribution") != "public-pypi-hash-locked":
        raise ValueError("wheel manifest is not the public-build protocol")
    forbidden = {
        canonical_name(str(item)) for item in manifest["forbidden_distributions"]
    }
    for relative, expected_sha in manifest.get("source_request_sha256", {}).items():
        path = lock_root / safe_relative(str(relative))
        if not path.is_file() or sha256_file(path) != str(expected_sha):
            raise ValueError(f"dependency source request hash mismatch: {relative}")
    entries: dict[str, dict[str, Any]] = {}
    for artifact in manifest["artifacts"]:
        canonical = canonical_name(str(artifact["name"]))
        if canonical in entries or canonical in forbidden:
            raise ValueError(f"invalid public wheel entry: {canonical}")
        entries[canonical] = artifact
    if set(entries) != set(locked):
        raise ValueError(
            "public wheel manifest differs from requirements.lock: "
            f"missing={sorted(set(locked)-set(entries))[:10]} "
            f"extra={sorted(set(entries)-set(locked))[:10]}"
        )
    for name, requirement in locked.items():
        artifact = entries[name]
        if str(artifact["version"]) != requirement["version"]:
            raise ValueError(f"version mismatch in public wheel manifest: {name}")
        if str(artifact["sha256"]) != requirement["sha256"]:
            raise ValueError(f"hash mismatch in public wheel manifest: {name}")
    return {
        "distributions": len(locked),
        "index_url": manifest["index_url"],
        "forbidden": sorted(forbidden),
    }


def verify_native_source_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    names: set[str] = set()
    archives: set[str] = set()
    for artifact in manifest["artifacts"]:
        name = str(artifact["name"])
        archive = safe_relative(str(artifact["archive"])).as_posix()
        url = str(artifact["url"])
        digest = str(artifact["sha256"])
        if name in names or archive in archives:
            raise ValueError(f"duplicate native source entry: {name}/{archive}")
        if not url.startswith("https://codeload.github.com/"):
            raise ValueError(f"native source is not a pinned codeload URL: {url}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid native source hash: {name}")
        safe_relative(str(artifact.get("source_directory", name)))
        if artifact.get("submodule_of"):
            safe_relative(str(artifact["submodule_path"]))
        names.add(name)
        archives.add(archive)
    return {"sources": len(names), "archives": len(archives)}


def selected_native_inputs(source_root: Path) -> tuple[int, str]:
    aggregate = hashlib.sha256()
    count = 0
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source_root).as_posix()
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        selected = (
            relative
            in {"CMakeLists.txt", "setup.py", "pyproject.toml", "use_existing_torch.py"}
            or relative.startswith("cmake/")
            or relative.startswith("csrc/")
        )
        if not selected:
            continue
        aggregate.update(relative.encode() + b"\0")
        aggregate.update(sha256_file(path).encode() + b"\n")
        count += 1
    return count, aggregate.hexdigest()


def verify_native_source_binding(
    lock_root: Path, source_root: Path | None
) -> dict[str, Any]:
    binding = load_json(lock_root / "native-source-binding.json")
    source_manifest = load_json(lock_root / "sources.lock.json")
    if binding.get("base_commit") != source_manifest.get("base_commit"):
        raise ValueError("base source and native binding commits differ")
    if binding.get("base_source_manifest") != "sources.lock.json":
        raise ValueError("native binding does not name the base source lock")
    if binding.get("build_protocol") != (
        "compile-from-hash-verified-public-base-archive"
    ):
        raise ValueError("native source binding protocol mismatch")
    result: dict[str, Any] = {
        "expected_files": int(binding["base_native_input_file_count"]),
        "expected_tree_sha256": binding["base_native_input_tree_sha256"],
    }
    if source_root is not None:
        count, digest = selected_native_inputs(source_root)
        if count != int(binding["base_native_input_file_count"]):
            raise ValueError("base native input file count changed")
        if digest != binding["base_native_input_tree_sha256"]:
            raise ValueError("base native input tree hash changed")
        result.update({"files": count, "tree_sha256": digest})
    return result


def verify_native_contract(path: Path) -> dict[str, Any]:
    contract = load_json(path)
    paths: set[str] = set()
    shared_objects = 0
    for artifact in contract["artifacts"]:
        relative = safe_relative(str(artifact["path"])).as_posix()
        if not relative.startswith("vllm/") or relative in paths:
            raise ValueError(f"invalid native runtime contract path: {relative}")
        paths.add(relative)
        shared_objects += int(relative.endswith(".so"))
    if len(paths) != int(contract["artifact_count"]):
        raise ValueError("native runtime contract count mismatch")
    return {"artifacts": len(paths), "shared_objects": shared_objects}


def verify_runtime_metadata_contract(path: Path) -> dict[str, Any]:
    contract = load_json(path)
    required = [safe_relative(str(item)).as_posix() for item in contract["required_files"]]
    if len(required) != len(set(required)) or not required:
        raise ValueError("invalid runtime metadata contract")
    return {"version": contract["version"], "required_files": len(required)}


def verify_public_build_manifest(lock_root: Path) -> dict[str, Any]:
    manifest = load_json(lock_root / "public-build-manifest.json")
    if manifest.get("protocol") != "public-source-build":
        raise ValueError("public-build protocol mismatch")
    if manifest.get("build_time_network") != "public-sources-only":
        raise ValueError("build-time network policy mismatch")
    if manifest.get("runtime_network") != "no-network":
        raise ValueError("runtime network policy mismatch")
    if manifest.get("local_bundle_required") is not False:
        raise ValueError("public build still declares a local bundle")
    image_pattern = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
    for field in ("base_image", "runtime_base_image"):
        if not image_pattern.fullmatch(str(manifest.get(field, ""))):
            raise ValueError(f"{field} is not a digest-pinned public image")
    expected_sources = {
        "digest-pinned public PyTorch base images",
        "https://pypi.org/simple",
        "https://codeload.github.com",
    }
    if set(manifest.get("public_sources", [])) != expected_sources:
        raise ValueError("public build source allowlist changed")
    for field in (
        "base_source_archive_count",
        "native_source_archive_count",
        "python_distribution_count",
    ):
        if not isinstance(manifest.get(field), int) or int(manifest[field]) < 1:
            raise ValueError(f"invalid public build count: {field}")
    return {
        "base_image": manifest["base_image"],
        "runtime_base_image": manifest["runtime_base_image"],
        "local_bundle_required": False,
        "base_source_archive_count": manifest["base_source_archive_count"],
        "native_source_archive_count": manifest["native_source_archive_count"],
        "python_distribution_count": manifest["python_distribution_count"],
    }


def verify_declarations(lock_root: Path) -> dict[str, Any]:
    source_manifest = load_json(lock_root / "sources.lock.json")
    expected_source_tree(lock_root / "sources.lock.json")
    results = {
        "public_build": verify_public_build_manifest(lock_root),
        "base_source": {
            "archive_sha256": source_manifest["archive"]["sha256"],
            "base_commit": source_manifest["base_commit"],
            "files": source_manifest["artifact_count"],
            "git_tree_sha1": source_manifest["git_tree_sha1"],
            "tree_sha256": source_manifest["tree_sha256"],
        },
        "public_wheels": verify_public_wheel_lock(lock_root),
        "native_sources": verify_native_source_manifest(
            lock_root / "native-build-deps-manifest.json"
        ),
        "native_source_binding": verify_native_source_binding(lock_root, None),
        "native_runtime_contract": verify_native_contract(
            lock_root / "native-manifest.json"
        ),
        "runtime_metadata_contract": verify_runtime_metadata_contract(
            lock_root / "runtime-metadata-manifest.json"
        ),
    }
    public = results["public_build"]
    if public["base_source_archive_count"] != 1:
        raise ValueError("exactly one base source archive is required")
    if results["native_sources"]["archives"] != public["native_source_archive_count"]:
        raise ValueError("public/native source archive counts disagree")
    if results["public_wheels"]["distributions"] != public["python_distribution_count"]:
        raise ValueError("public/Python distribution counts disagree")
    return results


def verify_source(lock_root: Path, source_root: Path) -> dict[str, Any]:
    results = verify_declarations(lock_root)
    results["base_source_tree"] = {
        "files": verify_tree(
            source_root, lock_root / "sources.lock.json", allow_additions=False
        )
    }
    results["native_source_binding"] = verify_native_source_binding(
        lock_root, source_root
    )
    return results


def public_version(value: str) -> str:
    return value.split("+", 1)[0]


def verify_installed(lock_root: Path) -> dict[str, Any]:
    locked = parse_requirements_lock(lock_root / "requirements.lock")
    mismatches: list[str] = []
    for name, requirement in locked.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{name}:missing")
            continue
        if public_version(actual) != public_version(requirement["version"]):
            mismatches.append(f"{name}:{actual}!={requirement['version']}")
    if mismatches:
        raise ValueError(f"installed package lock mismatch: {mismatches[:20]}")
    return {"verified_distributions": len(locked)}


def verify_runtime(
    lock_root: Path,
    runtime_root: Path,
    native_build_manifest: Path,
) -> dict[str, Any]:
    contract = load_json(lock_root / "native-manifest.json")
    generated = load_json(native_build_manifest)
    binding = load_json(lock_root / "native-source-binding.json")
    expected_paths = {str(item["path"]) for item in contract["artifacts"]}
    generated_entries = {str(item["path"]): item for item in generated["artifacts"]}
    if set(generated_entries) != expected_paths:
        raise ValueError("generated native manifest differs from runtime contract")
    excluded = generated.get("excluded_builder_shared_objects", [])
    if not isinstance(excluded, list) or any(
        not isinstance(path, str) or not path.endswith(".so") or path in expected_paths
        for path in excluded
    ):
        raise ValueError("invalid excluded builder shared-object evidence")
    if generated["native_input_tree_sha256"] != binding["base_native_input_tree_sha256"]:
        raise ValueError("generated native artifacts are bound to different inputs")
    for relative, artifact in generated_entries.items():
        path = runtime_root / safe_relative(relative)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"generated runtime artifact is absent: {relative}")
        if path.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"generated runtime artifact size mismatch: {relative}")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"generated runtime artifact hash mismatch: {relative}")

    dist = metadata.distribution("vllm")
    metadata_contract = load_json(lock_root / "runtime-metadata-manifest.json")
    if public_version(dist.version) != public_version(str(metadata_contract["version"])):
        raise ValueError(f"vLLM distribution version mismatch: {dist.version}")
    for required in metadata_contract["required_files"]:
        if not any(str(file).endswith(str(required)) for file in (dist.files or [])):
            raise ValueError(f"vLLM distribution metadata file missing: {required}")

    return {
        "base_source_files": verify_tree(
            runtime_root, lock_root / "sources.lock.json", allow_additions=True
        ),
        "native_artifacts": len(generated_entries),
        "native_input_tree_sha256": generated["native_input_tree_sha256"],
        "vllm_version": dist.version,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("declarations", "source", "installed", "runtime"),
        default="declarations",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--native-build-manifest", type=Path)
    parser.add_argument("--control-only", action="store_true")
    args = parser.parse_args()

    lock_root = args.lock_root.resolve()
    mode = "declarations" if args.control_only else args.mode
    if mode == "declarations":
        results = verify_declarations(lock_root)
    elif mode == "source":
        if args.source_root is None:
            parser.error("source mode requires --source-root")
        results = verify_source(lock_root, args.source_root.resolve())
    elif mode == "installed":
        if args.source_root is None:
            parser.error("installed mode requires --source-root")
        results = {
            "source_inputs": verify_source(lock_root, args.source_root.resolve()),
            "installed": verify_installed(lock_root),
        }
    else:
        if args.runtime_root is None or args.native_build_manifest is None:
            parser.error("runtime mode requires --runtime-root and --native-build-manifest")
        results = {
            "declarations": verify_declarations(lock_root),
            "installed": verify_installed(lock_root),
            "runtime": verify_runtime(
                lock_root,
                args.runtime_root.resolve(),
                args.native_build_manifest.resolve(),
            ),
        }
    print(json.dumps(results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
