#!/usr/bin/env python3
"""Deterministically regenerate the verifier integrity manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ENVIRONMENT_FILES = [
    ".dockerignore",
    "Dockerfile",
    "lock/ARTIFACTS.md",
    "lock/environment-manifest.json",
    "lock/environment_request.json",
    "lock/environment_spec.json",
    "lock/hardware-manifest.json",
    "lock/native-build-deps-manifest.json",
    "lock/native-manifest.json",
    "lock/native-source-binding.json",
    "lock/public-build-manifest.json",
    "lock/requirements.build.input",
    "lock/base-requirements.cuda.txt",
    "lock/requirements.input",
    "lock/requirements.lock",
    "lock/runtime-metadata-manifest.json",
    "lock/sources.lock.json",
    "lock/wheelhouse-manifest.json",
    "scripts/fetch_base_source.py",
    "scripts/fetch_native_sources.py",
    "scripts/measure_artifact_transfer.py",
    "scripts/prepare_runtime_tree.py",
    "scripts/verify_artifacts.py",
]
EXECUTABLE_FILES = ["pytest_entry.py", "run_verifier.py", "test.sh"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-root", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    args = parser.parse_args()
    tests_root = args.tests_root.resolve()
    environment_root = args.environment_root.resolve()

    test_files: dict[str, str] = {}
    for path in sorted(tests_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in verifier tree: {path}")
        if not path.is_file() or path.name == "integrity.json":
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        test_files[path.relative_to(tests_root).as_posix()] = sha256_file(path)

    environment_files: dict[str, str] = {}
    for relative in ENVIRONMENT_FILES:
        path = environment_root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"protected environment file missing: {relative}")
        environment_files[relative] = sha256_file(path)

    payload = {
        "schema_version": "1.2",
        "environment_spec_sha256": environment_files["lock/environment_spec.json"],
        "environment_files": environment_files,
        "executable_files": EXECUTABLE_FILES,
        "test_files": test_files,
    }
    destination = tests_root / "integrity.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "environment_files": len(environment_files),
                "test_files": len(test_files),
                "integrity_sha256": sha256_file(destination),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
