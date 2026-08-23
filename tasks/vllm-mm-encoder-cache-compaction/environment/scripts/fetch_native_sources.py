#!/usr/bin/env python3
"""Fetch and verify the public native-build source archives.

This script runs only while the image is built.  Every URL, revision, and
archive digest is committed in ``native-build-deps-manifest.json``.  Runtime
containers never execute this script and never need network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
from urllib.request import Request, urlopen


CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str) -> None:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = Request(
                url,
                headers={"User-Agent": "ai-infra-bench-public-builder/1.0"},
            )
            with urlopen(request, timeout=180) as response, destination.open("wb") as out:
                shutil.copyfileobj(response, out, length=CHUNK_SIZE)
            digest = sha256_file(destination)
            if digest != expected_sha256:
                raise ValueError(
                    f"downloaded SHA-256 mismatch for {url}: {digest}"
                )
            return
        except Exception as exc:  # noqa: BLE001 - retain the final network error
            last_error = exc
            destination.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(attempt * 2)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def validate_archive_members(archive: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive path: {member.name!r}")
            if member.issym() or member.islnk():
                target = PurePosixPath(member.linkname)
                if target.is_absolute() or ".." in target.parts:
                    raise ValueError(
                        f"unsafe archive link: {member.name!r} -> {member.linkname!r}"
                    )


def extract_single_root(
    archive: Path, destination: Path, *, replace_empty: bool = False
) -> None:
    validate_archive_members(archive)
    with tempfile.TemporaryDirectory(prefix="native-source-extract-") as temporary:
        temporary_root = Path(temporary)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(temporary_root)  # noqa: S202 - all members checked above
        roots = [item for item in temporary_root.iterdir()]
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError(f"archive does not contain exactly one root directory: {archive}")
        if destination.exists():
            if (
                replace_empty
                and destination.is_dir()
                and not destination.is_symlink()
                and not any(destination.iterdir())
            ):
                destination.rmdir()
            else:
                raise FileExistsError(
                    f"native source destination already exists: {destination}"
                )
        shutil.move(str(roots[0]), destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    archives = output / ".archives"
    archives.mkdir()

    extracted: dict[str, Path] = {}
    for artifact in manifest["artifacts"]:
        name = str(artifact["name"])
        archive = archives / Path(str(artifact["archive"])).name
        download(str(artifact["url"]), archive, str(artifact["sha256"]))
        if artifact.get("submodule_of"):
            continue
        destination = output / str(artifact["source_directory"])
        extract_single_root(archive, destination)
        extracted[name] = destination

    for artifact in manifest["artifacts"]:
        parent_name = artifact.get("submodule_of")
        if not parent_name:
            continue
        parent = extracted[str(parent_name)]
        destination = parent / str(artifact["submodule_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        archive = archives / Path(str(artifact["archive"])).name
        extract_single_root(archive, destination, replace_empty=True)

    summary = {
        "schema_version": "1.0",
        "sources": len(manifest["artifacts"]),
        "manifest_sha256": sha256_file(args.manifest),
        "network_phase": "build-only",
    }
    (output / "fetch-evidence.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(archives)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
