#!/usr/bin/env python3
"""Fetch and verify the exact public vLLM base source archive.

The compressed archive hash is a strict outer lock.  A matching semantic tree
is never accepted as a fallback when the archive bytes differ: curators must
review and update both locks explicitly.  Extraction rejects links, devices,
duplicate paths and path traversal before materializing the source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import urllib.request


CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {value!r}")
    return path


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "2.0":
        raise ValueError("sources.lock.json schema mismatch")
    archive = value.get("archive")
    if not isinstance(archive, dict):
        raise ValueError("sources.lock.json has no archive contract")
    expected_url = (
        "https://codeload.github.com/vllm-project/vllm/tar.gz/"
        + str(value["base_commit"])
    )
    if archive.get("url") != expected_url:
        raise ValueError("base archive URL is not bound to the exact commit")
    if len(str(archive.get("sha256", ""))) != 64:
        raise ValueError("base archive SHA-256 is invalid")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != value.get("artifact_count"):
        raise ValueError("base source artifact count mismatch")
    if value.get("symlink_count") != 0:
        raise ValueError("this source protocol does not permit symlinks")
    return value


def download(url: str, destination: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "ai-infra-bench-source-fetch/1.0"}
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 200:
                    raise ValueError(f"unexpected HTTP status: {response.status}")
                with destination.open("wb") as output:
                    shutil.copyfileobj(response, output, CHUNK_SIZE)
            return
        except Exception as error:  # report the final public-source failure
            last_error = error
            destination.unlink(missing_ok=True)
            if attempt == 3:
                break
    raise RuntimeError(f"unable to download locked base source: {last_error}")


def expected_artifacts(manifest: dict) -> dict[str, dict]:
    expected: dict[str, dict] = {}
    aggregate = hashlib.sha256()
    for item in manifest["artifacts"]:
        relative = safe_relative(str(item["path"])).as_posix()
        if relative in expected:
            raise ValueError(f"duplicate source manifest path: {relative}")
        if not isinstance(item.get("executable"), bool):
            raise ValueError(f"missing executable mode contract: {relative}")
        expected[relative] = item
    for relative, item in sorted(expected.items()):
        aggregate.update(
            (
                f"{relative}\0{item['sha256']}\0{item['size_bytes']}\0"
                f"{int(item['executable'])}\n"
            ).encode()
        )
    if aggregate.hexdigest() != manifest["tree_sha256"]:
        raise ValueError("sources.lock.json semantic tree hash is invalid")
    return expected


def extract_verified(archive_path: Path, manifest: dict, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"base source output already exists: {output}")
    expected = expected_artifacts(manifest)
    top = str(manifest["archive"]["top_level_directory"])
    seen: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="base-source-", dir=output.parent) as temp:
        temporary_root = Path(temp) / "tree"
        temporary_root.mkdir()
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                archive_name = safe_relative(member.name)
                if archive_name.parts[0] != top:
                    raise ValueError(f"archive member has unexpected root: {member.name}")
                relative_parts = archive_name.parts[1:]
                if not relative_parts:
                    if not member.isdir():
                        raise ValueError("archive root is not a directory")
                    continue
                relative = PurePosixPath(*relative_parts).as_posix()
                destination = temporary_root / Path(*relative_parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError(f"non-regular archive member is forbidden: {relative}")
                if relative in seen or relative not in expected:
                    raise ValueError(f"unexpected or duplicate archive file: {relative}")
                item = expected[relative]
                archive_executable = bool(member.mode & stat.S_IXUSR)
                if archive_executable != item["executable"]:
                    raise ValueError(f"archive executable bit mismatch: {relative}")
                if member.size != int(item["size_bytes"]):
                    raise ValueError(f"archive file size mismatch: {relative}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"archive file cannot be read: {relative}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with destination.open("wb") as handle:
                    while chunk := source.read(CHUNK_SIZE):
                        digest.update(chunk)
                        handle.write(chunk)
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError(f"archive file SHA-256 mismatch: {relative}")
                # Normalize only non-semantic group/other write bits. Git tracks
                # regular files as 100644 or 100755, exactly as locked here.
                os.chmod(destination, 0o755 if item["executable"] else 0o644)
                seen.add(relative)
        if seen != set(expected):
            missing = sorted(set(expected) - seen)
            raise ValueError(f"base archive is incomplete: {missing[:20]}")
        temporary_root.rename(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    with tempfile.TemporaryDirectory(prefix="base-archive-") as temporary:
        archive_path = Path(temporary) / "source.tar.gz"
        download(str(manifest["archive"]["url"]), archive_path)
        actual_sha256 = sha256_file(archive_path)
        expected_sha256 = str(manifest["archive"]["sha256"])
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "base archive SHA-256 mismatch; semantic-tree fallback is forbidden: "
                f"{actual_sha256} != {expected_sha256}"
            )
        extract_verified(archive_path, manifest, args.output)

    print(
        json.dumps(
            {
                "archive_sha256": manifest["archive"]["sha256"],
                "base_commit": manifest["base_commit"],
                "files": manifest["artifact_count"],
                "git_tree_sha1": manifest["git_tree_sha1"],
                "output": str(args.output),
                "tree_sha256": manifest["tree_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
