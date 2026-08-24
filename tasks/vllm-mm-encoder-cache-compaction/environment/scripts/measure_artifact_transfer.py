#!/usr/bin/env python3
"""Measure the exact Harbor v0.20 excluded-directory transfer archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time


CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def tree_size(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def write_report(path: Path | None, value: dict[str, object]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    print(text, end="")


def output_tail(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-1000:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/app"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--release-threshold-sec", type=float, default=60.0)
    parser.add_argument("--hard-timeout-sec", type=float, default=120.0)
    args = parser.parse_args()

    source = args.source.resolve()
    archive = args.archive.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"artifact source is absent: {source}")
    if source == archive or source in archive.parents:
        raise ValueError("transfer archive must be outside the artifact source")
    if (source / "build").exists():
        raise ValueError("release gate rejected /app/build")
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.unlink(missing_ok=True)

    command = ["tar", "czf", str(archive)]
    command.extend(f"--exclude={value}" for value in args.exclude)
    command.extend(["-C", str(source), "."])
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.hard_timeout_sec,
            check=False,
        )
        elapsed = time.monotonic() - started
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        report = {
            "schema_version": "1.0",
            "status": "HARD_TIMEOUT",
            "source": str(source),
            "command": command,
            "duration_seconds": elapsed,
            "hard_timeout_seconds": args.hard_timeout_sec,
            "release_threshold_seconds": args.release_threshold_sec,
            "stdout_tail": output_tail(exc.stdout),
            "stderr_tail": output_tail(exc.stderr),
            "app_size_bytes": tree_size(source),
            "git_size_bytes": tree_size(source / ".git"),
            "build_size_bytes": tree_size(source / "build"),
        }
        write_report(args.report, report)
        return 1

    if proc.returncode != 0 or not archive.is_file():
        report = {
            "schema_version": "1.0",
            "status": "TAR_FAILED",
            "source": str(source),
            "command": command,
            "duration_seconds": elapsed,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
            "app_size_bytes": tree_size(source),
            "git_size_bytes": tree_size(source / ".git"),
            "build_size_bytes": tree_size(source / "build"),
        }
        write_report(args.report, report)
        return 1

    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
    regular_members = [member for member in members if member.isfile()]
    report = {
        "schema_version": "1.0",
        "status": (
            "PASS" if elapsed <= args.release_threshold_sec else "TOO_SLOW"
        ),
        "source": str(source),
        "command": command,
        "excludes": list(args.exclude),
        "duration_seconds": elapsed,
        "hard_timeout_seconds": args.hard_timeout_sec,
        "release_threshold_seconds": args.release_threshold_sec,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "archive_member_count": len(members),
        "archive_regular_file_count": len(regular_members),
        "archive_uncompressed_file_bytes": sum(
            member.size for member in regular_members
        ),
        "app_size_bytes": tree_size(source),
        "git_size_bytes": tree_size(source / ".git"),
        "build_size_bytes": tree_size(source / "build"),
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-1000:],
    }
    write_report(args.report, report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
