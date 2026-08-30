#!/usr/bin/env python3
"""Generate self-contained vLLM Harbor Dockerfiles from task metadata."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys


TOKEN = "__VLLM_BASE_SHA__"
SHA_RE = re.compile(r"[0-9a-f]{40}")
TEMPLATE_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEMPLATE_DIR.parents[1]
TEMPLATE_PATH = TEMPLATE_DIR / "Dockerfile"


def metadata_value(task_file: Path, key: str) -> str:
    text = task_file.read_text()
    section_match = re.search(
        r"(?ms)^\[metadata\][ \t]*\n(.*?)(?=^\[|\Z)",
        text,
    )
    if section_match is None:
        raise ValueError(f"{task_file}: missing [metadata] section")
    value_match = re.search(
        rf'(?m)^{re.escape(key)}[ \t]*=[ \t]*"([^"]+)"[ \t]*$',
        section_match.group(1),
    )
    if value_match is None:
        raise ValueError(f"{task_file}: missing string metadata.{key}")
    return value_match.group(1)


def render(task_dir: Path, template: str) -> tuple[Path, str]:
    task_file = task_dir / "task.toml"
    if metadata_value(task_file, "repository") != "vllm-project/vllm":
        raise ValueError(f"{task_file}: not a vllm-project/vllm task")

    base_commit = metadata_value(task_file, "base_commit")
    if SHA_RE.fullmatch(base_commit) is None:
        raise ValueError(f"{task_file}: base_commit must be 40 lowercase hex characters")

    output = task_dir / "environment" / "Dockerfile"
    return output, template.replace(TOKEN, base_commit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task_dirs",
        nargs="+",
        type=Path,
        help="Task directories, absolute or relative to the repository root",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if a generated Dockerfile is missing or stale",
    )
    args = parser.parse_args()

    template = TEMPLATE_PATH.read_text()
    if template.count(TOKEN) != 1:
        raise ValueError(f"{TEMPLATE_PATH}: expected exactly one {TOKEN} token")

    stale = False
    for raw_task_dir in args.task_dirs:
        task_dir = raw_task_dir if raw_task_dir.is_absolute() else REPO_ROOT / raw_task_dir
        output, generated = render(task_dir.resolve(), template)
        digest = hashlib.sha256(generated.encode()).hexdigest()
        if args.check:
            if not output.is_file() or output.read_text() != generated:
                print(f"STALE {output}", file=sys.stderr)
                stale = True
            else:
                print(f"OK {digest} {output}")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(generated)
            print(f"WROTE {digest} {output}")

    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
