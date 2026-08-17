#!/usr/bin/env python3
"""Join complete PR evidence with complete semantic tagging results.

The Vela snapshot repeats the task input inside its execution envelope.  This
builder keeps the source PR instance and provenance exactly once, joins the
semantic output by instance id, and removes only execution-infrastructure
fields such as the Ludus request, sandbox, image, pod, workspace, and session
identifier. Credential-shaped values in Authorization headers are redacted
before publication while the surrounding public evidence is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO

SCHEMA_VERSION = "vllm_pr_tagging_evidence.v1"
MANIFEST_SCHEMA_VERSION = "vllm_pr_tagging_evidence_manifest.v1"
AUTHORIZATION_BEARER_TOKEN = re.compile(
    r"(?i)(Authorization:\s*Bearer\s+)(sk-[A-Za-z0-9_-]{20,})"
)
REDACTED_API_TOKEN = "<redacted_api_token>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--expected-compact-sha256")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--zstd-level", type=int, default=9)
    parser.add_argument("--agent-type", default="codex")
    parser.add_argument("--harness-version", default="0.144.1")
    parser.add_argument("--vela-model-id", type=int, default=210145)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_lines(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSON root must be an object")
            yield line_number, value


def public_tagging(value: Any) -> dict[str, Any]:
    tagging = value if isinstance(value, dict) else {}
    return {
        "snapshot_line": tagging.get("snapshot_line"),
        "reward": tagging.get("reward"),
        "task_success": tagging.get("task_success"),
        "exit_reason": tagging.get("exit_reason"),
        "validation": tagging.get("validation"),
        "tagging_result": tagging.get("tagging_result"),
        "derived_labels": tagging.get("derived_labels"),
        "labels": tagging.get("labels"),
        "usable": tagging.get("usable"),
    }


def load_tagging(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    by_id: dict[str, dict[str, Any]] = {}
    count = 0
    for line_number, row in json_lines(path):
        count = line_number
        instance_id = row.get("id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"{path}:{line_number}: missing string id")
        if instance_id in by_id:
            raise ValueError(f"{path}:{line_number}: duplicate id {instance_id}")
        by_id[instance_id] = public_tagging(row.get("tagging"))
    return by_id, count


def redact_api_credentials(value: Any) -> tuple[Any, int]:
    """Redact credential values only when used in Authorization headers."""
    if isinstance(value, str):
        return AUTHORIZATION_BEARER_TOKEN.subn(
            lambda match: f"{match.group(1)}{REDACTED_API_TOKEN}", value
        )
    if isinstance(value, list):
        redacted_items = []
        replacements = 0
        for item in value:
            redacted, count = redact_api_credentials(item)
            redacted_items.append(redacted)
            replacements += count
        return redacted_items, replacements
    if isinstance(value, dict):
        redacted_mapping = {}
        replacements = 0
        for key, item in value.items():
            redacted, count = redact_api_credentials(item)
            redacted_mapping[key] = redacted
            replacements += count
        return redacted_mapping, replacements
    return value, 0


def open_zstd(output: Path, level: int) -> tuple[subprocess.Popen[bytes], BinaryIO]:
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = output.open("wb")
    process = subprocess.Popen(
        ["zstd", f"-{level}", "-T0", "-q", "-c"],
        stdin=subprocess.PIPE,
        stdout=handle,
    )
    if process.stdin is None:
        handle.close()
        raise RuntimeError("zstd stdin pipe was not created")
    return process, handle


def main() -> int:
    args = parse_args()
    if args.output.suffix != ".zst":
        raise ValueError("--output must end in .zst")

    input_sha256 = sha256_file(args.input)
    compact_sha256 = sha256_file(args.compact)
    if args.expected_input_sha256 and input_sha256 != args.expected_input_sha256:
        raise ValueError(
            f"input SHA-256 mismatch: {input_sha256} != "
            f"{args.expected_input_sha256}"
        )
    if args.expected_compact_sha256 and compact_sha256 != args.expected_compact_sha256:
        raise ValueError(
            f"compact SHA-256 mismatch: {compact_sha256} != "
            f"{args.expected_compact_sha256}"
        )

    tagging_by_id, compact_records = load_tagging(args.compact)
    process, compressed_handle = open_zstd(args.output, args.zstd_level)
    uncompressed_hasher = hashlib.sha256()
    uncompressed_bytes = 0
    written = 0
    missing_tagging = 0
    seen_ids: set[str] = set()
    redaction_count = 0
    redacted_instance_ids: set[str] = set()
    try:
        assert process.stdin is not None
        for line_number, row in json_lines(args.input):
            instance_id = row.get("id")
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError(f"{args.input}:{line_number}: missing string id")
            if instance_id in seen_ids:
                raise ValueError(
                    f"{args.input}:{line_number}: duplicate id {instance_id}"
                )
            seen_ids.add(instance_id)
            tagging = tagging_by_id.get(instance_id)
            if tagging is None:
                raise ValueError(f"tagging compact is missing source id {instance_id}")
            missing_tagging += int(tagging.get("tagging_result") is None)
            instance, replacements = redact_api_credentials(row.get("instance"))
            redaction_count += replacements
            if replacements:
                redacted_instance_ids.add(instance_id)
            evidence = {
                "schema_version": SCHEMA_VERSION,
                "source_order": line_number,
                "id": instance_id,
                "data_source": row.get("data_source"),
                "source": row.get("source"),
                "instance": instance,
                "tagging": tagging,
            }
            encoded = (
                json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            process.stdin.write(encoded)
            uncompressed_hasher.update(encoded)
            uncompressed_bytes += len(encoded)
            written += 1
            if line_number % args.progress_every == 0:
                print(f"evidence: {line_number} rows", file=sys.stderr, flush=True)
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"zstd exited with status {return_code}")
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    finally:
        compressed_handle.close()

    extra_tagging_ids = sorted(set(tagging_by_id) - seen_ids)
    if extra_tagging_ids:
        raise ValueError(
            f"compact contains {len(extra_tagging_ids)} ids absent from source"
        )
    if written != compact_records:
        raise ValueError(
            f"record count mismatch: source={written}, compact={compact_records}"
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "labeling_configuration": {
            "agent_name": "default",
            "agent_type": args.agent_type,
            "harness_version": args.harness_version,
            "vela_model_id": args.vela_model_id,
        },
        "source_input": {
            "path": str(args.input),
            "bytes": args.input.stat().st_size,
            "sha256": input_sha256,
            "records": written,
        },
        "tagging_compact": {
            "path": str(args.compact),
            "bytes": args.compact.stat().st_size,
            "sha256": compact_sha256,
            "records": compact_records,
        },
        "evidence_output": {
            "path": str(args.output),
            "compression": f"zstd-{args.zstd_level}",
            "compressed_bytes": args.output.stat().st_size,
            "compressed_sha256": sha256_file(args.output),
            "uncompressed_bytes": uncompressed_bytes,
            "uncompressed_sha256": uncompressed_hasher.hexdigest(),
            "records": written,
            "records_with_tagging_result": written - missing_tagging,
            "records_missing_tagging_result": missing_tagging,
        },
        "security_redactions": {
            "policy": (
                "Replace credential-shaped values only when they occur as the "
                "value of an Authorization: Bearer header."
            ),
            "replacement": REDACTED_API_TOKEN,
            "affected_records": len(redacted_instance_ids),
            "replacement_count": redaction_count,
            "affected_instance_ids": sorted(redacted_instance_ids),
        },
        "retained_per_pr": [
            "data_source and source provenance",
            (
                "complete source instance including PR body, commits, review "
                "context, patch, CI evidence, and current GitHub metadata, "
                "except credential values redacted for safe publication"
            ),
            (
                "complete tagging result including Chinese reasoning, validation, "
                "derived labels, and flattened labels"
            ),
        ],
        "removed_execution_envelope": [
            "source environment image and workspace",
            "source node_results",
            "Vela and Ludus request envelope",
            "agent, model, tool, mount, image, and sandbox configuration",
            "pod, job, namespace, IP, workspace, result path, and session identifier",
            "duplicate copies of source metadata",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["evidence_output"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
