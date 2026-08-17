#!/usr/bin/env python3
"""Build a compact, auditable analysis layer from a Vela tagging snapshot.

The raw task output embeds the complete multi-gigabyte PR input and agent trace in
every row.  This script retains the semantic labels and the small set of source
fields needed by RQ1, while recording exact coverage and integrity statistics.
It deliberately does not repair or discard failed rows.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "vllm_rq1_tagging_compact.v1"
AUDIT_SCHEMA_VERSION = "vllm_rq1_tagging_snapshot_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--snapshot-as-of", required=True)
    parser.add_argument(
        "--snapshot-artifact", default="frozen-task-348689-snapshot-20260817"
    )
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def json_lines(
    path: Path, hasher: hashlib._Hash
) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            hasher.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSON root must be an object")
            yield line_number, value


def is_bot(login: Any, user_type: Any) -> bool:
    normalized_login = login.lower() if isinstance(login, str) else ""
    normalized_type = user_type.lower() if isinstance(user_type, str) else ""
    return normalized_type == "bot" or normalized_login.endswith("[bot]")


def compact_source(row: dict[str, Any], order: int) -> dict[str, Any]:
    instance = row.get("instance") if isinstance(row.get("instance"), dict) else {}
    identity = (
        instance.get("identity") if isinstance(instance.get("identity"), dict) else {}
    )
    description = (
        instance.get("description")
        if isinstance(instance.get("description"), dict)
        else {}
    )
    timestamps = (
        instance.get("timestamps")
        if isinstance(instance.get("timestamps"), dict)
        else {}
    )
    metrics = (
        instance.get("metrics") if isinstance(instance.get("metrics"), dict) else {}
    )
    git_refs = (
        instance.get("git_refs") if isinstance(instance.get("git_refs"), dict) else {}
    )
    current = (
        instance.get("github_current_metadata")
        if isinstance(instance.get("github_current_metadata"), dict)
        else {}
    )
    current_user = current.get("user") if isinstance(current.get("user"), dict) else {}
    code_changes = (
        instance.get("code_changes")
        if isinstance(instance.get("code_changes"), dict)
        else {}
    )
    review_context = (
        instance.get("review_context")
        if isinstance(instance.get("review_context"), dict)
        else {}
    )
    reviews = review_context.get("reviews") or []
    conversation = review_context.get("conversation_comments") or []
    inline = review_context.get("inline_review_comments") or []
    review_actors = [item for item in reviews if isinstance(item, dict)]
    conversation_actors = [item for item in conversation if isinstance(item, dict)]
    inline_actors = [item for item in inline if isinstance(item, dict)]

    def human_count(items: list[dict[str, Any]]) -> int:
        return sum(
            not is_bot(item.get("author_login"), item.get("author_type"))
            for item in items
        )

    base = git_refs.get("base") if isinstance(git_refs.get("base"), dict) else {}
    head = git_refs.get("head") if isinstance(git_refs.get("head"), dict) else {}
    author_login = description.get("author_login") or current_user.get("login")
    author_type = current_user.get("type")
    files = (
        code_changes.get("files") if isinstance(code_changes.get("files"), list) else []
    )
    compact_files = []
    for item in files:
        if not isinstance(item, dict):
            continue
        compact_files.append(
            {
                "path": item.get("path"),
                "previous_path": item.get("previous_path"),
                "status": item.get("status"),
                "additions": item.get("additions"),
                "deletions": item.get("deletions"),
                "changes": item.get("changes"),
            }
        )
    labels = instance.get("labels") if isinstance(instance.get("labels"), list) else []
    label_names = [
        item.get("name")
        for item in labels
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    return {
        "source_order": order,
        "id": row.get("id"),
        "repository": identity.get("repo"),
        "number": identity.get("number"),
        "html_url": identity.get("html_url"),
        "title": description.get("title"),
        "author_login": author_login,
        "author_type": author_type,
        "author_association": current.get("author_association")
        or description.get("author_association"),
        "author_is_bot": is_bot(author_login, author_type),
        "created_at": timestamps.get("created_at"),
        "merged_at": timestamps.get("merged_at"),
        "merged_month": (
            timestamps.get("merged_at", "")[:7]
            if isinstance(timestamps.get("merged_at"), str)
            else None
        ),
        "base_ref": base.get("ref"),
        "base_sha": base.get("sha"),
        "head_ref": head.get("ref"),
        "head_sha": head.get("sha"),
        "metrics": {
            "changed_files": metrics.get("changed_files"),
            "commits": metrics.get("commits"),
            "additions": metrics.get("additions"),
            "deletions": metrics.get("deletions"),
            "conversation_comments": metrics.get("conversation_comments"),
            "reviews": metrics.get("reviews"),
            "inline_review_comments": metrics.get("inline_review_comments"),
            "human_reviews": human_count(review_actors),
            "human_conversation_comments": human_count(conversation_actors),
            "human_inline_review_comments": human_count(inline_actors),
        },
        "labels": label_names,
        "files": compact_files,
        "patch_sha256": code_changes.get("patch_sha256"),
        "patch_bytes": (
            len(code_changes.get("patch").encode("utf-8"))
            if isinstance(code_changes.get("patch"), str)
            else None
        ),
    }


def leaf_value(value: Any) -> Any:
    return value.get("value") if isinstance(value, dict) else None


def multi_values(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [item.get("value") for item in value if isinstance(item, dict)]


def flatten_labels(tagging: Any) -> dict[str, Any] | None:
    if not isinstance(tagging, dict):
        return None
    verification = (
        tagging.get("verification")
        if isinstance(tagging.get("verification"), dict)
        else {}
    )
    reproduction = (
        tagging.get("reproduction")
        if isinstance(tagging.get("reproduction"), dict)
        else {}
    )
    return {
        "change_type": leaf_value(tagging.get("change_type")),
        "project_scope": multi_values(tagging.get("project_scope")),
        "architecture": multi_values(tagging.get("architecture")),
        "affected_platforms": multi_values(tagging.get("affected_platforms")),
        "verification_test_assets": leaf_value(verification.get("test_assets")),
        "verification_tested": leaf_value(verification.get("tested")),
        "verification_methods": multi_values(verification.get("methods")),
        "verification_performance_benchmark": leaf_value(
            verification.get("performance_benchmark")
        ),
        "reproduction_platform": leaf_value(reproduction.get("platform")),
        "reproduction_accelerator_model": leaf_value(
            reproduction.get("accelerator_model")
        ),
        "reproduction_topology": leaf_value(reproduction.get("topology")),
        "reproduction_accelerator_count": leaf_value(
            reproduction.get("accelerator_count")
        ),
        "reproduction_accelerator_memory": leaf_value(
            reproduction.get("accelerator_memory")
        ),
        "reproduction_host_cpu_architecture": leaf_value(
            reproduction.get("host_cpu_architecture")
        ),
        "reproduction_host_cpu_count": leaf_value(reproduction.get("host_cpu_count")),
        "reproduction_host_memory": leaf_value(reproduction.get("host_memory")),
        "reproduction_confidence": leaf_value(reproduction.get("confidence")),
        "reproduction_commands": multi_values(reproduction.get("commands")),
        "reproduction_software_requirements": multi_values(
            reproduction.get("software_requirements")
        ),
    }


def record_rank(record: dict[str, Any]) -> tuple[int, int, int]:
    validation = record.get("validation")
    valid = isinstance(validation, dict) and validation.get("valid") is True
    return (
        int(record.get("reward") == 1),
        int(valid),
        int(isinstance(record.get("tagging_result"), dict)),
    )


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    source_hasher = hashlib.sha256()
    source_by_id: dict[str, dict[str, Any]] = {}
    source_duplicates: list[str] = []
    source_line_count = 0
    for line_number, row in json_lines(args.input, source_hasher):
        source_line_count = line_number
        instance_id = row.get("id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"{args.input}:{line_number}: missing string id")
        if instance_id in source_by_id:
            source_duplicates.append(instance_id)
        source_by_id[instance_id] = compact_source(row, line_number)
        if line_number % args.progress_every == 0:
            print(f"source: {line_number} rows", file=sys.stderr, flush=True)
    source_sha256 = source_hasher.hexdigest()
    if args.expected_input_sha256 and source_sha256 != args.expected_input_sha256:
        raise ValueError(
            f"input SHA-256 mismatch: {source_sha256} != {args.expected_input_sha256}"
        )

    snapshot_hasher = hashlib.sha256()
    chosen_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: collections.Counter[str] = collections.Counter()
    reward_counts: collections.Counter[str] = collections.Counter()
    success_counts: collections.Counter[str] = collections.Counter()
    validation_counts: collections.Counter[str] = collections.Counter()
    exit_reason_counts: collections.Counter[str] = collections.Counter()
    snapshot_line_count = 0
    malformed_id_lines: list[int] = []
    for line_number, row in json_lines(args.snapshot, snapshot_hasher):
        snapshot_line_count = line_number
        instance_id = row.get("id")
        if not isinstance(instance_id, str) or not instance_id:
            malformed_id_lines.append(line_number)
            instance_id = f"__missing_id_line_{line_number}"
        node_output = (
            row.get("node_output") if isinstance(row.get("node_output"), dict) else {}
        )
        validation = (
            node_output.get("validation")
            if isinstance(node_output.get("validation"), dict)
            else None
        )
        tagging = node_output.get("tagging_result")
        reward_counts[repr(row.get("reward"))] += 1
        success_counts[repr(row.get("_success"))] += 1
        validation_key = (
            repr(validation.get("valid")) if validation is not None else "missing"
        )
        validation_counts[validation_key] += 1
        exit_reason_counts[str(node_output.get("exit_reason") or "missing")] += 1
        compact_result = {
            "snapshot_line": line_number,
            "reward": row.get("reward"),
            "task_success": row.get("_success"),
            "exit_reason": node_output.get("exit_reason"),
            "validation": validation,
            "tagging_result": tagging,
            "derived_labels": node_output.get("derived_labels"),
            "labels": flatten_labels(tagging),
        }
        if instance_id in chosen_by_id:
            duplicate_ids[instance_id] += 1
            if record_rank(compact_result) > record_rank(chosen_by_id[instance_id]):
                chosen_by_id[instance_id] = compact_result
        else:
            chosen_by_id[instance_id] = compact_result
        if line_number % args.progress_every == 0:
            print(f"snapshot: {line_number} rows", file=sys.stderr, flush=True)

    snapshot_ids = set(chosen_by_id)
    source_ids = set(source_by_id)
    missing_ids = sorted(source_ids - snapshot_ids)
    extra_ids = sorted(snapshot_ids - source_ids)

    written = 0
    usable = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for instance_id, source in sorted(
            source_by_id.items(), key=lambda item: item[1]["source_order"]
        ):
            task = chosen_by_id.get(instance_id)
            if task is None:
                task = {
                    "snapshot_line": None,
                    "reward": None,
                    "task_success": None,
                    "exit_reason": "missing_from_snapshot",
                    "validation": None,
                    "tagging_result": None,
                    "derived_labels": None,
                    "labels": None,
                }
            valid = (
                task.get("reward") == 1
                and isinstance(task.get("validation"), dict)
                and task["validation"].get("valid") is True
                and isinstance(task.get("tagging_result"), dict)
            )
            usable += int(valid)
            output = {
                "schema_version": SCHEMA_VERSION,
                **source,
                "tagging": {**task, "usable": valid},
            }
            handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            written += 1

    output_hasher = hashlib.sha256()
    output_bytes = 0
    with args.output.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            output_hasher.update(chunk)
            output_bytes += len(chunk)

    unusable = []
    for instance_id, source in sorted(
        source_by_id.items(), key=lambda item: item[1]["source_order"]
    ):
        task = chosen_by_id.get(instance_id)
        if task is None:
            unusable.append(
                {
                    "id": instance_id,
                    "number": source.get("number"),
                    "reason": "missing_from_snapshot",
                }
            )
            continue
        valid = (
            task.get("reward") == 1
            and isinstance(task.get("validation"), dict)
            and task["validation"].get("valid") is True
            and isinstance(task.get("tagging_result"), dict)
        )
        if not valid:
            unusable.append(
                {
                    "id": instance_id,
                    "number": source.get("number"),
                    "reward": task.get("reward"),
                    "task_success": task.get("task_success"),
                    "exit_reason": task.get("exit_reason"),
                    "validation": task.get("validation"),
                }
            )

    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "task": {
            "task_id": args.task_id,
            "snapshot_as_of": args.snapshot_as_of,
            "terminal_at_snapshot": False,
            "note": (
                "User authorized freezing the near-terminal snapshot and ignoring "
                "the remaining running records; missing and unusable rows remain "
                "explicit."
            ),
        },
        "source": {
            "path": str(args.input),
            "bytes": args.input.stat().st_size,
            "sha256": source_sha256,
            "line_count": source_line_count,
            "unique_ids": len(source_ids),
            "duplicate_ids": sorted(set(source_duplicates)),
        },
        "snapshot": {
            "artifact": args.snapshot_artifact,
            "bytes": args.snapshot.stat().st_size,
            "sha256": snapshot_hasher.hexdigest(),
            "line_count": snapshot_line_count,
            "unique_ids": len(snapshot_ids),
            "duplicate_extra_rows": sum(duplicate_ids.values()),
            "duplicate_ids": dict(sorted(duplicate_ids.items())),
            "malformed_id_lines": malformed_id_lines,
            "reward_counts": dict(sorted(reward_counts.items())),
            "task_success_counts": dict(sorted(success_counts.items())),
            "validation_counts": dict(sorted(validation_counts.items())),
            "exit_reason_counts": dict(sorted(exit_reason_counts.items())),
        },
        "reconciliation": {
            "source_ids_missing_from_snapshot": missing_ids,
            "snapshot_ids_absent_from_source": extra_ids,
            "usable_rows": usable,
            "unusable_rows": len(unusable),
            "coverage_rate": usable / len(source_ids) if source_ids else None,
            "unusable": unusable,
        },
        "compact_output": {
            "path": str(args.output),
            "bytes": output_bytes,
            "sha256": output_hasher.hexdigest(),
            "records": written,
        },
    }
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit["reconciliation"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
