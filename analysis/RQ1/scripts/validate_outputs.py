#!/usr/bin/env python3
"""Fail-closed validation for the complete RQ1 analysis package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

EXPECTED_DATABASE_SHA256 = (
    "2ac86507a95f9b8785e6ce0bbf2745e3fbba67c747e37b54020a7e57ce80f8b5"
)
EXPECTED_INPUT_SHA256 = (
    "936f6fe84f484005bd5b32b797ba8ae9b0f2ebc931c9cc9240327746b3fb5626"
)
EXPECTED_SNAPSHOT_SHA256 = (
    "5950542da1f98a4944e453ebee746835002f46e66635f4bf36f9c72c14f19327"
)
EXPECTED_TAXONOMY_SHA256 = (
    "84d938a781638a312d82a7b889d5d6ee1886ce37f025e5562c1edea57dd48ecd"
)
LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    required_docs = [
        "README.md",
        "RQ1_QUESTIONS.md",
        "METHODS.md",
        "FINDINGS_CN.md",
        "RQ2_BRIDGE.md",
        "LIMITATIONS.md",
        "SOURCE_PROVENANCE.md",
    ]
    for relative in required_docs:
        path = root / relative
        ok = path.is_file() and path.stat().st_size > 200
        checks.append({"check": f"required_document:{relative}", "passed": ok})
        if not ok:
            fail(errors, f"missing or empty required document: {relative}")

    snapshot_audit = load_json(root / "manifests/tagging_snapshot_audit.json")
    evidence_manifest = load_json(root / "manifests/tagging_evidence_manifest.json")
    analysis_manifest = load_json(root / "manifests/analysis_manifest.json")
    summary = load_json(root / "data/rq1_summary.json")
    reconciliation = load_json(root / "data/legacy_reconciliation.json")

    expected_values = {
        "database_sha256": (
            analysis_manifest.get("inputs", {}).get("database", {}).get("sha256"),
            EXPECTED_DATABASE_SHA256,
        ),
        "input_sha256": (
            snapshot_audit.get("source", {}).get("sha256"),
            EXPECTED_INPUT_SHA256,
        ),
        "snapshot_sha256": (
            snapshot_audit.get("snapshot", {}).get("sha256"),
            EXPECTED_SNAPSHOT_SHA256,
        ),
        "taxonomy_sha256": (
            analysis_manifest.get("inputs", {}).get("taxonomy", {}).get("sha256"),
            EXPECTED_TAXONOMY_SHA256,
        ),
        "selected_prs": (
            summary.get("deep_population", {}).get("selected"),
            5662,
        ),
        "core_labeled_prs": (
            summary.get("deep_population", {}).get("core_labeled"),
            5649,
        ),
        "missing_core_prs": (
            summary.get("deep_population", {}).get("missing_core"),
            13,
        ),
        "snapshot_rows": (
            snapshot_audit.get("snapshot", {}).get("line_count"),
            5649,
        ),
        "full_schema_valid": (
            snapshot_audit.get("reconciliation", {}).get("usable_rows"),
            5636,
        ),
    }
    for name, (observed, expected) in expected_values.items():
        ok = observed == expected
        checks.append(
            {"check": name, "passed": ok, "observed": observed, "expected": expected}
        )
        if not ok:
            fail(errors, f"{name}: {observed!r} != {expected!r}")

    compact = root / "data/tagging_compact.jsonl"
    compact_manifest = analysis_manifest.get("inputs", {}).get("compact_tagging", {})
    compact_hash = sha256(compact)
    compact_lines = 0
    compact_ids: set[str] = set()
    with compact.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row.get("id")
            if not isinstance(instance_id, str):
                fail(errors, f"compact line {line_number}: invalid id")
            elif instance_id in compact_ids:
                fail(errors, f"compact duplicate id: {instance_id}")
            else:
                compact_ids.add(instance_id)
            compact_lines += 1
    compact_ok = (
        compact_lines == 5662
        and len(compact_ids) == 5662
        and compact_hash == compact_manifest.get("sha256")
    )
    checks.append(
        {
            "check": "compact_integrity",
            "passed": compact_ok,
            "lines": compact_lines,
            "unique_ids": len(compact_ids),
            "sha256": compact_hash,
        }
    )
    if not compact_ok:
        fail(errors, "compact JSONL integrity mismatch")

    evidence = evidence_manifest.get("evidence_output", {})
    redactions = evidence_manifest.get("security_redactions", {})
    evidence_manifest_ok = (
        evidence_manifest.get("source_input", {}).get("sha256")
        == EXPECTED_INPUT_SHA256
        and evidence_manifest.get("tagging_compact", {}).get("sha256")
        == compact_hash
        and evidence.get("records") == 5662
        and evidence.get("records_with_tagging_result") == 5649
        and evidence.get("records_missing_tagging_result") == 13
        and redactions.get("replacement") == "<redacted_api_token>"
        and redactions.get("affected_records") == 1
        and redactions.get("replacement_count") == 2
        and redactions.get("affected_instance_ids")
        == ["vllm-project__vllm-46344"]
    )
    evidence_path = root / "data/tagging_evidence.jsonl.zst"
    evidence_asset_ok = None
    if evidence_path.exists():
        evidence_asset_ok = (
            evidence_path.stat().st_size == evidence.get("compressed_bytes")
            and sha256(evidence_path) == evidence.get("compressed_sha256")
        )
        if not evidence_asset_ok:
            fail(errors, "complete evidence release asset integrity mismatch")
    checks.append(
        {
            "check": "complete_evidence_manifest",
            "passed": evidence_manifest_ok and evidence_asset_ok is not False,
            "release_asset_present": evidence_path.exists(),
            "release_asset_integrity": evidence_asset_ok,
        }
    )
    if not evidence_manifest_ok:
        fail(errors, "complete evidence manifest mismatch")

    tables = root / "tables"
    csv_files = sorted(tables.glob("*.csv"))
    md_tables = sorted(tables.glob("*.md"))
    table_inventory_ok = len(csv_files) == 33 and not md_tables
    checks.append(
        {
            "check": "table_inventory",
            "passed": table_inventory_ok,
            "csv": len(csv_files),
            "markdown": len(md_tables),
        }
    )
    if not table_inventory_ok:
        fail(errors, "expected 33 CSV tables and no generated Markdown tables")
    for path in csv_files:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        if not rows or not rows[0]:
            fail(errors, f"empty CSV table: {path.name}")

    for name in [
        "tag_distribution_change_type",
        "tag_distribution_project_scope",
        "tag_distribution_architecture",
        "tag_distribution_affected_platforms",
    ]:
        with (tables / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        denominators = {int(row["denominator"]) for row in rows}
        if denominators != {5649}:
            fail(errors, f"{name}: unexpected denominators {denominators}")
    with (tables / "tag_distribution_change_type.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        intent_rows = list(csv.DictReader(handle))
    if sum(int(row["n"]) for row in intent_rows) != 5649:
        fail(errors, "single-label change_type counts do not sum to 5649")

    figures = root / "figures"
    pngs = sorted(figures.glob("*.png"))
    svgs = sorted(figures.glob("*.svg"))
    figure_ok = (
        len(pngs) == 12
        and not svgs
        and all(path.stat().st_size > 10_000 for path in pngs)
    )
    checks.append(
        {
            "check": "figure_inventory",
            "passed": figure_ok,
            "png": len(pngs),
            "svg": len(svgs),
        }
    )
    if not figure_ok:
        fail(
            errors,
            "PNG figure inventory or size validation failed, or stale SVGs remain",
        )

    for markdown in root.glob("*.md"):
        text = markdown.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            target = target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("#"):
                continue
            resolved = (markdown.parent / target).resolve()
            if not resolved.exists():
                fail(errors, f"broken link in {markdown.name}: {target}")

    legacy_checks = [
        (
            reconciliation["all_prs_opened_2026_monthly_mean"]["recomputed"],
            reconciliation["all_prs_opened_2026_monthly_mean"]["legacy"],
            "legacy PR arrivals",
        ),
        (
            reconciliation["all_prs_merged_2026_monthly_mean"]["recomputed"],
            reconciliation["all_prs_merged_2026_monthly_mean"]["legacy"],
            "legacy PR merges",
        ),
        (
            reconciliation["review_concentration"]["recomputed"]["review_submissions"],
            reconciliation["review_concentration"]["legacy"]["review_submissions"],
            "legacy review submissions",
        ),
        (
            reconciliation["review_concentration"]["recomputed"]["gini"],
            reconciliation["review_concentration"]["legacy"]["gini"],
            "legacy review Gini",
        ),
    ]
    for observed, expected, name in legacy_checks:
        ok = observed == expected
        checks.append(
            {"check": name, "passed": ok, "observed": observed, "expected": expected}
        )
        if not ok:
            fail(errors, f"{name} mismatch: {observed} != {expected}")

    inventory_rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {
            "artifact_inventory.json",
            "validation_report.json",
        }:
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root)
        if relative == Path("data/tagging_evidence.jsonl.zst"):
            continue
        if relative.parts[0] in {"notion", "paper", "sources"}:
            continue
        inventory_rows.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    inventory = {
        "schema_version": "vllm_rq1_artifact_inventory.v1",
        "files": inventory_rows,
        "file_count": len(inventory_rows),
        "total_bytes": sum(row["bytes"] for row in inventory_rows),
    }
    (root / "manifests/artifact_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = {
        "schema_version": "vllm_rq1_validation_report.v1",
        "passed": not errors,
        "checks": checks,
        "errors": errors,
        "inventory": {
            "file_count": inventory["file_count"],
            "total_bytes": inventory["total_bytes"],
        },
    }
    report_path = root / "manifests/validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"passed": report["passed"], "errors": errors}, ensure_ascii=False)
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
