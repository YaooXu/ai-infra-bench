#!/usr/bin/env python3
"""Add 15 CPU performance candidates without changing the representative 300."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sample_cpu_candidates import (
    architecture_shape,
    concrete_architectures,
    eligibility_failure,
    hardware_scope,
    is_target_population,
    jsonl,
    labels,
    largest_remainder,
    primary_scope,
    sha256_file,
    write_selected_evidence,
)

SCHEMA_VERSION = "vllm_cpu_performance_topup.v1"
POOL_SCHEMA_VERSION = "vllm_cpu_candidate_pool.v1"
DOCUMENTATION_TITLE = re.compile(
    r"^\s*\[(docs?|documentation)\]", re.IGNORECASE
)
FAMILY_WEIGHTS = {
    "architecture_components": 0.35,
    "project_scope": 0.25,
    "merged_month": 0.15,
    "affected_hardware": 0.10,
    "test_assets": 0.10,
    "reproduction_confidence": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compact",
        type=Path,
        default=Path("analysis/RQ1/data/tagging_compact.jsonl"),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("analysis/RQ1/data/tagging_evidence.jsonl.zst"),
    )
    parser.add_argument(
        "--core-jsonl",
        type=Path,
        default=Path(
            "analysis/RQ2/candidates/cpu_300/cpu_candidates_300.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/RQ2/candidates/cpu_300"),
    )
    parser.add_argument("--topup-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--iterations", type=int, default=300_000)
    parser.add_argument("--zstd-level", type=int, default=9)
    return parser.parse_args()


def stable_random(seed: int, instance_id: str) -> float:
    value = hashlib.sha256(f"{seed}:{instance_id}".encode()).hexdigest()[:16]
    return int(value, 16) / float(2**64)


def load_core(path: Path) -> list[dict[str, Any]]:
    rows = list(jsonl(path))
    if len(rows) != 300:
        raise ValueError(f"core candidate set has {len(rows)} rows, expected 300")
    return rows


def retained_performance_row(row: Mapping[str, Any]) -> bool:
    return bool(
        is_target_population(row)
        and labels(row).get("change_type") == "performance"
        and primary_scope(row) != "documentation_examples"
        and not DOCUMENTATION_TITLE.search(str(row.get("title") or ""))
    )


def benchmark_status(row: Mapping[str, Any]) -> str:
    return str(labels(row).get("verification_performance_benchmark"))


def row_cell(row: Mapping[str, Any]) -> str:
    return f"{architecture_shape(row)} | {benchmark_status(row)}"


def multilabel_target(
    rows: Sequence[dict[str, Any]], getter: Any, size: int
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(str(value) for value in getter(row))
    return {
        key: math.floor(value / len(rows) * size + 0.5)
        for key, value in counts.items()
    }


def categorical_target(
    rows: Sequence[dict[str, Any]], getter: Any, size: int
) -> dict[str, int]:
    return largest_remainder(Counter(str(getter(row)) for row in rows), size)


def features(row: Mapping[str, Any]) -> dict[str, set[str]]:
    row_labels = labels(row)
    return {
        "architecture_components": set(concrete_architectures(row)),
        "project_scope": set(row_labels.get("project_scope") or []),
        "merged_month": {str(row.get("merged_month"))},
        "affected_hardware": {
            f"scope:{hardware_scope(row)}",
            *(
                f"platform:{value}"
                for value in row_labels.get("affected_platforms") or []
            ),
        },
        "test_assets": {str(row_labels.get("verification_test_assets"))},
        "reproduction_confidence": {
            str(row_labels.get("reproduction_confidence"))
        },
    }


def targets(
    eligible_performance: Sequence[dict[str, Any]], size: int
) -> dict[str, dict[str, int]]:
    return {
        "architecture_components": multilabel_target(
            eligible_performance, concrete_architectures, size
        ),
        "project_scope": multilabel_target(
            eligible_performance,
            lambda row: labels(row).get("project_scope") or [],
            size,
        ),
        "merged_month": categorical_target(
            eligible_performance, lambda row: row.get("merged_month"), size
        ),
        "affected_hardware": {
            **categorical_target(
                eligible_performance,
                lambda row: f"scope:{hardware_scope(row)}",
                size,
            ),
            **{
                f"platform:{key}": value
                for key, value in multilabel_target(
                    eligible_performance,
                    lambda row: labels(row).get("affected_platforms") or [],
                    size,
                ).items()
            },
        },
        "test_assets": categorical_target(
            eligible_performance,
            lambda row: labels(row).get("verification_test_assets"),
            size,
        ),
        "reproduction_confidence": categorical_target(
            eligible_performance,
            lambda row: labels(row).get("reproduction_confidence"),
            size,
        ),
    }


def selected_counts(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for family, values in features(row).items():
            counts[family].update(values)
    return counts


def feature_loss(
    family: str,
    feature: str,
    count: int,
    target_values: Mapping[str, Mapping[str, int]],
) -> float:
    target = target_values[family][feature]
    return (
        FAMILY_WEIGHTS[family]
        / max(len(target_values[family]), 1)
        * ((count - target) / math.sqrt(max(target, 1) + 1)) ** 2
    )


def total_loss(
    counts: Mapping[str, Counter[str]],
    target_values: Mapping[str, Mapping[str, int]],
) -> float:
    return sum(
        feature_loss(family, feature, counts[family][feature], target_values)
        for family in target_values
        for feature in target_values[family]
    )


def initial_exact_selection(
    available: Sequence[dict[str, Any]],
    shape_quotas: Mapping[str, int],
    benchmark_quotas: Mapping[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:
        raise RuntimeError("performance top-up requires numpy and scipy") from exc

    ordered = sorted(
        available,
        key=lambda row: (int(row.get("source_order") or 0), str(row["id"])),
    )
    matrix = []
    values = []
    for shape, quota in sorted(shape_quotas.items()):
        matrix.append([int(architecture_shape(row) == shape) for row in ordered])
        values.append(quota)
    for status, quota in sorted(benchmark_quotas.items()):
        matrix.append([int(benchmark_status(row) == status) for row in ordered])
        values.append(quota)
    result = milp(
        [stable_random(seed, str(row["id"])) for row in ordered],
        integrality=np.ones(len(ordered), dtype=int),
        bounds=Bounds(np.zeros(len(ordered)), np.ones(len(ordered))),
        constraints=LinearConstraint(
            np.asarray(matrix, dtype=float),
            np.asarray(values, dtype=float),
            np.asarray(values, dtype=float),
        ),
        options={"time_limit": 60},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"performance top-up MILP failed: {result.message}")
    return [
        row
        for row, value in zip(ordered, result.x, strict=True)
        if value > 0.5
    ]


def optimize_within_cells(
    available: Sequence[dict[str, Any]],
    initial: Sequence[dict[str, Any]],
    target_values: Mapping[str, Mapping[str, int]],
    seed: int,
    iterations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed ^ 0xBEEF)
    selected_ids = {str(row["id"]) for row in initial}
    selected_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unselected_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in available:
        destination = (
            selected_by_cell
            if str(row["id"]) in selected_ids
            else unselected_by_cell
        )
        destination[row_cell(row)].append(row)
    movable = [
        cell
        for cell in sorted(selected_by_cell)
        if selected_by_cell[cell] and unselected_by_cell[cell]
    ]
    weights = [len(selected_by_cell[cell]) for cell in movable]
    row_features = {str(row["id"]): features(row) for row in available}
    counts = selected_counts(initial)
    current_loss = total_loss(counts, target_values)
    initial_loss = current_loss
    best_loss = current_loss
    best_ids = set(selected_ids)
    accepted = 0

    for iteration in range(iterations):
        cell = rng.choices(movable, weights=weights, k=1)[0]
        selected_rows = selected_by_cell[cell]
        unselected_rows = unselected_by_cell[cell]
        old_index = rng.randrange(len(selected_rows))
        new_index = rng.randrange(len(unselected_rows))
        old = selected_rows[old_index]
        new = unselected_rows[new_index]
        old_features = row_features[str(old["id"])]
        new_features = row_features[str(new["id"])]
        delta = 0.0
        for family in target_values:
            for feature in old_features[family] ^ new_features[family]:
                if feature not in target_values[family]:
                    continue
                before = counts[family][feature]
                after = before - int(feature in old_features[family]) + int(
                    feature in new_features[family]
                )
                delta += feature_loss(
                    family, feature, after, target_values
                ) - feature_loss(family, feature, before, target_values)
        progress = iteration / max(iterations - 1, 1)
        temperature = max(initial_loss * 0.003 * (1 - progress) ** 2, 1e-12)
        if delta > 0 and rng.random() >= math.exp(-delta / temperature):
            continue
        accepted += 1
        for family in target_values:
            for feature in old_features[family] - new_features[family]:
                counts[family][feature] -= 1
            for feature in new_features[family] - old_features[family]:
                counts[family][feature] += 1
        selected_rows[old_index], unselected_rows[new_index] = new, old
        current_loss += delta
        if current_loss < best_loss:
            best_loss = current_loss
            best_ids = {
                str(row["id"])
                for rows in selected_by_cell.values()
                for row in rows
            }
    selected = sorted(
        (row for row in available if str(row["id"]) in best_ids),
        key=lambda row: (int(row.get("source_order") or 0), str(row["id"])),
    )
    return selected, {
        "iterations": iterations,
        "accepted_swaps": accepted,
        "initial_loss": initial_loss,
        "best_loss": best_loss,
        "fixed_cell": "architecture_shape x performance_benchmark_status",
    }


def topup_metadata(row: Mapping[str, Any], rank: int) -> dict[str, Any]:
    return {
        "candidate_id": f"vllm-performance-topup-{rank:02d}",
        "candidate_rank": rank,
        "track": "performance_oversample",
        "change_type": "performance",
        "primary_scope": primary_scope(row),
        "project_scopes": labels(row).get("project_scope") or [],
        "architecture_shape": architecture_shape(row),
        "architectures": labels(row).get("architecture") or [],
        "performance_benchmark": benchmark_status(row),
        "affected_hardware_scope": hardware_scope(row),
        "affected_platforms": labels(row).get("affected_platforms") or [],
        "merged_month": row.get("merged_month"),
    }


def write_topup_jsonl(
    path: Path,
    selected: Sequence[dict[str, Any]],
    metadata: Mapping[str, dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "selection": metadata[str(row["id"])],
                        "pr": row,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def csv_row(
    track: str,
    candidate_id: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    row_labels = labels(row)
    return {
        "track": track,
        "candidate_id": candidate_id,
        "pr_number": row.get("number"),
        "html_url": row.get("html_url"),
        "title": row.get("title"),
        "merged_month": row.get("merged_month"),
        "change_type": row_labels.get("change_type"),
        "primary_scope": primary_scope(row),
        "project_scopes": ";".join(row_labels.get("project_scope") or []),
        "architecture_shape": architecture_shape(row),
        "architectures": ";".join(row_labels.get("architecture") or []),
        "performance_benchmark": row_labels.get(
            "verification_performance_benchmark"
        ),
        "reproduction_confidence": row_labels.get("reproduction_confidence"),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    core = load_core(args.core_jsonl)
    core_ids = {str(row["pr"]["id"]) for row in core}
    core_performance = [
        row["pr"] for row in core if row["selection"]["change_type"] == "performance"
    ]
    if len(core_performance) != args.topup_size:
        raise ValueError(
            f"core has {len(core_performance)} performance rows, "
            f"expected {args.topup_size}"
        )

    all_rows = list(jsonl(args.compact))
    eligible_performance = [
        row
        for row in all_rows
        if retained_performance_row(row) and eligibility_failure(row) is None
    ]
    available = [
        row for row in eligible_performance if str(row["id"]) not in core_ids
    ]
    shape_quotas = Counter(architecture_shape(row) for row in core_performance)
    combined_benchmark_target = largest_remainder(
        Counter(benchmark_status(row) for row in eligible_performance),
        args.topup_size * 2,
    )
    core_benchmark = Counter(benchmark_status(row) for row in core_performance)
    benchmark_quotas = {
        status: target - core_benchmark[status]
        for status, target in combined_benchmark_target.items()
    }
    if (
        min(benchmark_quotas.values()) < 0
        or sum(benchmark_quotas.values()) != args.topup_size
    ):
        raise ValueError(
            "cannot derive non-negative performance benchmark top-up quotas"
        )

    initial = initial_exact_selection(
        available, shape_quotas, benchmark_quotas, args.seed
    )
    target_values = targets(eligible_performance, args.topup_size)
    selected, optimizer = optimize_within_cells(
        available,
        initial,
        target_values,
        args.seed,
        args.iterations,
    )
    if len(selected) != args.topup_size:
        raise AssertionError(
            f"selected {len(selected)} rows, expected {args.topup_size}"
        )
    if {str(row["id"]) for row in selected} & core_ids:
        raise AssertionError("performance top-up overlaps the representative core")
    if Counter(architecture_shape(row) for row in selected) != shape_quotas:
        raise AssertionError("performance top-up violates architecture-shape quotas")
    if Counter(benchmark_status(row) for row in selected) != Counter(
        benchmark_quotas
    ):
        raise AssertionError("performance top-up violates benchmark-status quotas")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        str(row["id"]): topup_metadata(row, rank)
        for rank, row in enumerate(selected, 1)
    }
    topup_jsonl = args.output_dir / "performance_topup_15.jsonl"
    topup_csv = args.output_dir / "performance_topup_15.csv"
    topup_evidence = args.output_dir / "performance_topup_15.evidence.jsonl.zst"
    combined_jsonl = args.output_dir / "cpu_candidates_315.jsonl"
    combined_csv = args.output_dir / "cpu_candidates_315.csv"
    manifest_path = args.output_dir / "performance_topup_manifest.json"

    write_topup_jsonl(topup_jsonl, selected, metadata)
    topup_csv_rows = [
        csv_row(
            "performance_oversample",
            metadata[str(row["id"])]["candidate_id"],
            row,
        )
        for row in selected
    ]
    write_csv(topup_csv, topup_csv_rows)
    evidence_count = write_selected_evidence(
        args.evidence,
        topup_evidence,
        set(metadata),
        metadata,
        args.zstd_level,
    )
    if evidence_count != args.topup_size:
        raise AssertionError(
            f"top-up evidence contains {evidence_count}, expected {args.topup_size}"
        )

    combined_rows = []
    combined_csv_rows = []
    for row in core:
        combined_rows.append(
            {
                "schema_version": POOL_SCHEMA_VERSION,
                "track": "representative_core",
                "selection": row["selection"],
                "pr": row["pr"],
            }
        )
        combined_csv_rows.append(
            csv_row(
                "representative_core",
                str(row["selection"]["candidate_id"]),
                row["pr"],
            )
        )
    for row in selected:
        combined_rows.append(
            {
                "schema_version": POOL_SCHEMA_VERSION,
                "track": "performance_oversample",
                "selection": metadata[str(row["id"])],
                "pr": row,
            }
        )
    combined_rows.sort(
        key=lambda value: (
            int(value["pr"].get("source_order") or 0),
            str(value["pr"]["id"]),
        )
    )
    combined_csv_rows.extend(topup_csv_rows)
    combined_csv_rows.sort(key=lambda value: (value["track"], value["candidate_id"]))
    with combined_jsonl.open("w", encoding="utf-8") as handle:
        for row in combined_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    write_csv(combined_csv, combined_csv_rows)

    output_paths = {
        path.name: path
        for path in [
            topup_jsonl,
            topup_csv,
            topup_evidence,
            combined_jsonl,
            combined_csv,
        ]
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "counts": {
            "eligible_cpu_performance": len(eligible_performance),
            "representative_core_performance": len(core_performance),
            "available_after_core_and_documentation_title_filter": len(available),
            "performance_topup": len(selected),
            "combined_candidate_pool": len(combined_rows),
        },
        "selection_contract": {
            "core_300_unchanged": True,
            "no_overlap_with_core": True,
            "change_type": "performance",
            "reproduction_platform": "cpu",
            "excluded_primary_scope": "documentation_examples",
            "excluded_title_pattern": DOCUMENTATION_TITLE.pattern,
            "architecture_shape_topup_quotas": dict(sorted(shape_quotas.items())),
            "benchmark_status_topup_quotas": dict(
                sorted(benchmark_quotas.items())
            ),
            "combined_30_benchmark_target": dict(
                sorted(combined_benchmark_target.items())
            ),
        },
        "optimizer": optimizer,
        "inputs": {
            "compact": {"path": str(args.compact), "sha256": sha256_file(args.compact)},
            "evidence": {
                "path": str(args.evidence),
                "sha256": sha256_file(args.evidence),
            },
            "representative_core": {
                "path": str(args.core_jsonl),
                "sha256": sha256_file(args.core_jsonl),
            },
        },
        "outputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in output_paths.items()
        },
        "interpretation": (
            "The representative sample remains 300. These 15 rows form a separate "
            "performance oversample and must not be given representative weight."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "core": 300,
                "performance_topup": len(selected),
                "combined": len(combined_rows),
                "output_dir": str(args.output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
