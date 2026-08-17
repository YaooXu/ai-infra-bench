#!/usr/bin/env python3
"""Select a three-times-overprovisioned CPU candidate set for RQ2.

The target distribution is measured on every fully schema-valid PR whose
selected reproduction platform is CPU.  Candidate eligibility adds practical
execution-evidence and size gates.  The selected 300 PRs are exactly three
times the integer workload-bucket quotas derived for a final 100-task set.

Fixed workload bucket:
    change_type x primary_project_scope x architecture_shape

Within those fixed quotas, deterministic local search calibrates the remaining
multi-label and secondary distributions to the observed CPU workload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "vllm_cpu_candidate_selection.v1"
EVIDENCE_SCHEMA_VERSION = "vllm_cpu_candidate_evidence.v1"
ALGORITHM_VERSION = "cpu-workload-calibrated-threefold.v1"
PRIMARY_SCOPE_ORDER = (
    "production_code",
    "tests",
    "benchmarks",
    "build",
    "ci",
    "documentation_examples",
    "developer_tooling",
    "other",
    "unknown",
)
ARCHITECTURE_SENTINELS = {"support_only", "unknown"}
PLATFORM_SENTINELS = {"backend_agnostic", "unknown"}
ROUTINE_TITLE = re.compile(
    r"\b(revert|backport|cherry[- ]?pick|merge main|release branch)\b",
    re.IGNORECASE,
)
DEFAULT_EXCLUDED_CHANGE_TYPES = ("documentation",)
DEFAULT_EXCLUDED_PRIMARY_SCOPES = ("documentation_examples",)
FAMILY_WEIGHTS = {
    "architecture_components": 0.32,
    "project_scope": 0.27,
    "scope_integration": 0.16,
    "merged_month": 0.10,
    "affected_hardware": 0.08,
    "patch_size": 0.04,
    "author_association": 0.03,
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
        "--output-dir",
        type=Path,
        default=Path("analysis/RQ2/candidates/cpu_300"),
    )
    parser.add_argument("--final-size", type=int, default=100)
    parser.add_argument("--candidate-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--iterations", type=int, default=600_000)
    parser.add_argument("--zstd-level", type=int, default=9)
    parser.add_argument(
        "--exclude-change-type",
        action="append",
        default=list(DEFAULT_EXCLUDED_CHANGE_TYPES),
        help=(
            "Dominant change_type to exclude before deriving workload quotas. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--exclude-primary-scope",
        action="append",
        default=list(DEFAULT_EXCLUDED_PRIMARY_SCOPES),
        help=(
            "Primary project scope to exclude before deriving workload quotas. "
            "May be repeated."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def labels(row: Mapping[str, Any]) -> dict[str, Any]:
    tagging = row.get("tagging")
    if not isinstance(tagging, dict):
        return {}
    value = tagging.get("labels")
    return value if isinstance(value, dict) else {}


def churn(row: Mapping[str, Any]) -> int:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return 0
    return int(metrics.get("additions") or 0) + int(metrics.get("deletions") or 0)


def concrete_architectures(row: Mapping[str, Any]) -> list[str]:
    values = labels(row).get("architecture") or []
    return [value for value in values if value not in ARCHITECTURE_SENTINELS]


def primary_scope(row: Mapping[str, Any]) -> str:
    scopes = set(labels(row).get("project_scope") or [])
    return next((value for value in PRIMARY_SCOPE_ORDER if value in scopes), "unknown")


def architecture_shape(row: Mapping[str, Any]) -> str:
    values = labels(row).get("architecture") or []
    concrete = concrete_architectures(row)
    if "support_only" in values:
        return "support_only"
    if "unknown" in values or not concrete:
        return "unknown"
    if len(concrete) == 1:
        return "single_component"
    return "multi_component"


def hardware_scope(row: Mapping[str, Any]) -> str:
    platforms = labels(row).get("affected_platforms") or []
    concrete = [value for value in platforms if value not in PLATFORM_SENTINELS]
    if platforms == ["backend_agnostic"]:
        return "backend_agnostic"
    if platforms == ["unknown"] or not platforms:
        return "unknown"
    if len(concrete) == 1:
        return "backend_specific"
    if len(concrete) > 1:
        return "cross_backend"
    return "unknown"


def workload_bucket(row: Mapping[str, Any]) -> str:
    return " | ".join(
        [
            str(labels(row).get("change_type")),
            primary_scope(row),
            architecture_shape(row),
        ]
    )


def count_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    return "3+"


def churn_bucket(value: int) -> str:
    if value <= 10:
        return "5-10"
    if value <= 50:
        return "11-50"
    if value <= 200:
        return "51-200"
    if value <= 500:
        return "201-500"
    return "501-2000"


def is_target_population(row: Mapping[str, Any]) -> bool:
    tagging = row.get("tagging")
    return bool(
        isinstance(tagging, dict)
        and tagging.get("usable")
        and labels(row).get("reproduction_platform") == "cpu"
    )


def eligibility_failure(row: Mapping[str, Any]) -> str | None:
    row_labels = labels(row)
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    if bool(row.get("author_is_bot")):
        return "bot_author"
    if not row_labels.get("reproduction_commands"):
        return "no_reproduction_command"
    if row_labels.get("reproduction_confidence") not in {"high", "medium"}:
        return "low_reproduction_confidence"
    if row_labels.get("verification_tested") != "passed":
        return "relevant_verification_not_passed"
    if not 5 <= churn(row) <= 2_000:
        return "churn_outside_5_2000"
    if int(metrics.get("changed_files") or 0) > 30:
        return "more_than_30_changed_files"
    if ROUTINE_TITLE.search(str(row.get("title") or "")):
        return "routine_revert_backport_or_merge"
    return None


def largest_remainder(counts: Mapping[str, int], total: int) -> dict[str, int]:
    denominator = sum(counts.values())
    if denominator <= 0:
        raise ValueError("largest-remainder allocation requires a positive denominator")
    raw = {key: value / denominator * total for key, value in counts.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(
        raw,
        key=lambda key: (raw[key] - result[key], str(key)),
        reverse=True,
    )
    for key in order[:remainder]:
        result[key] += 1
    return result


def build_workload_quotas(
    population: Sequence[dict[str, Any]],
    eligible: Sequence[dict[str, Any]],
    final_size: int,
    multiplier: int,
    seed: int,
    iterations: int = 300_000,
) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    by_change_type = Counter(str(labels(row).get("change_type")) for row in population)
    final_change_type = largest_remainder(by_change_type, final_size)
    final_buckets: dict[str, int] = {}
    for change_type, change_quota in final_change_type.items():
        within_type = Counter(
            workload_bucket(row)
            for row in population
            if labels(row).get("change_type") == change_type
        )
        final_buckets.update(largest_remainder(within_type, change_quota))

    population_counts = Counter(workload_bucket(row) for row in population)
    eligible_counts = Counter(workload_bucket(row) for row in eligible)
    scope_targets = largest_remainder(
        Counter(primary_scope(row) for row in population), final_size
    )
    shape_targets = largest_remainder(
        Counter(architecture_shape(row) for row in population), final_size
    )
    raw_cell_targets = {
        bucket: count / len(population) * final_size
        for bucket, count in population_counts.items()
    }
    cells_by_change_type: dict[str, list[str]] = defaultdict(list)
    for bucket in sorted(population_counts):
        change_type, _, _ = bucket.split(" | ")
        cells_by_change_type[change_type].append(bucket)

    def quota_loss(quotas: Mapping[str, int]) -> tuple[float, int, float]:
        scope_counts: Counter[str] = Counter()
        shape_counts: Counter[str] = Counter()
        for bucket, count in quotas.items():
            _, scope, shape = bucket.split(" | ")
            scope_counts[scope] += count
            shape_counts[shape] += count
        margin_loss = sum(
            (scope_counts[key] - value) ** 2
            for key, value in scope_targets.items()
        ) + sum(
            (shape_counts[key] - value) ** 2
            for key, value in shape_targets.items()
        )
        cell_loss = sum(
            (quotas.get(bucket, 0) - target) ** 2 / max(target, 0.5)
            for bucket, target in raw_cell_targets.items()
        )
        return 1_000.0 * margin_loss + cell_loss, margin_loss, cell_loss

    rng = random.Random(seed ^ 0xC0FFEE)
    change_types = sorted(cells_by_change_type)
    current = dict(final_buckets)
    current_loss, _, _ = quota_loss(current)
    initial_loss, initial_margin_loss, initial_cell_loss = quota_loss(current)
    best = dict(current)
    best_loss = current_loss
    for iteration in range(iterations):
        change_type = rng.choice(change_types)
        cells = cells_by_change_type[change_type]
        sources = [bucket for bucket in cells if current.get(bucket, 0) > 0]
        source = rng.choice(sources)
        destination = rng.choice(cells)
        if source == destination:
            continue
        if eligible_counts[destination] < (
            current.get(destination, 0) + 1
        ) * multiplier:
            continue
        current[source] -= 1
        current[destination] = current.get(destination, 0) + 1
        proposed_loss, _, _ = quota_loss(current)
        delta = proposed_loss - current_loss
        progress = iteration / max(iterations - 1, 1)
        temperature = 5_000.0 * (1.0 - progress) ** 2 + 0.001
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current_loss = proposed_loss
            if current_loss < best_loss:
                best_loss = current_loss
                best = dict(current)
        else:
            current[source] += 1
            current[destination] -= 1

    final_buckets = best
    candidate_buckets = {
        bucket: final_count * multiplier
        for bucket, final_count in final_buckets.items()
    }
    _, best_margin_loss, best_cell_loss = quota_loss(final_buckets)
    return final_buckets, candidate_buckets, {
        "iterations": iterations,
        "initial_loss": initial_loss,
        "initial_margin_squared_error": initial_margin_loss,
        "initial_cell_loss": initial_cell_loss,
        "best_loss": best_loss,
        "best_margin_squared_error": best_margin_loss,
        "best_cell_loss": best_cell_loss,
        "target_change_type": dict(sorted(final_change_type.items())),
        "target_primary_scope": dict(sorted(scope_targets.items())),
        "target_architecture_shape": dict(sorted(shape_targets.items())),
        "note": (
            "Change-type totals are exact. Primary-scope and architecture-shape "
            "targets may be jointly infeasible under observed structural cells."
        ),
    }


def categorical_targets(
    population: Sequence[dict[str, Any]],
    getter: Any,
    final_size: int,
    multiplier: int,
) -> dict[str, int]:
    counts = Counter(str(getter(row)) for row in population)
    return {
        key: value * multiplier
        for key, value in largest_remainder(counts, final_size).items()
    }


def multilabel_targets(
    population: Sequence[dict[str, Any]],
    getter: Any,
    final_size: int,
    multiplier: int,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in population:
        counts.update(str(value) for value in getter(row))
    return {
        key: math.floor(value / len(population) * final_size + 0.5) * multiplier
        for key, value in counts.items()
    }


def calibration_features(row: Mapping[str, Any]) -> dict[str, set[str]]:
    row_labels = labels(row)
    scopes = [str(value) for value in row_labels.get("project_scope") or []]
    architectures = [str(value) for value in row_labels.get("architecture") or []]
    platforms = [str(value) for value in row_labels.get("affected_platforms") or []]
    return {
        "architecture_components": {
            value for value in architectures if value not in ARCHITECTURE_SENTINELS
        },
        "project_scope": set(scopes),
        "scope_integration": {
            f"cross_scope:{str(len(scopes) > 1).lower()}",
            f"project_scope_count:{count_bucket(len(scopes))}",
        },
        "merged_month": {str(row.get("merged_month"))},
        "affected_hardware": {
            f"scope:{hardware_scope(row)}",
            *(f"platform:{value}" for value in platforms),
        },
        "patch_size": {churn_bucket(churn(row))},
        "author_association": {str(row.get("author_association") or "unknown")},
    }


def build_calibration_targets(
    population: Sequence[dict[str, Any]], final_size: int, multiplier: int
) -> dict[str, dict[str, int]]:
    return {
        "architecture_components": multilabel_targets(
            population,
            concrete_architectures,
            final_size,
            multiplier,
        ),
        "project_scope": multilabel_targets(
            population,
            lambda row: labels(row).get("project_scope") or [],
            final_size,
            multiplier,
        ),
        "scope_integration": {
            **categorical_targets(
                population,
                lambda row: (
                    "cross_scope:"
                    f"{str(len(labels(row).get('project_scope') or []) > 1).lower()}"
                ),
                final_size,
                multiplier,
            ),
            **categorical_targets(
                population,
                lambda row: (
                    "project_scope_count:"
                    f"{count_bucket(len(labels(row).get('project_scope') or []))}"
                ),
                final_size,
                multiplier,
            ),
        },
        "merged_month": categorical_targets(
            population,
            lambda row: row.get("merged_month"),
            final_size,
            multiplier,
        ),
        "affected_hardware": {
            **categorical_targets(
                population,
                lambda row: f"scope:{hardware_scope(row)}",
                final_size,
                multiplier,
            ),
            **{
                f"platform:{key}": value
                for key, value in multilabel_targets(
                    population,
                    lambda row: labels(row).get("affected_platforms") or [],
                    final_size,
                    multiplier,
                ).items()
            },
        },
        "patch_size": categorical_targets(
            population, lambda row: churn_bucket(churn(row)), final_size, multiplier
        ),
        "author_association": categorical_targets(
            population,
            lambda row: row.get("author_association") or "unknown",
            final_size,
            multiplier,
        ),
    }


def feature_counts(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for family, values in calibration_features(row).items():
            result[family].update(values)
    return result


def feature_loss(
    family: str,
    feature: str,
    count: int,
    targets: Mapping[str, Mapping[str, int]],
) -> float:
    target = targets[family][feature]
    scale = math.sqrt(max(target, 1) + 1)
    return (
        FAMILY_WEIGHTS[family]
        / max(len(targets[family]), 1)
        * ((count - target) / scale) ** 2
    )


def total_loss(
    counts: Mapping[str, Counter[str]],
    targets: Mapping[str, Mapping[str, int]],
) -> float:
    return sum(
        feature_loss(family, feature, counts[family][feature], targets)
        for family in targets
        for feature in targets[family]
    )


def swap_delta(
    removed: Mapping[str, set[str]],
    added: Mapping[str, set[str]],
    counts: Mapping[str, Counter[str]],
    targets: Mapping[str, Mapping[str, int]],
) -> float:
    delta = 0.0
    for family in targets:
        changed = removed[family] ^ added[family]
        for feature in changed:
            if feature not in targets[family]:
                continue
            before = counts[family][feature]
            after = before - int(feature in removed[family]) + int(
                feature in added[family]
            )
            delta += feature_loss(family, feature, after, targets) - feature_loss(
                family, feature, before, targets
            )
    return delta


def optimize_selection(
    eligible: Sequence[dict[str, Any]],
    bucket_quotas: Mapping[str, int],
    targets: Mapping[str, Mapping[str, int]],
    seed: int,
    iterations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    rows_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        rows_by_bucket[workload_bucket(row)].append(row)
    for rows in rows_by_bucket.values():
        rows.sort(key=lambda row: (int(row.get("source_order") or 0), str(row["id"])))

    selected_by_bucket: dict[str, list[dict[str, Any]]] = {}
    unselected_by_bucket: dict[str, list[dict[str, Any]]] = {}
    for bucket, quota in sorted(bucket_quotas.items()):
        available = rows_by_bucket.get(bucket, [])
        if quota > len(available):
            raise ValueError(
                f"bucket {bucket!r} needs {quota} candidates but has {len(available)}"
            )
        shuffled = available.copy()
        rng.shuffle(shuffled)
        selected_by_bucket[bucket] = shuffled[:quota]
        unselected_by_bucket[bucket] = shuffled[quota:]

    selected = [row for rows in selected_by_bucket.values() for row in rows]
    counts = feature_counts(selected)
    features_by_id = {str(row["id"]): calibration_features(row) for row in eligible}
    current_loss = total_loss(counts, targets)
    initial_loss = current_loss
    best_loss = current_loss
    best_ids = {str(row["id"]) for row in selected}
    movable = [
        bucket
        for bucket in sorted(bucket_quotas)
        if selected_by_bucket[bucket] and unselected_by_bucket[bucket]
    ]
    bucket_weights = [len(selected_by_bucket[bucket]) for bucket in movable]
    accepted = 0
    improving = 0

    for iteration in range(iterations):
        bucket = rng.choices(movable, weights=bucket_weights, k=1)[0]
        selected_rows = selected_by_bucket[bucket]
        unselected_rows = unselected_by_bucket[bucket]
        selected_index = rng.randrange(len(selected_rows))
        unselected_index = rng.randrange(len(unselected_rows))
        old_row = selected_rows[selected_index]
        new_row = unselected_rows[unselected_index]
        delta = swap_delta(
            features_by_id[str(old_row["id"])],
            features_by_id[str(new_row["id"])],
            counts,
            targets,
        )
        progress = iteration / max(iterations - 1, 1)
        temperature = max(initial_loss * 0.0025 * (1.0 - progress) ** 2, 1e-12)
        accept = delta <= 0 or rng.random() < math.exp(-delta / temperature)
        if not accept:
            continue
        accepted += 1
        improving += int(delta < 0)
        old_features = features_by_id[str(old_row["id"])]
        new_features = features_by_id[str(new_row["id"])]
        for family in targets:
            for feature in old_features[family] - new_features[family]:
                counts[family][feature] -= 1
            for feature in new_features[family] - old_features[family]:
                counts[family][feature] += 1
        selected_rows[selected_index], unselected_rows[unselected_index] = (
            new_row,
            old_row,
        )
        current_loss += delta
        if current_loss < best_loss:
            best_loss = current_loss
            best_ids = {
                str(row["id"])
                for rows in selected_by_bucket.values()
                for row in rows
            }

    best = sorted(
        (row for row in eligible if str(row["id"]) in best_ids),
        key=lambda row: (int(row.get("source_order") or 0), str(row["id"])),
    )
    return best, {
        "iterations": iterations,
        "accepted_swaps": accepted,
        "improving_swaps": improving,
        "initial_loss": initial_loss,
        "best_loss": best_loss,
    }


def exact_calibration_selection(
    eligible: Sequence[dict[str, Any]],
    initial_selected: Sequence[dict[str, Any]],
    bucket_quotas: Mapping[str, int],
    targets: Mapping[str, Mapping[str, int]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:
        raise RuntimeError(
            "exact calibration requires numpy and scipy; run with the pinned "
            "analysis/RQ2 requirements"
        ) from exc

    ordered = sorted(
        eligible,
        key=lambda row: (int(row.get("source_order") or 0), str(row["id"])),
    )
    row_features = [calibration_features(row) for row in ordered]
    matrix_rows: list[list[int]] = []
    constraint_values: list[int] = []
    constraint_names: list[str] = []

    for bucket, quota in sorted(bucket_quotas.items()):
        matrix_rows.append(
            [int(workload_bucket(row) == bucket) for row in ordered]
        )
        constraint_values.append(quota)
        constraint_names.append(f"workload_bucket:{bucket}")
    for family in sorted(targets):
        for feature, target in sorted(targets[family].items()):
            matrix_rows.append(
                [int(feature in values[family]) for values in row_features]
            )
            constraint_values.append(target)
            constraint_names.append(f"{family}:{feature}")

    initial_ids = {str(row["id"]) for row in initial_selected}
    objective = []
    for row in ordered:
        stable_random = int(
            hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()[:16], 16
        ) / float(2**64)
        objective.append(
            (0.0 if str(row["id"]) in initial_ids else 1.0)
            + stable_random * 1e-6
        )

    matrix = np.asarray(matrix_rows, dtype=float)
    values = np.asarray(constraint_values, dtype=float)
    result = milp(
        np.asarray(objective, dtype=float),
        integrality=np.ones(len(ordered), dtype=int),
        bounds=Bounds(np.zeros(len(ordered)), np.ones(len(ordered))),
        constraints=LinearConstraint(matrix, values, values),
        options={"time_limit": 120},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"exact calibration MILP failed: {result.message}")
    selected = [
        row
        for row, value in zip(ordered, result.x, strict=True)
        if value > 0.5
    ]
    selected_ids = {str(row["id"]) for row in selected}

    observed_buckets = Counter(workload_bucket(row) for row in selected)
    if observed_buckets != Counter(bucket_quotas):
        raise AssertionError("MILP output violates fixed workload-bucket quotas")
    observed_features = feature_counts(selected)
    violations = []
    for family in targets:
        for feature, target in targets[family].items():
            if observed_features[family][feature] != target:
                violations.append(
                    {
                        "constraint": f"{family}:{feature}",
                        "expected": target,
                        "actual": observed_features[family][feature],
                    }
                )
    if violations:
        raise AssertionError(f"MILP output violates calibration targets: {violations}")

    return selected, {
        "solver": "scipy.optimize.milp (HiGHS)",
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun),
        "constraint_count": len(constraint_names),
        "variable_count": len(ordered),
        "changed_from_local_search": len(initial_ids ^ selected_ids) // 2,
        "all_fixed_and_calibration_targets_exact": True,
    }


def count_named_features(
    rows: Sequence[dict[str, Any]], family: str
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(calibration_features(row)[family])
    return counts


def append_comparison_rows(
    output: list[dict[str, Any]],
    family: str,
    role: str,
    categories: Iterable[str],
    population_counts: Mapping[str, int],
    eligible_counts: Mapping[str, int],
    selected_counts: Mapping[str, int],
    population_size: int,
    eligible_size: int,
    selected_size: int,
    final_targets: Mapping[str, int],
    candidate_targets: Mapping[str, int],
) -> None:
    for category in sorted(set(categories)):
        population_n = int(population_counts.get(category, 0))
        eligible_n = int(eligible_counts.get(category, 0))
        selected_n = int(selected_counts.get(category, 0))
        population_share = population_n / population_size
        selected_share = selected_n / selected_size
        output.append(
            {
                "family": family,
                "bucket": category,
                "role": role,
                "population_n": population_n,
                "population_share": population_share,
                "eligible_n": eligible_n,
                "eligible_share": eligible_n / eligible_size,
                "final_100_target": int(final_targets.get(category, 0)),
                "candidate_300_target": int(candidate_targets.get(category, 0)),
                "selected_300_n": selected_n,
                "selected_share": selected_share,
                "selected_minus_target": selected_n
                - int(candidate_targets.get(category, 0)),
                "share_delta_percentage_points": (selected_share - population_share)
                * 100,
            }
        )


def build_comparison(
    population: Sequence[dict[str, Any]],
    eligible: Sequence[dict[str, Any]],
    selected: Sequence[dict[str, Any]],
    final_buckets: Mapping[str, int],
    candidate_buckets: Mapping[str, int],
    calibration_targets: Mapping[str, Mapping[str, int]],
    multiplier: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    fixed_getters = {
        "change_type": lambda row: str(labels(row).get("change_type")),
        "primary_scope": primary_scope,
        "architecture_shape": architecture_shape,
        "workload_bucket": workload_bucket,
    }
    for family, getter in fixed_getters.items():
        population_counts = Counter(getter(row) for row in population)
        eligible_counts = Counter(getter(row) for row in eligible)
        selected_counts = Counter(getter(row) for row in selected)
        if family == "workload_bucket":
            final_targets = dict(final_buckets)
            candidate_targets = dict(candidate_buckets)
        else:
            final_targets = Counter()
            for bucket, value in final_buckets.items():
                change_type, scope, shape = bucket.split(" | ")
                key = {
                    "change_type": change_type,
                    "primary_scope": scope,
                    "architecture_shape": shape,
                }[family]
                final_targets[key] += value
            candidate_targets = {
                key: value * multiplier for key, value in final_targets.items()
            }
        append_comparison_rows(
            output,
            family,
            "fixed_threefold_quota",
            set(population_counts) | set(final_targets),
            population_counts,
            eligible_counts,
            selected_counts,
            len(population),
            len(eligible),
            len(selected),
            final_targets,
            candidate_targets,
        )

    for family, candidate_targets in calibration_targets.items():
        population_counts = count_named_features(population, family)
        eligible_counts = count_named_features(eligible, family)
        selected_counts = count_named_features(selected, family)
        final_targets = {
            key: value // multiplier for key, value in candidate_targets.items()
        }
        append_comparison_rows(
            output,
            family,
            "calibrated_threefold_target",
            set(population_counts) | set(candidate_targets),
            population_counts,
            eligible_counts,
            selected_counts,
            len(population),
            len(eligible),
            len(selected),
            final_targets,
            candidate_targets,
        )
    return output


def selection_metadata(
    row: Mapping[str, Any],
    final_buckets: Mapping[str, int],
    candidate_buckets: Mapping[str, int],
    rank: int,
) -> dict[str, Any]:
    bucket = workload_bucket(row)
    scopes = labels(row).get("project_scope") or []
    concrete_arch = concrete_architectures(row)
    return {
        "candidate_id": f"vllm-cpu-300-{rank:03d}",
        "candidate_rank": rank,
        "workload_bucket": bucket,
        "final_100_bucket_quota": final_buckets[bucket],
        "candidate_300_bucket_quota": candidate_buckets[bucket],
        "change_type": labels(row).get("change_type"),
        "primary_scope": primary_scope(row),
        "project_scopes": scopes,
        "architecture_shape": architecture_shape(row),
        "architectures": labels(row).get("architecture") or [],
        "production_component_count": len(concrete_arch),
        "cross_component": len(concrete_arch) > 1,
        "cross_scope": len(scopes) > 1,
        "affected_hardware_scope": hardware_scope(row),
        "affected_platforms": labels(row).get("affected_platforms") or [],
        "merged_month": row.get("merged_month"),
        "churn": churn(row),
    }


def write_jsonl_candidates(
    path: Path,
    selected: Sequence[dict[str, Any]],
    metadata_by_id: Mapping[str, dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in selected:
            value = {
                "schema_version": SCHEMA_VERSION,
                "selection": metadata_by_id[str(row["id"])],
                "pr": row,
            }
            handle.write(
                json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def write_candidate_csv(
    path: Path,
    selected: Sequence[dict[str, Any]],
    metadata_by_id: Mapping[str, dict[str, Any]],
) -> None:
    fieldnames = [
        "candidate_id",
        "pr_number",
        "html_url",
        "title",
        "merged_month",
        "change_type",
        "primary_scope",
        "project_scopes",
        "architecture_shape",
        "architectures",
        "cross_component",
        "cross_scope",
        "affected_hardware_scope",
        "affected_platforms",
        "changed_files",
        "churn",
        "verification_test_assets",
        "verification_methods",
        "reproduction_confidence",
        "reproduction_commands",
        "workload_bucket",
        "final_100_bucket_quota",
        "candidate_300_bucket_quota",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in selected:
            row_labels = labels(row)
            metadata = metadata_by_id[str(row["id"])]
            metrics = row.get("metrics") or {}
            writer.writerow(
                {
                    "candidate_id": metadata["candidate_id"],
                    "pr_number": row.get("number"),
                    "html_url": row.get("html_url"),
                    "title": row.get("title"),
                    "merged_month": row.get("merged_month"),
                    "change_type": row_labels.get("change_type"),
                    "primary_scope": metadata["primary_scope"],
                    "project_scopes": ";".join(row_labels.get("project_scope") or []),
                    "architecture_shape": metadata["architecture_shape"],
                    "architectures": ";".join(row_labels.get("architecture") or []),
                    "cross_component": metadata["cross_component"],
                    "cross_scope": metadata["cross_scope"],
                    "affected_hardware_scope": metadata[
                        "affected_hardware_scope"
                    ],
                    "affected_platforms": ";".join(
                        row_labels.get("affected_platforms") or []
                    ),
                    "changed_files": metrics.get("changed_files"),
                    "churn": metadata["churn"],
                    "verification_test_assets": row_labels.get(
                        "verification_test_assets"
                    ),
                    "verification_methods": ";".join(
                        row_labels.get("verification_methods") or []
                    ),
                    "reproduction_confidence": row_labels.get(
                        "reproduction_confidence"
                    ),
                    "reproduction_commands": " || ".join(
                        row_labels.get("reproduction_commands") or []
                    ),
                    "workload_bucket": metadata["workload_bucket"],
                    "final_100_bucket_quota": metadata[
                        "final_100_bucket_quota"
                    ],
                    "candidate_300_bucket_quota": metadata[
                        "candidate_300_bucket_quota"
                    ],
                }
            )


def write_comparison_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_workload_bucket_quotas(
    path: Path,
    population: Sequence[dict[str, Any]],
    eligible: Sequence[dict[str, Any]],
    selected: Sequence[dict[str, Any]],
    final_buckets: Mapping[str, int],
    candidate_buckets: Mapping[str, int],
) -> None:
    population_counts = Counter(workload_bucket(row) for row in population)
    eligible_counts = Counter(workload_bucket(row) for row in eligible)
    selected_counts = Counter(workload_bucket(row) for row in selected)
    fieldnames = [
        "change_type",
        "primary_scope",
        "architecture_shape",
        "workload_bucket",
        "population_n",
        "population_share",
        "eligible_n",
        "final_100_quota",
        "candidate_300_quota",
        "selected_300_n",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for bucket, final_quota in sorted(final_buckets.items()):
            if final_quota <= 0:
                continue
            change_type, scope, shape = bucket.split(" | ")
            writer.writerow(
                {
                    "change_type": change_type,
                    "primary_scope": scope,
                    "architecture_shape": shape,
                    "workload_bucket": bucket,
                    "population_n": population_counts[bucket],
                    "population_share": population_counts[bucket] / len(population),
                    "eligible_n": eligible_counts[bucket],
                    "final_100_quota": final_quota,
                    "candidate_300_quota": candidate_buckets[bucket],
                    "selected_300_n": selected_counts[bucket],
                }
            )


def write_selected_evidence(
    source: Path,
    output: Path,
    selected_ids: set[str],
    metadata_by_id: Mapping[str, dict[str, Any]],
    zstd_level: int,
) -> int:
    output_handle = output.open("wb")
    decoder = subprocess.Popen(["zstd", "-dc", str(source)], stdout=subprocess.PIPE)
    encoder = subprocess.Popen(
        ["zstd", f"-{zstd_level}", "-T0", "-q", "-c"],
        stdin=subprocess.PIPE,
        stdout=output_handle,
    )
    if decoder.stdout is None or encoder.stdin is None:
        output_handle.close()
        raise RuntimeError("failed to open zstd streaming pipes")
    written = 0
    try:
        for raw_line in decoder.stdout:
            if not raw_line.strip():
                continue
            evidence = json.loads(raw_line)
            instance_id = str(evidence.get("id"))
            if instance_id not in selected_ids:
                continue
            value = {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "selection": metadata_by_id[instance_id],
                "evidence": evidence,
            }
            encoder.stdin.write(
                (
                    json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
            )
            written += 1
        decoder.stdout.close()
        decoder_status = decoder.wait()
        encoder.stdin.close()
        encoder_status = encoder.wait()
        if decoder_status != 0 or encoder_status != 0:
            raise RuntimeError(
                f"zstd failed: decoder={decoder_status}, encoder={encoder_status}"
            )
    finally:
        if decoder.poll() is None:
            decoder.kill()
            decoder.wait()
        if encoder.poll() is None:
            if not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.kill()
            encoder.wait()
        output_handle.close()
    return written


def main() -> int:
    args = parse_args()
    if args.final_size <= 0 or args.candidate_size <= 0:
        raise ValueError("sample sizes must be positive")
    if args.candidate_size % args.final_size:
        raise ValueError("candidate size must be an integer multiple of final size")
    multiplier = args.candidate_size // args.final_size
    if multiplier != 3:
        raise ValueError("this selection contract requires exactly threefold quotas")

    rows = list(jsonl(args.compact))
    cpu_population = [row for row in rows if is_target_population(row)]
    exclusion_counts: Counter[str] = Counter()
    executable_frame: list[dict[str, Any]] = []
    for row in cpu_population:
        failure = eligibility_failure(row)
        if failure is None:
            executable_frame.append(row)
        else:
            exclusion_counts[failure] += 1
    excluded_change_types = set(args.exclude_change_type)
    excluded_primary_scopes = set(args.exclude_primary_scope)

    def retained_by_workload_filter(row: Mapping[str, Any]) -> bool:
        return bool(
            labels(row).get("change_type") not in excluded_change_types
            and primary_scope(row) not in excluded_primary_scopes
        )

    population = [
        row
        for row in cpu_population
        if retained_by_workload_filter(row)
    ]
    eligible = [
        row
        for row in executable_frame
        if retained_by_workload_filter(row)
    ]
    eligible.sort(key=lambda row: (int(row.get("source_order") or 0), str(row["id"])))

    final_buckets, candidate_buckets, quota_optimizer = build_workload_quotas(
        population,
        eligible,
        args.final_size,
        multiplier,
        args.seed,
    )
    if sum(final_buckets.values()) != args.final_size:
        raise AssertionError("final workload quotas do not sum to final size")
    if sum(candidate_buckets.values()) != args.candidate_size:
        raise AssertionError("candidate workload quotas do not sum to candidate size")
    calibration_targets = build_calibration_targets(
        population, args.final_size, multiplier
    )
    selected, optimizer = optimize_selection(
        eligible,
        candidate_buckets,
        calibration_targets,
        args.seed,
        args.iterations,
    )
    selected, exact_solver = exact_calibration_selection(
        eligible,
        selected,
        candidate_buckets,
        calibration_targets,
        args.seed,
    )
    if len(selected) != args.candidate_size:
        raise AssertionError(
            f"selected {len(selected)} rows, expected {args.candidate_size}"
        )
    if len({str(row["id"]) for row in selected}) != len(selected):
        raise AssertionError("selected candidate ids are not unique")

    actual_buckets = Counter(workload_bucket(row) for row in selected)
    if actual_buckets != Counter(candidate_buckets):
        raise AssertionError("selected workload buckets differ from threefold quotas")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_by_id = {
        str(row["id"]): selection_metadata(
            row, final_buckets, candidate_buckets, rank
        )
        for rank, row in enumerate(selected, 1)
    }
    candidates_jsonl = args.output_dir / "cpu_candidates_300.jsonl"
    candidates_csv = args.output_dir / "cpu_candidates_300.csv"
    evidence_zst = args.output_dir / "cpu_candidates_300.evidence.jsonl.zst"
    comparison_csv = args.output_dir / "distribution_comparison.csv"
    workload_quotas_csv = args.output_dir / "workload_bucket_quotas.csv"
    manifest_path = args.output_dir / "selection_manifest.json"

    write_jsonl_candidates(candidates_jsonl, selected, metadata_by_id)
    write_candidate_csv(candidates_csv, selected, metadata_by_id)
    comparison = build_comparison(
        population,
        eligible,
        selected,
        final_buckets,
        candidate_buckets,
        calibration_targets,
        multiplier,
    )
    write_comparison_csv(comparison_csv, comparison)
    write_workload_bucket_quotas(
        workload_quotas_csv,
        population,
        eligible,
        selected,
        final_buckets,
        candidate_buckets,
    )
    evidence_count = write_selected_evidence(
        args.evidence,
        evidence_zst,
        set(metadata_by_id),
        metadata_by_id,
        args.zstd_level,
    )
    if evidence_count != args.candidate_size:
        raise AssertionError(
            f"selected evidence contains {evidence_count}, "
            f"expected {args.candidate_size}"
        )

    comparison_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison:
        comparison_by_family[str(row["family"])].append(row)
    family_quality = {}
    for family, family_rows in comparison_by_family.items():
        deltas = [
            abs(float(row["share_delta_percentage_points"]))
            for row in family_rows
        ]
        family_quality[family] = {
            "max_absolute_share_delta_percentage_points": max(deltas, default=0.0),
            "mean_absolute_share_delta_percentage_points": sum(deltas)
            / max(len(deltas), 1),
            "exact_target_buckets": sum(
                int(row["selected_minus_target"] == 0) for row in family_rows
            ),
            "bucket_count": len(family_rows),
        }

    outputs = {
        "cpu_candidates_300.jsonl": candidates_jsonl,
        "cpu_candidates_300.csv": candidates_csv,
        "cpu_candidates_300.evidence.jsonl.zst": evidence_zst,
        "distribution_comparison.csv": comparison_csv,
        "workload_bucket_quotas.csv": workload_quotas_csv,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "seed": args.seed,
        "iterations": args.iterations,
        "population_definition": {
            "tagging_usable": True,
            "reproduction_platform": "cpu",
            "excluded_change_types": sorted(excluded_change_types),
            "excluded_primary_scopes": sorted(excluded_primary_scopes),
        },
        "eligibility_definition": {
            "author_is_bot": False,
            "reproduction_commands": "non_empty",
            "reproduction_confidence": ["high", "medium"],
            "verification_tested": "passed",
            "churn_inclusive": [5, 2_000],
            "changed_files_max": 30,
            "excluded_title_pattern": ROUTINE_TITLE.pattern,
        },
        "counts": {
            "compact_rows": len(rows),
            "cpu_workload_population_before_workload_filter": len(
                cpu_population
            ),
            "target_population_after_workload_filter": len(population),
            "executable_frame_before_workload_filter": len(executable_frame),
            "eligible_pool_after_workload_filter": len(eligible),
            "excluded_population_rows_by_workload_filter": len(cpu_population)
            - len(population),
            "excluded_executable_rows_by_workload_filter": len(executable_frame)
            - len(eligible),
            "selected_candidates": len(selected),
            "intended_final_tasks": args.final_size,
            "overprovisioning_multiplier": multiplier,
        },
        "eligibility_exclusions": dict(sorted(exclusion_counts.items())),
        "fixed_bucket_definition": [
            "change_type",
            "primary_project_scope",
            "architecture_shape",
        ],
        "architecture_shape_definition": {
            "support_only": "no production component",
            "single_component": "exactly one production component",
            "multi_component": "two or more production components",
        },
        "fixed_final_100_bucket_quotas": dict(sorted(final_buckets.items())),
        "fixed_candidate_300_bucket_quotas": dict(sorted(candidate_buckets.items())),
        "calibration_family_weights": FAMILY_WEIGHTS,
        "quota_optimizer": quota_optimizer,
        "optimizer": optimizer,
        "exact_calibration_solver": exact_solver,
        "distribution_quality": family_quality,
        "inputs": {
            "compact": {
                "path": str(args.compact),
                "sha256": sha256_file(args.compact),
            },
            "evidence": {
                "path": str(args.evidence),
                "sha256": sha256_file(args.evidence),
            },
        },
        "outputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in outputs.items()
        },
        "important_interpretation": [
            "change_type is the RQ1 engineering intent, not an RQ2 task contract",
            (
                "documentation-dominant PRs are excluded by change type or primary "
                "scope; documentation_examples may remain as a secondary scope of "
                "another change type"
            ),
            (
                "the 300 rows are candidates and still require human review and "
                "actual CPU execution"
            ),
            "future selection of 100 tasks should retain the final bucket quotas",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "target_cpu_workload_population": len(population),
                "eligible_pool": len(eligible),
                "selected": len(selected),
                "bucket_count": sum(value > 0 for value in final_buckets.values()),
                "best_loss": optimizer["best_loss"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
