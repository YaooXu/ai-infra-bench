#!/usr/bin/env python3
"""Reproduce the RQ1 ecosystem and semantic-workload analysis.

All tables and figures are derived from the release SQLite database and the
compact task-348689 snapshot.  The script keeps population denominators explicit
and validates the four RQ1 label dimensions independently from the RQ2-oriented
verification and reproduction fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections import Counter
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

CUTOFF = pd.Timestamp("2026-07-31T23:59:59Z")
DEEP_START = pd.Timestamp("2026-02-01T00:00:00Z")
DEEP_END = pd.Timestamp("2026-08-01T00:00:00Z")
CONTEXT_START = pd.Timestamp("2024-01-01T00:00:00Z")
CORE_FIELDS = (
    "change_type",
    "project_scope",
    "architecture",
    "affected_platforms",
)
EXCLUSIVE_VALUES = {
    "project_scope": {"unknown"},
    "architecture": {"support_only", "unknown"},
    "affected_platforms": {"backend_agnostic", "unknown"},
}
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
PLATFORM_SENTINELS = {"backend_agnostic", "unknown"}
CJK_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(
            "artifacts/rq1/vllm_pr_tagging/source/vllm_github_2026-07-31.sqlite"
        ),
    )
    parser.add_argument(
        "--compact", type=Path, default=root / "data/tagging_compact.jsonl"
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("analysis/RQ1/tagging/vllm_pr_tagging_taxonomy.yaml"),
    )
    parser.add_argument(
        "--legacy-reconciliation",
        type=Path,
        default=root / "data/legacy_reconciliation.json",
    )
    parser.add_argument("--output-root", type=Path, default=root)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def public_path(path: Path) -> str:
    """Return a repository-relative path without publishing host directories."""
    resolved = path.resolve()
    repository = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(repository))
    except ValueError:
        return path.name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def value_definitions(taxonomy: dict[str, Any], field: str) -> dict[str, Any]:
    node = taxonomy.get(field)
    if not isinstance(node, dict) or not isinstance(node.get("values"), dict):
        raise ValueError(f"taxonomy field {field!r} has no closed values")
    return node["values"]


def validate_core_result(
    result: Any, taxonomy: dict[str, Any]
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return False, ["tagging_result is missing or not an object"]
    extra = sorted(set(result) - set(CORE_FIELDS) - {"verification", "reproduction"})
    if extra:
        errors.append(f"unexpected root fields: {extra}")
    for field in CORE_FIELDS:
        allowed = set(value_definitions(taxonomy, field))
        raw = result.get(field)
        items = [raw] if field == "change_type" else raw
        if not isinstance(items, list) or not items:
            errors.append(f"{field}: expected a non-empty label list")
            continue
        labels: list[str] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"{field}.{index}: label is not an object")
                continue
            label = item.get("value")
            reasoning = item.get("reasoning")
            if not isinstance(label, str) or label not in allowed:
                errors.append(f"{field}.{index}: invalid value {label!r}")
            else:
                labels.append(label)
            if (
                not isinstance(reasoning, str)
                or not reasoning.strip()
                or CJK_CHARACTER.search(reasoning) is None
            ):
                errors.append(f"{field}.{index}: missing reasoning")
        if len(labels) != len(set(labels)):
            errors.append(f"{field}: duplicate values")
        if len(labels) > 1 and EXCLUSIVE_VALUES.get(field, set()).intersection(labels):
            errors.append(f"{field}: exclusive sentinel combined with another label")
        if field == "change_type" and len(labels) != 1:
            errors.append("change_type: expected exactly one value")
    return not errors, errors


def flatten_tag_rows(
    rows: list[dict[str, Any]], taxonomy: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    flattened: list[dict[str, Any]] = []
    validation_audit: list[dict[str, Any]] = []
    for row in rows:
        tagging = row.get("tagging") if isinstance(row.get("tagging"), dict) else {}
        result = tagging.get("tagging_result")
        core_valid, core_errors = validate_core_result(result, taxonomy)
        labels = (
            tagging.get("labels") if isinstance(tagging.get("labels"), dict) else {}
        )
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        scopes = (
            labels.get("project_scope")
            if isinstance(labels.get("project_scope"), list)
            else []
        )
        architectures = (
            labels.get("architecture")
            if isinstance(labels.get("architecture"), list)
            else []
        )
        platforms = (
            labels.get("affected_platforms")
            if isinstance(labels.get("affected_platforms"), list)
            else []
        )
        concrete_arch = [
            x for x in architectures if x not in {"support_only", "unknown"}
        ]
        concrete_platforms = [x for x in platforms if x not in PLATFORM_SENTINELS]
        primary_scope = next((x for x in PRIMARY_SCOPE_ORDER if x in scopes), "unknown")
        if "support_only" in architectures:
            architecture_shape = "support_only"
        elif "unknown" in architectures:
            architecture_shape = "unknown"
        elif len(concrete_arch) == 1:
            architecture_shape = f"single:{concrete_arch[0]}"
        elif len(concrete_arch) > 1:
            architecture_shape = "multi_component"
        else:
            architecture_shape = "unknown"
        if platforms == ["backend_agnostic"]:
            hardware_scope = "backend_agnostic"
        elif platforms == ["unknown"] or not platforms:
            hardware_scope = "unknown"
        elif len(concrete_platforms) == 1:
            hardware_scope = "backend_specific"
        elif len(concrete_platforms) > 1:
            hardware_scope = "cross_backend"
        else:
            hardware_scope = "unknown"
        change_type = labels.get("change_type")
        churn = (metrics.get("additions") or 0) + (metrics.get("deletions") or 0)
        archetype = " / ".join(
            [str(change_type), primary_scope, architecture_shape, hardware_scope]
        )
        exact_signature = " | ".join(
            [
                str(change_type),
                "+".join(sorted(scopes)),
                "+".join(sorted(architectures)),
                "+".join(sorted(platforms)),
            ]
        )
        flattened.append(
            {
                "id": row.get("id"),
                "number": row.get("number"),
                "title": row.get("title"),
                "html_url": row.get("html_url"),
                "author_login": row.get("author_login"),
                "author_type": row.get("author_type"),
                "author_association": row.get("author_association"),
                "author_is_bot": bool(row.get("author_is_bot")),
                "created_at": row.get("created_at"),
                "merged_at": row.get("merged_at"),
                "merged_month": row.get("merged_month"),
                "full_schema_valid": bool(tagging.get("usable")),
                "core_valid": core_valid,
                "core_validation_errors": core_errors,
                "change_type": change_type,
                "project_scope": scopes,
                "architecture": architectures,
                "affected_platforms": platforms,
                "primary_scope": primary_scope,
                "architecture_shape": architecture_shape,
                "hardware_scope": hardware_scope,
                "project_scope_count": len(scopes),
                "production_component_count": len(concrete_arch),
                "affected_platform_count": len(concrete_platforms),
                "cross_scope": len(scopes) > 1,
                "cross_component": len(concrete_arch) > 1,
                "cross_backend": len(concrete_platforms) > 1,
                "archetype": archetype,
                "exact_signature": exact_signature,
                "changed_files": metrics.get("changed_files"),
                "commits": metrics.get("commits"),
                "additions": metrics.get("additions"),
                "deletions": metrics.get("deletions"),
                "churn": churn,
                "human_reviews": metrics.get("human_reviews"),
                "human_inline_review_comments": metrics.get(
                    "human_inline_review_comments"
                ),
                "human_conversation_comments": metrics.get(
                    "human_conversation_comments"
                ),
            }
        )
        if not core_valid or not bool(tagging.get("usable")):
            validation_audit.append(
                {
                    "id": row.get("id"),
                    "number": row.get("number"),
                    "full_schema_valid": bool(tagging.get("usable")),
                    "core_valid": core_valid,
                    "core_errors": core_errors,
                    "full_validation": tagging.get("validation"),
                    "exit_reason": tagging.get("exit_reason"),
                }
            )
    frame = pd.DataFrame(flattened)
    frame["merged_at"] = pd.to_datetime(frame["merged_at"], utc=True, format="mixed")
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, format="mixed")
    return frame, validation_audit


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def distribution_table(
    frame: pd.DataFrame,
    field: str,
    ordered_values: Sequence[str],
    denominator: int,
) -> pd.DataFrame:
    if field == "change_type":
        counts = frame[field].value_counts()
    else:
        counts = frame.explode(field)[field].value_counts()
    rows = []
    for value in ordered_values:
        count = int(counts.get(value, 0))
        low, high = wilson_interval(count, denominator)
        rows.append(
            {
                "label": value,
                "n": count,
                "denominator": denominator,
                "share": count / denominator if denominator else math.nan,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows)


def pair_table(frame: pd.DataFrame, field: str, denominator: int) -> pd.DataFrame:
    counts: Counter[tuple[str, str]] = Counter()
    for values in frame[field]:
        for left, right in combinations(sorted(values), 2):
            counts[(left, right)] += 1
    rows = [
        {
            "left": left,
            "right": right,
            "n": count,
            "denominator": denominator,
            "share": count / denominator,
        }
        for (left, right), count in counts.most_common()
    ]
    return pd.DataFrame(rows, columns=["left", "right", "n", "denominator", "share"])


def joint_table(
    frame: pd.DataFrame,
    left: str,
    right: str,
    left_multi: bool,
    right_multi: bool,
) -> pd.DataFrame:
    rows: list[tuple[str, str]] = []
    for _, item in frame.iterrows():
        left_values = item[left] if left_multi else [item[left]]
        right_values = item[right] if right_multi else [item[right]]
        for left_value in left_values:
            for right_value in right_values:
                rows.append((left_value, right_value))
    counts = Counter(rows)
    return pd.DataFrame(
        [
            {left: key[0], right: key[1], "n": count}
            for key, count in counts.most_common()
        ]
    )


def write_table(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    clean = frame.copy()
    for column in clean.select_dtypes(include=["object", "string"]).columns:
        clean[column] = clean[column].map(
            lambda value: value.rstrip() if isinstance(value, str) else value
        )
    clean.to_csv(stem.with_suffix(".csv"), index=False)
    stale_markdown = stem.with_suffix(".md")
    if stale_markdown.exists():
        stale_markdown.unlink()


def parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, format="mixed", errors="coerce")


def month_period(series: pd.Series) -> pd.Series:
    """Convert UTC timestamps to calendar-month periods without timezone warnings."""
    return series.dt.tz_convert(None).dt.to_period("M")


def is_bot_series(frame: pd.DataFrame, login_col: str, type_col: str) -> pd.Series:
    login = frame[login_col].fillna("").astype(str).str.lower()
    user_type = frame[type_col].fillna("").astype(str).str.lower()
    return user_type.eq("bot") | login.str.endswith("[bot]")


def load_ecosystem(
    database: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, str]]:
    connection = sqlite3.connect(database)
    artifacts = pd.read_sql_query(
        """
        SELECT a.database_id, a.number, a.artifact_type, a.author_id,
               a.author_login, a.author_association, u.type AS author_type,
               a.created_at, a.closed_at_cutoff, a.merged_at_cutoff,
               a.state_at_cutoff
        FROM canonical_artifact AS a
        LEFT JOIN user AS u ON u.id = a.author_id
        """,
        connection,
    )
    collaborators = pd.read_sql_query(
        """
        SELECT user_id, role_name, pull, triage, push, maintain, admin,
               _fivetran_synced
        FROM repo_collaborator
        WHERE COALESCE(_fivetran_deleted, 0) = 0 AND triage = 1
        """,
        connection,
    )
    reviews = pd.read_sql_query(
        """
        SELECT r.database_id, r.pull_request_id, r.author_id, r.author_login,
               u.type AS author_type, r.submitted_at, r.state,
               a.author_id AS pr_author_id
        FROM canonical_pull_request_review AS r
        LEFT JOIN user AS u ON u.id = r.author_id
        LEFT JOIN canonical_pull_request AS p ON p.database_id = r.pull_request_id
        LEFT JOIN canonical_artifact AS a ON a.database_id = p.artifact_id
        """,
        connection,
    )
    inline = pd.read_sql_query(
        """
        SELECT c.database_id, c.pull_request_id, c.author_id, c.author_login,
               u.type AS author_type, c.created_at,
               a.author_id AS pr_author_id
        FROM canonical_review_comment AS c
        LEFT JOIN user AS u ON u.id = c.author_id
        LEFT JOIN canonical_pull_request AS p ON p.database_id = c.pull_request_id
        LEFT JOIN canonical_artifact AS a ON a.database_id = p.artifact_id
        """,
        connection,
    )
    connection.close()

    artifacts["created_at"] = parse_datetime(artifacts["created_at"])
    artifacts["closed_at"] = parse_datetime(artifacts["closed_at_cutoff"])
    artifacts["merged_at"] = parse_datetime(artifacts["merged_at_cutoff"])
    artifacts["is_bot"] = is_bot_series(artifacts, "author_login", "author_type")
    artifacts["is_nonbot"] = ~artifacts["is_bot"]
    reviews["submitted_at"] = parse_datetime(reviews["submitted_at"])
    reviews["is_bot"] = is_bot_series(reviews, "author_login", "author_type")
    inline["created_at"] = parse_datetime(inline["created_at"])
    inline["is_bot"] = is_bot_series(inline, "author_login", "author_type")

    collaborator_roles: dict[int, str] = {}
    for item in collaborators.itertuples(index=False):
        write_plus = bool(item.push)
        collaborator_roles[int(item.user_id)] = (
            "snapshot_write_plus" if write_plus else "snapshot_nonwrite"
        )
    return artifacts, reviews, inline, collaborator_roles


def add_author_role(
    frame: pd.DataFrame, collaborator_roles: dict[int, str]
) -> pd.DataFrame:
    def classify(row: pd.Series) -> str:
        if row["is_bot"]:
            return "bot"
        role = (
            collaborator_roles.get(int(row["author_id"]))
            if pd.notna(row["author_id"])
            else None
        )
        return role or "external_nonbot"

    frame = frame.copy()
    frame["author_role"] = frame.apply(classify, axis=1)
    return frame


def monthly_ecosystem_tables(
    artifacts: pd.DataFrame,
    reviews: pd.DataFrame,
    inline: pd.DataFrame,
    collaborator_roles: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    months = pd.period_range("2024-01", "2026-07", freq="M")
    relevant = artifacts[
        artifacts["created_at"].between(CONTEXT_START, CUTOFF, inclusive="both")
    ].copy()
    relevant["created_month"] = month_period(relevant["created_at"])
    relevant["merged_month"] = month_period(relevant["merged_at"])
    relevant["closed_month"] = month_period(relevant["closed_at"])

    all_pr = artifacts[artifacts["artifact_type"].eq("PullRequest")].copy()
    all_pr["created_month"] = month_period(all_pr["created_at"])
    all_issue = artifacts[artifacts["artifact_type"].eq("Issue")].copy()
    all_issue["created_month"] = month_period(all_issue["created_at"])
    first_pr = (
        all_pr[all_pr["is_nonbot"] & all_pr["author_login"].notna()]
        .groupby("author_login")["created_at"]
        .min()
        .to_dict()
    )
    first_issue = (
        all_issue[all_issue["is_nonbot"] & all_issue["author_login"].notna()]
        .groupby("author_login")["created_at"]
        .min()
        .to_dict()
    )

    flow_rows: list[dict[str, Any]] = []
    contributor_rows: list[dict[str, Any]] = []
    for month in months:
        month_start = month.start_time.tz_localize("UTC")
        month_end = (month + 1).start_time.tz_localize("UTC")
        created = relevant[relevant["created_month"].eq(month)]
        pr_created = created[created["artifact_type"].eq("PullRequest")]
        issue_created = created[created["artifact_type"].eq("Issue")]
        pr_all = artifacts[artifacts["artifact_type"].eq("PullRequest")]
        flow_rows.append(
            {
                "month": str(month),
                "issues_opened_all": len(issue_created),
                "issues_opened_nonbot": int(issue_created["is_nonbot"].sum()),
                "prs_opened_all": len(pr_created),
                "prs_opened_nonbot": int(pr_created["is_nonbot"].sum()),
                "prs_merged_all": int(
                    pr_all["merged_at"]
                    .between(month_start, month_end, inclusive="left")
                    .sum()
                ),
                "prs_merged_nonbot": int(
                    (
                        pr_all["merged_at"].between(
                            month_start, month_end, inclusive="left"
                        )
                        & pr_all["is_nonbot"]
                    ).sum()
                ),
                "prs_closed_unmerged_all": int(
                    (
                        pr_all["closed_at"].between(
                            month_start, month_end, inclusive="left"
                        )
                        & pr_all["merged_at"].isna()
                    ).sum()
                ),
            }
        )
        month_pr_nonbot = pr_created[
            pr_created["is_nonbot"] & pr_created["author_login"].notna()
        ]
        month_issue_nonbot = issue_created[
            issue_created["is_nonbot"] & issue_created["author_login"].notna()
        ]
        pr_authors = set(month_pr_nonbot["author_login"])
        issue_authors = set(month_issue_nonbot["author_login"])
        new_pr_authors = {
            login for login in pr_authors if month_start <= first_pr[login] < month_end
        }
        new_issue_authors = {
            login
            for login in issue_authors
            if month_start <= first_issue[login] < month_end
        }
        contributor_rows.append(
            {
                "month": str(month),
                "active_pr_authors_nonbot": len(pr_authors),
                "new_pr_authors_nonbot": len(new_pr_authors),
                "repeat_pr_authors_nonbot": len(pr_authors - new_pr_authors),
                "active_issue_authors_nonbot": len(issue_authors),
                "new_issue_authors_nonbot": len(new_issue_authors),
            }
        )

    reviews = reviews[
        reviews["submitted_at"].between(CONTEXT_START, CUTOFF, inclusive="both")
        & reviews["author_type"].eq("User")
        & reviews["pr_author_id"].notna()
        & reviews["author_id"].ne(reviews["pr_author_id"])
    ].copy()
    reviews["month"] = month_period(reviews["submitted_at"])
    reviews["snapshot_collaborator"] = (
        reviews["author_id"].map(collaborator_roles).notna()
    )
    inline = inline[
        inline["created_at"].between(CONTEXT_START, CUTOFF, inclusive="both")
        & inline["author_type"].eq("User")
        & inline["pr_author_id"].notna()
        & inline["author_id"].ne(inline["pr_author_id"])
    ].copy()
    inline["month"] = month_period(inline["created_at"])
    inline["snapshot_collaborator"] = (
        inline["author_id"].map(collaborator_roles).notna()
    )
    capacity_rows = []
    for month in months:
        month_reviews = reviews[reviews["month"].eq(month)]
        month_inline = inline[inline["month"].eq(month)]
        collab_reviews = month_reviews[month_reviews["snapshot_collaborator"]]
        collab_inline = month_inline[month_inline["snapshot_collaborator"]]
        capacity_rows.append(
            {
                "month": str(month),
                "submitted_reviews_nonbot": len(month_reviews),
                "submitted_reviews_snapshot_collaborator": len(collab_reviews),
                "active_snapshot_reviewers": collab_reviews["author_login"].nunique(),
                "inline_comments_nonbot": len(month_inline),
                "inline_comments_snapshot_collaborator": len(collab_inline),
            }
        )

    period_reviews = reviews[
        reviews["submitted_at"].between(
            pd.Timestamp("2026-01-01T00:00:00Z"), CUTOFF, inclusive="both"
        )
        & reviews["snapshot_collaborator"]
    ]
    reviewer_counts = (
        period_reviews["author_login"].value_counts().sort_values(ascending=False)
    )
    total = int(reviewer_counts.sum())
    shares = reviewer_counts / total if total else reviewer_counts
    cumulative = shares.cumsum()
    values = reviewer_counts.to_numpy(dtype=float)
    if len(values) and values.sum():
        sorted_values = np.sort(values)
        n = len(sorted_values)
        gini = float(
            (2 * np.arange(1, n + 1) - n - 1).dot(sorted_values)
            / (n * sorted_values.sum())
        )
    else:
        gini = math.nan
    concentration = {
        "period": "2026-01_to_2026-07",
        "review_submissions": total,
        "active_snapshot_collaborator_reviewers": int(len(reviewer_counts)),
        "top_5_share": float(shares.iloc[:5].sum()) if total else None,
        "top_10_share": float(shares.iloc[:10].sum()) if total else None,
        "reviewers_for_50pct": int((cumulative < 0.5).sum() + 1) if total else None,
        "reviewers_for_80pct": int((cumulative < 0.8).sum() + 1) if total else None,
        "gini": gini,
    }
    return (
        pd.DataFrame(flow_rows),
        pd.DataFrame(contributor_rows),
        pd.DataFrame(capacity_rows),
        concentration,
    )


def period_context(
    flow: pd.DataFrame, contributors: pd.DataFrame, artifacts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = flow.merge(contributors, on="month")
    period_specs = [
        ("2024", "2024-01", "2024-12"),
        ("2025", "2025-01", "2025-12"),
        ("2026 Jan-Jul", "2026-01", "2026-07"),
    ]
    rows = []
    for name, start, end in period_specs:
        selected = combined[combined["month"].between(start, end)]
        item: dict[str, Any] = {"period": name, "months": len(selected)}
        for column in [
            "issues_opened_all",
            "issues_opened_nonbot",
            "prs_opened_all",
            "prs_opened_nonbot",
            "prs_merged_all",
            "prs_merged_nonbot",
            "active_pr_authors_nonbot",
            "new_pr_authors_nonbot",
        ]:
            item[f"{column}_monthly_mean"] = selected[column].mean()
        rows.append(item)
    period = pd.DataFrame(rows)

    state_rows = []
    for artifact_type in ["PullRequest", "Issue"]:
        selected = artifacts[artifacts["artifact_type"].eq(artifact_type)]
        for state in ["OPEN", "MERGED", "CLOSED"]:
            if artifact_type == "Issue" and state == "MERGED":
                continue
            mask = selected["state_at_cutoff"].eq(state)
            state_rows.append(
                {
                    "artifact_type": artifact_type,
                    "state_at_cutoff": state,
                    "all": int(mask.sum()),
                    "nonbot": int((mask & selected["is_nonbot"]).sum()),
                    "bot": int((mask & selected["is_bot"]).sum()),
                }
            )
    return period, pd.DataFrame(state_rows)


def capacity_period_comparison(
    flow: pd.DataFrame, contributors: pd.DataFrame, capacity: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = flow.merge(contributors, on="month").merge(capacity, on="month")
    specs = [("2025", "2025-01", "2025-12"), ("2026 Jan-Jul", "2026-01", "2026-07")]
    metrics = [
        "issues_opened_all",
        "prs_opened_all",
        "prs_opened_nonbot",
        "prs_merged_all",
        "active_pr_authors_nonbot",
        "new_pr_authors_nonbot",
        "submitted_reviews_snapshot_collaborator",
        "active_snapshot_reviewers",
        "inline_comments_snapshot_collaborator",
    ]
    rows = []
    for period_name, start, end in specs:
        selected = combined[combined["month"].between(start, end)]
        for metric in metrics:
            rows.append(
                {
                    "period": period_name,
                    "metric": metric,
                    "months": len(selected),
                    "monthly_mean": selected[metric].mean(),
                    "monthly_median": selected[metric].median(),
                }
            )
    means = pd.DataFrame(rows)
    pivot = means.pivot(
        index="metric", columns="period", values="monthly_mean"
    ).reset_index()
    pivot["relative_change_2026_vs_2025"] = pivot["2026 Jan-Jul"] / pivot["2025"] - 1
    return means, pivot


def contributor_role_period_table(artifacts: pd.DataFrame) -> pd.DataFrame:
    frame = artifacts[artifacts["artifact_type"].eq("PullRequest")].copy()
    frame["period"] = np.select(
        [
            frame["created_at"].between(
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-12-31T23:59:59Z"),
                inclusive="both",
            ),
            frame["created_at"].between(
                pd.Timestamp("2025-01-01T00:00:00Z"),
                pd.Timestamp("2025-12-31T23:59:59Z"),
                inclusive="both",
            ),
            frame["created_at"].between(
                pd.Timestamp("2026-01-01T00:00:00Z"), CUTOFF, inclusive="both"
            ),
        ],
        ["2024", "2025", "2026 Jan-Jul"],
        default="outside_report_window",
    )
    frame = frame[frame["period"].ne("outside_report_window")]
    table = (
        frame.groupby(["period", "author_role"])
        .agg(prs=("database_id", "size"), unique_authors=("author_login", "nunique"))
        .reset_index()
    )
    table["period_denominator"] = table.groupby("period")["prs"].transform("sum")
    table["share"] = table["prs"] / table["period_denominator"]
    return table


def save_figure(fig: plt.Figure, figures: Path, name: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        figures / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white"
    )
    stale_svg = figures / f"{name}.svg"
    if stale_svg.exists():
        stale_svg.unlink()
    plt.close(fig)


def plot_bar_distribution(
    frame: pd.DataFrame, title: str, figures: Path, name: str
) -> None:
    plot = frame[frame["n"].gt(0)].sort_values("share", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(plot))))
    ax.barh(plot["label"], plot["share"] * 100, color="#2E6F95")
    for index, (_, row) in enumerate(plot.iterrows()):
        ax.text(
            row["share"] * 100 + 0.35,
            index,
            f"{int(row['n']):,} ({row['share']:.1%})",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Share of labeled merged PRs (%)")
    ax.set_title(title)
    ax.set_xlim(0, max(plot["share"].max() * 112, 5))
    sns.despine(ax=ax)
    save_figure(fig, figures, name)


def plot_heatmap(
    joint: pd.DataFrame,
    row_field: str,
    col_field: str,
    row_order: Sequence[str],
    col_order: Sequence[str],
    denominator: int,
    title: str,
    figures: Path,
    name: str,
    figsize: tuple[float, float],
) -> None:
    matrix = joint.pivot_table(
        index=row_field, columns=col_field, values="n", aggfunc="sum", fill_value=0
    )
    matrix = matrix.reindex(index=row_order, columns=col_order, fill_value=0)
    pct = matrix / denominator * 100
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pct,
        cmap="Blues",
        linewidths=0.25,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "% of labeled PRs"},
    )
    ax.set_title(title)
    ax.set_xlabel(col_field.replace("_", " ").title())
    ax.set_ylabel(row_field.replace("_", " ").title())
    save_figure(fig, figures, name)


def make_figures(
    flow: pd.DataFrame,
    contributors: pd.DataFrame,
    capacity: pd.DataFrame,
    distributions: dict[str, pd.DataFrame],
    monthly_intent: pd.DataFrame,
    joints: dict[str, pd.DataFrame],
    archetypes: pd.DataFrame,
    tags: pd.DataFrame,
    taxonomy: dict[str, Any],
    figures: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    colors = {
        "PRs opened": "#B64926",
        "PRs merged": "#2E6F95",
        "Issues opened": "#5B8E7D",
    }
    x = pd.to_datetime(flow["month"] + "-01")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        x,
        flow["prs_opened_all"],
        label="PRs opened",
        color=colors["PRs opened"],
        linewidth=2,
    )
    ax.plot(
        x,
        flow["prs_merged_all"],
        label="PRs merged",
        color=colors["PRs merged"],
        linewidth=2,
    )
    ax.plot(
        x,
        flow["issues_opened_all"],
        label="Issues opened",
        color=colors["Issues opened"],
        linewidth=2,
    )
    ax.axvline(
        pd.Timestamp("2026-02-01"),
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Selected window",
    )
    ax.set_ylabel("Artifacts per month")
    ax.set_title("vLLM contribution intake and integration flow")
    ax.legend(ncol=4, frameon=False)
    sns.despine(ax=ax)
    save_figure(fig, figures, "01_ecosystem_activity_monthly")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    cx = pd.to_datetime(contributors["month"] + "-01")
    ax.plot(
        cx,
        contributors["active_pr_authors_nonbot"],
        label="Active PR authors",
        linewidth=2,
    )
    ax.plot(
        cx,
        contributors["new_pr_authors_nonbot"],
        label="First-observed PR authors",
        linewidth=2,
    )
    ax.plot(
        cx,
        contributors["repeat_pr_authors_nonbot"],
        label="Repeat PR authors",
        linewidth=2,
    )
    ax.set_ylabel("Unique non-bot authors")
    ax.set_title("Growth and composition of the PR contributor population")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    save_figure(fig, figures, "02_contributor_growth_monthly")

    merged_capacity = flow[["month", "prs_opened_all"]].merge(capacity, on="month")
    base = merged_capacity[merged_capacity["month"].eq("2025-01")].iloc[0]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for column, label, color in [
        ("prs_opened_all", "PR intake", "#B64926"),
        ("submitted_reviews_snapshot_collaborator", "Collaborator reviews", "#2E6F95"),
        ("active_snapshot_reviewers", "Active collaborator reviewers", "#5B8E7D"),
    ]:
        baseline = float(base[column]) or 1.0
        ax.plot(
            pd.to_datetime(merged_capacity["month"] + "-01"),
            merged_capacity[column] / baseline * 100,
            label=label,
            linewidth=2,
            color=color,
        )
    ax.axhline(100, color="grey", linewidth=0.8)
    ax.set_ylabel("Index (2025-01 = 100)")
    ax.set_title("Contribution intake versus snapshot-collaborator review capacity")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    save_figure(fig, figures, "03_intake_vs_review_capacity")

    plot_bar_distribution(
        distributions["change_type"],
        "Dominant engineering intent",
        figures,
        "04_change_type",
    )
    plot_bar_distribution(
        distributions["project_scope"],
        "Repository surfaces materially changed",
        figures,
        "05_project_scope",
    )
    plot_bar_distribution(
        distributions["architecture"],
        "vLLM architecture components affected",
        figures,
        "06_architecture",
    )
    plot_bar_distribution(
        distributions["affected_platforms"],
        "Hardware backends materially affected",
        figures,
        "07_affected_platforms",
    )

    intent_order = list(value_definitions(taxonomy, "change_type"))
    intent_matrix = monthly_intent.pivot(
        index="merged_month", columns="change_type", values="share"
    ).fillna(0)
    intent_matrix = intent_matrix.reindex(columns=intent_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    intent_matrix.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("Share within month")
    ax.set_xlabel("Merge month")
    ax.set_title("Engineering-intent mix across the six-month selected window")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    sns.despine(ax=ax)
    save_figure(fig, figures, "08_change_type_by_month")

    plot_heatmap(
        joints["change_scope"],
        "change_type",
        "project_scope",
        intent_order,
        list(value_definitions(taxonomy, "project_scope")),
        len(tags),
        "Joint workload: intent x repository surface",
        figures,
        "09_change_type_x_project_scope",
        (11, 6.5),
    )
    top_arch = (
        distributions["architecture"]
        .sort_values("n", ascending=False)
        .head(14)["label"]
        .tolist()
    )
    plot_heatmap(
        joints["arch_platform"],
        "architecture",
        "affected_platforms",
        top_arch,
        list(value_definitions(taxonomy, "affected_platforms")),
        len(tags),
        "Joint workload: architecture x affected hardware",
        figures,
        "10_architecture_x_hardware",
        (11, 8),
    )

    top_archetypes = archetypes.head(20).sort_values("n")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(top_archetypes["archetype"], top_archetypes["share"] * 100, color="#6A4C93")
    ax.set_xlabel("Share of labeled merged PRs (%)")
    ax.set_title("Most common joint workload archetypes")
    sns.despine(ax=ax)
    save_figure(fig, figures, "11_workload_archetypes")

    complexity = tags.copy()
    complexity["component_shape"] = np.where(
        complexity["production_component_count"].eq(0),
        "support-only",
        np.where(
            complexity["production_component_count"].eq(1),
            "one component",
            "multiple components",
        ),
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.boxplot(
        data=complexity,
        x="component_shape",
        y="churn",
        showfliers=False,
        order=["support-only", "one component", "multiple components"],
        ax=ax,
        color="#7FB3D5",
    )
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("")
    ax.set_ylabel("Patch churn (additions + deletions, symlog)")
    ax.set_title("Patch size rises with architectural integration breadth")
    sns.despine(ax=ax)
    save_figure(fig, figures, "12_complexity_by_component_shape")


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    tables = output_root / "tables"
    figures = output_root / "figures"
    manifests = output_root / "manifests"
    data_dir = output_root / "data"
    for path in [tables, figures, manifests, data_dir]:
        path.mkdir(parents=True, exist_ok=True)

    taxonomy = yaml.safe_load(args.taxonomy.read_text(encoding="utf-8"))
    compact_rows = read_jsonl(args.compact)
    tag_frame, validation_audit = flatten_tag_rows(compact_rows, taxonomy)
    core = tag_frame[tag_frame["core_valid"]].copy()
    if not (
        len(tag_frame) == 5662
        and len(core) >= 5600
        and core["merged_at"].ge(DEEP_START).all()
        and core["merged_at"].lt(DEEP_END).all()
    ):
        raise ValueError("deep-label population validation failed")

    artifacts, reviews, inline, collaborator_roles = load_ecosystem(args.database)
    artifacts = add_author_role(artifacts, collaborator_roles)
    role_by_number = (
        artifacts[artifacts["artifact_type"].eq("PullRequest")]
        .set_index("number")["author_role"]
        .to_dict()
    )
    core["author_role"] = core["number"].map(role_by_number).fillna("unmatched")
    tag_frame["author_role"] = (
        tag_frame["number"].map(role_by_number).fillna("unmatched")
    )

    label_orders = {
        field: list(value_definitions(taxonomy, field)) for field in CORE_FIELDS
    }
    distributions = {
        field: distribution_table(core, field, label_orders[field], len(core))
        for field in CORE_FIELDS
    }
    for field, table in distributions.items():
        write_table(table, tables / f"tag_distribution_{field}")

    scope_pairs = pair_table(core, "project_scope", len(core))
    arch_pairs = pair_table(core, "architecture", len(core))
    platform_pairs = pair_table(core, "affected_platforms", len(core))
    write_table(scope_pairs, tables / "cooccurrence_project_scope")
    write_table(arch_pairs, tables / "cooccurrence_architecture")
    write_table(platform_pairs, tables / "cooccurrence_affected_platforms")

    joints = {
        "change_scope": joint_table(core, "change_type", "project_scope", False, True),
        "change_arch": joint_table(core, "change_type", "architecture", False, True),
        "arch_platform": joint_table(
            core, "architecture", "affected_platforms", True, True
        ),
        "scope_platform": joint_table(
            core, "project_scope", "affected_platforms", True, True
        ),
    }
    for name, table in joints.items():
        write_table(table, tables / f"joint_{name}")

    monthly_counts = (
        core.groupby(["merged_month", "change_type"]).size().rename("n").reset_index()
    )
    monthly_total = core.groupby("merged_month").size().rename("denominator")
    monthly_counts = monthly_counts.merge(monthly_total, on="merged_month")
    monthly_counts["share"] = monthly_counts["n"] / monthly_counts["denominator"]
    write_table(monthly_counts, tables / "monthly_change_type")
    for field in ["project_scope", "architecture", "affected_platforms"]:
        exploded = core[["merged_month", field]].explode(field)
        table = (
            exploded.groupby(["merged_month", field]).size().rename("n").reset_index()
        )
        table = table.merge(monthly_total, on="merged_month")
        table["share"] = table["n"] / table["denominator"]
        write_table(table, tables / f"monthly_{field}")

    archetypes = (
        core.groupby("archetype")
        .agg(
            n=("id", "size"),
            median_churn=("churn", "median"),
            median_files=("changed_files", "median"),
            median_human_reviews=("human_reviews", "median"),
        )
        .sort_values("n", ascending=False)
        .reset_index()
    )
    archetypes["denominator"] = len(core)
    archetypes["share"] = archetypes["n"] / len(core)
    examples = (
        core.sort_values(["archetype", "number"])
        .groupby("archetype")
        .apply(
            lambda group: "; ".join(
                f"#{int(row.number)} {row.title}" for row in group.head(3).itertuples()
            ),
            include_groups=False,
        )
        .rename("examples")
    )
    archetypes = archetypes.merge(examples, on="archetype", how="left")
    write_table(archetypes, tables / "workload_archetypes")
    exact = (
        core["exact_signature"]
        .value_counts()
        .rename_axis("signature")
        .reset_index(name="n")
    )
    exact["denominator"] = len(core)
    exact["share"] = exact["n"] / len(core)
    write_table(exact, tables / "workload_exact_signatures")

    multiplicity = (
        core.groupby(
            ["project_scope_count", "production_component_count", "hardware_scope"]
        )
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    multiplicity["denominator"] = len(core)
    multiplicity["share"] = multiplicity["n"] / len(core)
    write_table(multiplicity, tables / "workload_multiplicity")
    complexity = (
        core.groupby(["change_type", "cross_scope", "cross_component"])
        .agg(
            n=("id", "size"),
            median_churn=("churn", "median"),
            p75_churn=("churn", lambda values: values.quantile(0.75)),
            median_files=("changed_files", "median"),
            median_commits=("commits", "median"),
            median_human_reviews=("human_reviews", "median"),
        )
        .reset_index()
    )
    write_table(complexity, tables / "complexity_by_intent_and_integration")

    role_counts = (
        core.groupby("author_role")
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    role_counts["denominator"] = len(core)
    role_counts["share"] = role_counts["n"] / len(core)
    write_table(role_counts, tables / "deep_population_author_roles")
    intent_role = (
        core.groupby(["author_role", "change_type"]).size().rename("n").reset_index()
    )
    intent_role["role_denominator"] = intent_role.groupby("author_role")["n"].transform(
        "sum"
    )
    intent_role["within_role_share"] = (
        intent_role["n"] / intent_role["role_denominator"]
    )
    write_table(intent_role, tables / "change_type_by_author_role")

    hardware_scope = (
        core.groupby("hardware_scope")
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    hardware_scope["denominator"] = len(core)
    hardware_scope["share"] = hardware_scope["n"] / len(core)
    write_table(hardware_scope, tables / "affected_hardware_scope")
    integration_summary = pd.DataFrame(
        [
            {
                "measure": "multiple_project_scopes",
                "n": int(core["cross_scope"].sum()),
                "denominator": len(core),
                "share": float(core["cross_scope"].mean()),
            },
            {
                "measure": "multiple_production_components",
                "n": int(core["cross_component"].sum()),
                "denominator": len(core),
                "share": float(core["cross_component"].mean()),
            },
            {
                "measure": "multiple_affected_backends",
                "n": int(core["cross_backend"].sum()),
                "denominator": len(core),
                "share": float(core["cross_backend"].mean()),
            },
        ]
    )
    write_table(integration_summary, tables / "integration_shape_summary")

    coverage_rows = [
        {
            "population": "selected_merged_prs",
            "n": len(tag_frame),
            "denominator": len(tag_frame),
            "share": 1.0,
        },
        {
            "population": "full_schema_valid",
            "n": int(tag_frame["full_schema_valid"].sum()),
            "denominator": len(tag_frame),
            "share": float(tag_frame["full_schema_valid"].mean()),
        },
        {
            "population": "rq1_core_valid",
            "n": int(tag_frame["core_valid"].sum()),
            "denominator": len(tag_frame),
            "share": float(tag_frame["core_valid"].mean()),
        },
        {
            "population": "missing_rq1_core_labels",
            "n": int((~tag_frame["core_valid"]).sum()),
            "denominator": len(tag_frame),
            "share": float((~tag_frame["core_valid"]).mean()),
        },
    ]
    coverage = pd.DataFrame(coverage_rows)
    write_table(coverage, tables / "tagging_coverage")
    pd.DataFrame(validation_audit).to_json(
        data_dir / "core_validation_audit.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    flow, contributors, capacity, concentration = monthly_ecosystem_tables(
        artifacts, reviews, inline, collaborator_roles
    )
    period, state = period_context(flow, contributors, artifacts)
    contributor_roles = contributor_role_period_table(artifacts)
    capacity_means, capacity_comparison = capacity_period_comparison(
        flow, contributors, capacity
    )
    write_table(flow, tables / "ecosystem_monthly_flow")
    write_table(contributors, tables / "contributor_growth_monthly")
    write_table(capacity, tables / "review_capacity_monthly")
    write_table(period, tables / "ecosystem_period_monthly_means")
    write_table(state, tables / "artifact_state_at_cutoff")
    write_table(contributor_roles, tables / "ecosystem_pr_author_roles_by_period")
    write_table(capacity_means, tables / "ecosystem_capacity_period_means")
    write_table(capacity_comparison, tables / "ecosystem_2025_vs_2026_change")
    write_table(
        pd.DataFrame([concentration]), tables / "review_concentration_2026_jan_jul"
    )

    make_figures(
        flow,
        contributors,
        capacity,
        distributions,
        monthly_counts,
        joints,
        archetypes,
        core,
        taxonomy,
        figures,
    )

    legacy_baseline = json.loads(
        args.legacy_reconciliation.read_text(encoding="utf-8")
    )
    current_2026 = period[period["period"].eq("2026 Jan-Jul")].iloc[0]
    reconciliation = {
        "legacy_branch_commit": legacy_baseline.get("legacy_branch_commit"),
        "all_prs_opened_2026_monthly_mean": {
            "recomputed": float(current_2026["prs_opened_all_monthly_mean"]),
            "legacy": legacy_baseline.get(
                "all_prs_opened_2026_monthly_mean", {}
            ).get("legacy"),
        },
        "all_prs_merged_2026_monthly_mean": {
            "recomputed": float(current_2026["prs_merged_all_monthly_mean"]),
            "legacy": legacy_baseline.get(
                "all_prs_merged_2026_monthly_mean", {}
            ).get("legacy"),
        },
        "review_concentration": {
            "recomputed": concentration,
            "legacy": legacy_baseline.get("review_concentration", {}).get("legacy"),
        },
    }
    (data_dir / "legacy_reconciliation.json").write_text(
        json.dumps(reconciliation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    unknown_bound = len(tag_frame) - len(core)
    key_findings = {
        "deep_population": {
            "selected": len(tag_frame),
            "core_labeled": len(core),
            "missing_core": unknown_bound,
            "coverage": len(core) / len(tag_frame),
            "months": sorted(core["merged_month"].unique().tolist()),
        },
        "headline_distributions": {
            field: distributions[field].to_dict(orient="records")
            for field in CORE_FIELDS
        },
        "integration_shape": {
            "multi_scope_n": int(core["cross_scope"].sum()),
            "multi_scope_share": float(core["cross_scope"].mean()),
            "multi_component_n": int(core["cross_component"].sum()),
            "multi_component_share": float(core["cross_component"].mean()),
            "cross_backend_n": int(core["cross_backend"].sum()),
            "cross_backend_share": float(core["cross_backend"].mean()),
        },
        "missing_label_sensitivity": {
            "missing_n": unknown_bound,
            "max_absolute_share_shift_if_all_missing_take_one_label": unknown_bound
            / len(tag_frame),
        },
        "ecosystem": {
            "period_monthly_means": period.to_dict(orient="records"),
            "capacity_comparison_2025_vs_2026": capacity_comparison.to_dict(
                orient="records"
            ),
            "state_at_cutoff": state.to_dict(orient="records"),
            "pr_author_roles_by_period": contributor_roles.to_dict(orient="records"),
            "review_concentration_2026_jan_jul": concentration,
        },
        "top_archetypes": archetypes.head(25).to_dict(orient="records"),
        "legacy_reconciliation": reconciliation,
    }
    summary_path = data_dir / "rq1_summary.json"
    summary_path.write_text(
        json.dumps(key_findings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "vllm_rq1_analysis_manifest.v1",
        "analysis_cutoff": CUTOFF.isoformat(),
        "deep_label_window": {
            "start_inclusive": DEEP_START.isoformat(),
            "end_exclusive": DEEP_END.isoformat(),
        },
        "inputs": {
            "database": {
                "path": public_path(args.database),
                "bytes": args.database.stat().st_size,
                "sha256": sha256_file(args.database),
            },
            "compact_tagging": {
                "path": public_path(args.compact),
                "bytes": args.compact.stat().st_size,
                "sha256": sha256_file(args.compact),
            },
            "taxonomy": {
                "path": public_path(args.taxonomy),
                "bytes": args.taxonomy.stat().st_size,
                "sha256": sha256_file(args.taxonomy),
                "taxonomy_id": taxonomy.get("taxonomy_id"),
            },
            "legacy_reconciliation_baseline": {
                "path": public_path(args.legacy_reconciliation),
                "bytes": args.legacy_reconciliation.stat().st_size,
                "sha256": sha256_file(args.legacy_reconciliation),
                "git_commit": reconciliation["legacy_branch_commit"],
            },
        },
        "populations": {
            "release_artifacts": len(artifacts),
            "selected_merged_prs": len(tag_frame),
            "rq1_core_labeled_prs": len(core),
        },
        "outputs": {
            "tables_csv": len(list(tables.glob("*.csv"))),
            "figures_png": len(list(figures.glob("*.png"))),
            "summary": public_path(summary_path),
        },
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "seaborn": sns.__version__,
        },
    }
    (manifests / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["populations"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
