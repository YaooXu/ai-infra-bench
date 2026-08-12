#!/usr/bin/env python3
"""Aggregate, privacy-preserving analysis of the vLLM GitHub snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


CUTOFF = pd.Timestamp("2026-05-18 20:02:21")
EXPECTED_SHA256 = "1992a9f7011ebe35ba6f62511d5ccc727b233e21d7279db3d3496f9f4892c44d"
PERIOD_ORDER = ["Launch–2024", "2025", "2026 to May 18"]
COLORS = {
    "navy": "#16324F",
    "blue": "#2E86AB",
    "cyan": "#69B3C5",
    "orange": "#F18F01",
    "red": "#C73E1D",
    "green": "#3A7D44",
    "gray": "#73808D",
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def read(conn: sqlite3.Connection, table: str, columns: str = "*") -> pd.DataFrame:
    return pd.read_sql_query(f'SELECT {columns} FROM "{table}"', conn)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dates(frame: pd.DataFrame, *columns: str) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")


def period(series: pd.Series) -> pd.Categorical:
    values = np.select(
        [series < pd.Timestamp("2025-01-01"), series < pd.Timestamp("2026-01-01")],
        PERIOD_ORDER[:2],
        default=PERIOD_ORDER[2],
    )
    return pd.Categorical(values, categories=PERIOD_ORDER, ordered=True)


def wilson(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float, float]:
    if total == 0:
        return (float("nan"), float("nan"), float("nan"))
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return p, center - margin, center + margin


def gini(values: pd.Series) -> float:
    array = np.sort(values.to_numpy(dtype=float))
    if len(array) == 0 or array.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, len(array) + 1) - len(array) - 1) @ array / (len(array) * array.sum()))


def km_curve(duration: pd.Series, event: pd.Series) -> pd.DataFrame:
    data = pd.DataFrame({"time": duration, "event": event}).dropna().sort_values("time")
    grouped = data.groupby("time")["event"].agg(["count", "sum"]).rename(columns={"sum": "events"})
    at_risk = len(data)
    survival = 1.0
    rows = [{"time": 0.0, "survival": 1.0}]
    for time, row in grouped.iterrows():
        if at_risk > 0:
            survival *= 1 - row["events"] / at_risk
        rows.append({"time": float(time), "survival": float(survival)})
        at_risk -= int(row["count"])
    return pd.DataFrame(rows)


def competing_curve(duration: pd.Series, outcome: pd.Series) -> pd.DataFrame:
    data = pd.DataFrame({"time": duration, "outcome": outcome}).dropna(subset=["time"]).sort_values("time")
    grouped = data.groupby("time")["outcome"].agg(
        count="count",
        merged=lambda x: int((x == "merged").sum()),
        closed=lambda x: int((x == "closed_unmerged").sum()),
    )
    at_risk = len(data)
    survival = 1.0
    cif_merged = 0.0
    cif_closed = 0.0
    rows = [{"time": 0.0, "cif_merged": 0.0, "cif_closed_unmerged": 0.0}]
    for time, row in grouped.iterrows():
        if at_risk > 0:
            cif_merged += survival * row["merged"] / at_risk
            cif_closed += survival * row["closed"] / at_risk
            survival *= 1 - (row["merged"] + row["closed"]) / at_risk
        rows.append({"time": float(time), "cif_merged": float(cif_merged), "cif_closed_unmerged": float(cif_closed)})
        at_risk -= int(row["count"])
    return pd.DataFrame(rows)


def first_tag(title: str) -> str:
    match = re.match(r"\s*\[([^]]+)\]", title or "")
    return re.sub(r"\s+", " ", match.group(1).strip().lower()) if match else ""


def issue_intent(title: str, labels: set[str]) -> str:
    tag = first_tag(title)
    labels = {x.lower() for x in labels}
    if tag.startswith("bug") or "bug" in labels:
        return "Bug/correctness"
    if tag in {"new model", "model"} or tag.startswith("feature") or {"feature request", "new-model"} & labels:
        return "Feature/model/backend request"
    if tag.startswith("usage") or tag in {"question", "help wanted"} or "usage" in labels:
        return "Usage/configuration"
    if tag.startswith("rfc") or tag in {"design", "discussion"} or "rfc" in labels:
        return "Design/RFC"
    if tag.startswith("perf") or "performance" in labels:
        return "Performance"
    if tag.startswith("ci") or "ci-failure" in labels:
        return "CI/infrastructure"
    if tag.startswith("install") or "installation" in labels:
        return "Installation/build"
    if tag.startswith("doc") or "documentation" in labels:
        return "Documentation/API"
    return "Other/tracking"


def pr_work_type(title: str, labels: set[str]) -> str:
    text = (title or "").lower()
    tag = first_tag(title)
    labels = {x.lower() for x in labels}
    if tag in {"doc", "docs", "documentation"} or "documentation" in labels:
        return "Documentation/API/UX"
    if tag in {"ci", "ci/build", "build", "release"} or "ci/build" in labels:
        return "CI/build/release"
    if tag.startswith("perf") or "performance" in labels or re.search(r"\b(optimi[sz]|throughput|latency|speedup)\w*", text):
        return "Performance/efficiency"
    if tag in {"test", "tests", "benchmark", "eval", "evaluation"} or re.search(r"\b(add|update|improve)\w*\s+(unit\s+|e2e\s+)?tests?\b", text):
        return "Test/evaluation"
    if "refactor" in tag or tag in {"cleanup", "v0 deprecation"} or re.search(r"\b(refactor|cleanup|simplif|deprecat|migrat)\w*", text):
        return "Refactor/maintainability"
    if tag in {"chore", "deps", "dependency", "dependencies"} or re.search(r"\b(bump|chore|dependenc)\w*", text):
        return "Dependency/chore"
    if tag.startswith(("bug", "fix")) or "bug" in labels or re.search(r"\b(fix|resolve|correct|prevent|avoid|restore)\w*", text):
        return "Bug/correctness"
    if tag in {"feature", "feat", "new model", "model"} or "new-model" in labels or re.search(r"\b(add|support|implement|enable|introduce)\w*", text):
        return "Feature/capability"
    return "Other/unclear"


def classification_evidence(title: str, labels: set[str], assigned: str, unresolved: str) -> str:
    """Coarse provenance for deterministic labels, not a calibrated confidence."""
    if assigned == unresolved:
        return "Unresolved/default"
    tag = first_tag(title)
    normalized_labels = {value.lower() for value in labels}
    explicit = {
        "Bug/correctness": tag.startswith(("bug", "fix")) or "bug" in normalized_labels,
        "Feature/model/backend request": tag in {"new model", "model"} or tag.startswith("feature") or bool({"feature request", "new-model"} & normalized_labels),
        "Usage/configuration": tag.startswith("usage") or tag in {"question", "help wanted"} or "usage" in normalized_labels,
        "Design/RFC": tag.startswith("rfc") or tag in {"design", "discussion"} or "rfc" in normalized_labels,
        "Performance": tag.startswith("perf") or "performance" in normalized_labels,
        "CI/infrastructure": tag.startswith("ci") or "ci-failure" in normalized_labels,
        "Installation/build": tag.startswith("install") or "installation" in normalized_labels,
        "Documentation/API": tag.startswith("doc") or "documentation" in normalized_labels,
        "Documentation/API/UX": tag in {"doc", "docs", "documentation"} or "documentation" in normalized_labels,
        "CI/build/release": tag in {"ci", "ci/build", "build", "release"} or "ci/build" in normalized_labels,
        "Performance/efficiency": tag.startswith("perf") or "performance" in normalized_labels,
        "Test/evaluation": tag in {"test", "tests", "benchmark", "eval", "evaluation"},
        "Refactor/maintainability": "refactor" in tag or tag in {"cleanup", "v0 deprecation"},
        "Dependency/chore": tag in {"chore", "deps", "dependency", "dependencies"},
        "Feature/capability": tag in {"feature", "feat", "new model", "model"} or "new-model" in normalized_labels,
    }.get(assigned, False)
    if explicit:
        return "Explicit title tag/current label"
    return "Lexical heuristic"


def flag_dimensions(title: str, labels: set[str], paths: str) -> tuple[dict[str, bool], dict[str, bool]]:
    blob = " ".join([title or "", " ".join(labels), paths or ""]).lower()
    subsystem_patterns = {
        "Model support": r"model_executor/models|/models/|tests/models|new-model",
        "Engine/scheduler": r"scheduler|vllm/engine|vllm/v1/core|engine_core",
        "Memory/KV cache": r"kv[_/-]|kv cache|cache_manager|block_manager",
        "Distributed serving": r"distributed|parallel|executor|ray|collective|multi[_ -]node",
        "Kernels/operators": r"csrc|kernel|/ops/|attention|quantization|fused_moe|triton|cutlass",
        "Frontend/API": r"entrypoints|frontend|openai|api_server|tool_parser|structured_output|reasoning",
        "Platform/build/CI": r"\.buildkite|docker|cmake|setup\.py|pyproject|requirements|ci/build",
        "Tests/evaluation": r"(^|[ /])tests?/|benchmark|evals?",
        "Documentation": r"(^|[ /])docs?/|\.md\b|\.rst\b|documentation",
    }
    hardware_patterns = {
        "NVIDIA/CUDA": r"cuda|nvidia|cutlass",
        "AMD/ROCm": r"rocm|\bamd\b|aiter|quark",
        "Intel/XPU": r"\bxpu\b|intel[-_ ]gpu",
        "TPU": r"\btpu\b",
        "CPU": r"\bcpu\b",
        "Ascend/NPU": r"ascend|\bnpu\b",
        "MLU": r"\bmlu\b",
    }
    subsystems = {name: bool(re.search(pattern, blob)) for name, pattern in subsystem_patterns.items()}
    hardware = {name: bool(re.search(pattern, blob)) for name, pattern in hardware_patterns.items()}
    hardware["Cross-backend"] = sum(hardware.values()) >= 2
    hardware["Hardware-independent"] = not any(hardware.values())
    return subsystems, hardware


def flag_topics(title: str, labels: set[str], paths: str) -> dict[str, bool]:
    """Multi-label vLLM topic signals used to describe recent engineering demand."""
    blob = " ".join([title or "", " ".join(labels), paths or ""]).lower()
    patterns = {
        "Attention and kernels": r"attention|flash[_ -]?attn|flashinfer|kernel|triton|cutlass|csrc|/ops/",
        "Distributed and parallelism": r"distributed|parallel|multi[_ -]?node|collective|all[_ -]?reduce|ray|executor",
        "Frontend, serving, and APIs": r"entrypoints|api_server|frontend|openai|serving|chat_template",
        "KV cache, connectors, and offload": r"kv[_ /-]?cache|kv[_ /-]?connector|cache_manager|block_manager|offload|lmcache",
        "LoRA and adapters": r"\blora\b|adapter|prompt[_ -]?adapter",
        "MoE and expert parallelism": r"\bmoe\b|fused_moe|expert[_ -]?parallel|eplb|expert[_ -]?load",
        "Model support": r"model_executor/models|/models/|tests/models|new-model|model support",
        "Multimodal and audio": r"multimodal|multi[_ -]?modal|\bvlm\b|vision|image|audio|video",
        "Quantization and low precision": r"quantization|quantized|\bfp8\b|\bfp4\b|\bint8\b|\bint4\b|awq|gptq|bitsandbytes|compressed_tensors",
        "Speculative decoding": r"speculative|spec[_ -]?decod|draft[_ -]?model|\bmtp\b|medusa|eagle",
        "Structured output, tools, reasoning": r"structured[_ -]?output|guided[_ -]?decod|tool[_ -]?(call|parser)|reasoning[_ -]?parser|grammar",
        "torch.compile and CUDA graphs": r"torch[._ -]?compile|dynamo|inductor|cuda[_ -]?graph|cudagraph|vllm[_ -]?ir",
        "V1 engine and model runner": r"vllm/v1|\bv1 engine\b|model[_ -]?runner|gpu[_ -]?model[_ -]?runner",
        "Disaggregated serving": r"disaggregat|prefill[_ -]?decode|p[_ -]?d separation|remote[_ -]?prefill",
    }
    return {name: bool(re.search(pattern, blob)) for name, pattern in patterns.items()}


def quantile(series: pd.Series, value: float) -> float:
    clean = series.dropna()
    return float(clean.quantile(value)) if len(clean) else float("nan")


def json_safe(value):
    """Convert pandas/numpy scalars and missing values to strict JSON values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def label_sets(issue_labels: pd.DataFrame, labels: pd.DataFrame) -> dict[int, set[str]]:
    joined = issue_labels.merge(labels, left_on="label_id", right_on="id", how="left")
    return joined.groupby("issue_id")["name"].agg(lambda x: set(x.dropna())).to_dict()


def first_response(
    artifacts: pd.DataFrame,
    events: pd.DataFrame,
    actor_ids: set[int] | None = None,
) -> pd.Series:
    subset = events.merge(artifacts[["id", "user_id", "at_risk_at"]], left_on="issue_id", right_on="id", how="inner")
    subset = subset[(subset["user_id_x"] != subset["user_id_y"]) & (subset["at"] >= subset["at_risk_at"])]
    if actor_ids is not None:
        subset = subset[subset["user_id_x"].isin(actor_ids)]
    return subset.groupby("issue_id")["at"].min()


def fixed_horizon_rows(frame: pd.DataFrame, duration: str, event: str, artifact: str) -> list[dict]:
    rows = []
    for label, group in frame.groupby("period", observed=False):
        for horizon in (2, 7, 14, 30):
            eligible = group[group["age_days"] >= horizon]
            successes = int(((eligible[event]) & (eligible[duration] <= horizon)).sum())
            rate, low, high = wilson(successes, len(eligible))
            rows.append({
                "artifact": artifact,
                "period": str(label),
                "horizon_days": horizon,
                "eligible": len(eligible),
                "responded": successes,
                "rate": rate,
                "ci_low": low,
                "ci_high": high,
            })
    return rows


def backlog_series(artifacts: pd.DataFrame, history: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    months = list(pd.period_range(artifacts["created_at"].min(), cutoff, freq="M"))
    rows = []
    for month in months:
        end = min(month.end_time.floor("s"), cutoff)
        start = month.start_time
        available = artifacts[artifacts["created_at"] <= end]
        hist = history[history["updated_at"] <= end].sort_values(["updated_at", "closed"]).drop_duplicates("issue_id", keep="last")
        states = hist.set_index("issue_id")["closed"]
        closed = available["id"].map(states)
        fallback = available["closed_at"].notna() & (available["closed_at"] <= end)
        is_closed = closed.fillna(fallback.astype(int)).astype(bool)
        if end == cutoff:
            is_closed = available["state"].eq("closed")
        rows.append({
            "month": start,
            "opened": int(((artifacts["created_at"] >= start) & (artifacts["created_at"] <= end)).sum()),
            "closed": int(((history["closed"] == 1) & (history["updated_at"] >= start) & (history["updated_at"] <= end)).groupby(history["issue_id"]).any().sum()),
            "backlog": int((~is_closed).sum()),
        })
    return pd.DataFrame(rows)


def aggregate_mix(frame: pd.DataFrame, dimension: str, artifact: str) -> pd.DataFrame:
    grouped = frame.groupby(["period", dimension], observed=False).size().rename("count").reset_index()
    grouped["share"] = grouped["count"] / grouped.groupby("period", observed=False)["count"].transform("sum")
    grouped.insert(0, "artifact", artifact)
    return grouped


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def main() -> None:
    opt = args()
    opt.output.mkdir(parents=True, exist_ok=True)
    opt.figures.mkdir(parents=True, exist_ok=True)
    opt.summary.parent.mkdir(parents=True, exist_ok=True)
    snapshot_sha256 = sha256_file(opt.snapshot)
    if snapshot_sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"Snapshot SHA-256 mismatch: expected {EXPECTED_SHA256}, observed {snapshot_sha256}"
        )
    conn = sqlite3.connect(f"file:{opt.snapshot}?mode=ro", uri=True)

    users = read(conn, "user", "id, type")
    collaborators = read(conn, "repo_collaborator", "user_id, role_name, pull, triage, push, maintain, admin, _fivetran_deleted")
    all_issues = read(conn, "issue", "id, created_at, updated_at, number, state, state_reason, title, closed_at, pull_request, user_id")
    prs_raw = read(conn, "pull_request", "id, issue_id, created_at, closed_at, draft, merge_commit_sha, base_sha, head_sha")
    comments = read(conn, "issue_comment", "id, issue_id, created_at, user_id")
    reviews = read(conn, "pull_request_review", "id, pull_request_id, submitted_at, state, user_id, commit_sha")
    inline = read(conn, "pull_request_review_comments", "id, pull_request_id, pull_request_review_id, created_at, user_id, path")
    closed = read(conn, "issue_closed_history", "closed, issue_id, updated_at, actor_id")
    merged = read(conn, "issue_merged", "issue_id, merged_at, actor_id, commit_sha")
    ready = read(conn, "pull_request_ready_for_review_history", "created_at, pull_request_id, ready_for_review, actor_id")
    issue_labels = read(conn, "issue_label", "issue_id, label_id")
    labels = read(conn, "label", "id, name")
    commit_pr = read(conn, "commit_pull_request", "commit_sha, pull_request_id")
    commit_files = read(conn, "commit_file", "commit_sha, filename, additions, deletions, changes")
    issue_refs = read(conn, "issue_referenced", "issue_id, referenced_at, commit_sha, actor_id")
    label_history = read(conn, "issue_label_history", "issue_id, updated_at, actor_id, labeled")
    issue_assignees = read(conn, "issue_assignee", "issue_id, user_id")
    reviewer_requests = read(
        conn,
        "requested_reviewer_history",
        "created_at, pull_request_id, requested_id, actor_id, removed, requested_reviewer_type",
    )
    direct_main_commits = pd.read_sql_query(
        """
        SELECT strftime('%Y', c.committer_date) AS year,
               COUNT(DISTINCT b.commit_sha) AS commits
        FROM branch_commit_relation b
        JOIN "commit" c ON c.sha = b.commit_sha
        LEFT JOIN commit_pull_request cp ON cp.commit_sha = b.commit_sha
        WHERE b.branch_name = 'main' AND cp.commit_sha IS NULL
        GROUP BY 1 ORDER BY 1
        """,
        conn,
    )
    conn.close()

    for frame, columns in [
        (all_issues, ("created_at", "updated_at", "closed_at")),
        (prs_raw, ("created_at", "closed_at")),
        (comments, ("created_at",)),
        (reviews, ("submitted_at",)),
        (inline, ("created_at",)),
        (closed, ("updated_at",)),
        (merged, ("merged_at",)),
        (ready, ("created_at",)),
        (issue_refs, ("referenced_at",)),
        (label_history, ("updated_at",)),
        (reviewer_requests, ("created_at",)),
    ]:
        dates(frame, *columns)

    pr_issue_ids = set(prs_raw["issue_id"])
    flag_conflicts = all_issues[(all_issues["id"].isin(pr_issue_ids)) != (all_issues["pull_request"] == 1)]
    issues = all_issues[~all_issues["id"].isin(pr_issue_ids)].copy()
    prs = prs_raw.merge(all_issues.add_prefix("issue_"), left_on="issue_id", right_on="issue_id", how="inner")
    prs = prs.rename(columns={"id": "pr_id", "issue_id": "id", "issue_user_id": "user_id", "issue_title": "title", "issue_state": "state", "issue_closed_at": "artifact_closed_at", "issue_created_at": "artifact_created_at"})
    prs["created_at"] = prs["artifact_created_at"].fillna(prs["created_at"])
    prs["closed_at"] = prs["artifact_closed_at"].fillna(prs["closed_at"])

    bot_ids = set(users.loc[users["type"] == "Bot", "id"])
    active_collabs = collaborators[(collaborators["_fivetran_deleted"] == 0) & (collaborators["triage"] == 1)]
    collab_ids = set(active_collabs["user_id"])
    write_ids = set(active_collabs.loc[active_collabs["push"] == 1, "user_id"])
    human_ids = set(users.loc[users["type"] == "User", "id"])

    labels_by_issue = label_sets(issue_labels, labels)
    issues["period"] = period(issues["created_at"])
    issues["month"] = issues["created_at"].dt.to_period("M").dt.to_timestamp()
    issues["intent"] = [issue_intent(t, labels_by_issue.get(i, set())) for i, t in zip(issues["id"], issues["title"])]
    issues["classification_evidence"] = [
        classification_evidence(t, labels_by_issue.get(i, set()), assigned, "Other/tracking")
        for i, t, assigned in zip(issues["id"], issues["title"], issues["intent"])
    ]
    issues["author_role"] = np.select([issues["user_id"].isin(bot_ids), issues["user_id"].isin(collab_ids)], ["Bot", "Snapshot collaborator"], default="External human")
    issues["at_risk_at"] = issues["created_at"]

    comments_human = comments[comments["user_id"].isin(human_ids)].rename(columns={"created_at": "at"})[["issue_id", "at", "user_id"]]
    first_issue_human = first_response(issues, comments_human)
    first_issue_collab = first_response(issues, comments_human, collab_ids)
    issues["first_human_at"] = issues["id"].map(first_issue_human)
    issues["first_collab_at"] = issues["id"].map(first_issue_collab)
    issues["age_days"] = (CUTOFF - issues["created_at"]).dt.total_seconds() / 86400
    issues["human_days"] = (issues["first_human_at"] - issues["created_at"]).dt.total_seconds() / 86400
    issues["collab_days"] = (issues["first_collab_at"] - issues["created_at"]).dt.total_seconds() / 86400
    issues["human_event"] = issues["first_human_at"].notna()
    issues["collab_event"] = issues["first_collab_at"].notna()

    first_ready = (
        ready[ready["ready_for_review"] == 1]
        .sort_values("created_at")
        .drop_duplicates("pull_request_id", keep="first")
        .set_index("pull_request_id")
    )
    prs["at_risk_at"] = prs["created_at"]
    has_ready = prs["pr_id"].isin(first_ready.index)
    prs.loc[has_ready, "at_risk_at"] = prs.loc[has_ready, "pr_id"].map(first_ready["created_at"])
    never_ready = (~has_ready) & prs["draft"].eq(1)
    prs.loc[never_ready, "at_risk_at"] = pd.NaT
    prs["period"] = period(prs["created_at"])
    prs["month"] = prs["created_at"].dt.to_period("M").dt.to_timestamp()
    prs["work_type"] = [pr_work_type(t, labels_by_issue.get(i, set())) for i, t in zip(prs["id"], prs["title"])]
    prs["classification_evidence"] = [
        classification_evidence(t, labels_by_issue.get(i, set()), assigned, "Other/unclear")
        for i, t, assigned in zip(prs["id"], prs["title"], prs["work_type"])
    ]
    prs["author_role"] = np.select([prs["user_id"].isin(bot_ids), prs["user_id"].isin(collab_ids)], ["Bot", "Snapshot collaborator"], default="External human")

    pr_map = prs.set_index("pr_id")["id"]
    author_by_issue = all_issues.set_index("id")["user_id"]
    pr_comment_events = comments_human[comments_human["issue_id"].isin(pr_issue_ids)]
    review_events = reviews[reviews["user_id"].isin(human_ids)].assign(issue_id=lambda x: x["pull_request_id"].map(pr_map), at=lambda x: x["submitted_at"])[["issue_id", "at", "user_id"]]
    inline_events = inline[inline["user_id"].isin(human_ids)].assign(issue_id=lambda x: x["pull_request_id"].map(pr_map), at=lambda x: x["created_at"])[["issue_id", "at", "user_id"]]
    pr_response_events = pd.concat([pr_comment_events, review_events, inline_events], ignore_index=True).dropna(subset=["issue_id", "at"])
    ready_prs = prs[prs["at_risk_at"].notna()].copy()
    first_pr_human = first_response(ready_prs, pr_response_events)
    first_pr_collab = first_response(ready_prs, pr_response_events, collab_ids)
    prs["first_human_at"] = prs["id"].map(first_pr_human)
    prs["first_collab_at"] = prs["id"].map(first_pr_collab)
    prs["age_days"] = (CUTOFF - prs["at_risk_at"]).dt.total_seconds() / 86400
    prs["human_days"] = (prs["first_human_at"] - prs["at_risk_at"]).dt.total_seconds() / 86400
    prs["collab_days"] = (prs["first_collab_at"] - prs["at_risk_at"]).dt.total_seconds() / 86400
    prs["human_event"] = prs["first_human_at"].notna()
    prs["collab_event"] = prs["first_collab_at"].notna()
    merge_by_issue = merged.sort_values("merged_at").drop_duplicates("issue_id", keep="last").set_index("issue_id")
    prs["merged_at"] = prs["id"].map(merge_by_issue["merged_at"])
    prs["merged"] = prs["merged_at"].notna()
    prs["closed_unmerged"] = prs["closed_at"].notna() & ~prs["merged"]
    prs["merge_days"] = (prs["merged_at"] - prs["at_risk_at"]).dt.total_seconds() / 86400
    prs["close_days"] = (prs["closed_at"] - prs["at_risk_at"]).dt.total_seconds() / 86400

    review_human = reviews[reviews["user_id"].isin(human_ids)].copy()
    review_human["issue_id"] = review_human["pull_request_id"].map(pr_map)
    review_human = review_human[
        review_human["issue_id"].notna()
        & review_human["user_id"].ne(review_human["issue_id"].map(author_by_issue))
    ].copy()
    review_human["state"] = review_human["state"].str.upper()
    review_human["is_collab"] = review_human["user_id"].isin(collab_ids)
    review_agg = review_human.groupby("issue_id").agg(
        human_reviews=("id", "count"),
        unique_reviewers=("user_id", "nunique"),
        change_requests=("state", lambda x: int((x == "CHANGES_REQUESTED").sum())),
    )
    collab_review_agg = review_human[review_human["is_collab"]].groupby("issue_id").agg(
        collab_reviews=("id", "count"),
        review_rounds=("commit_sha", "nunique"),
        first_review_at=("submitted_at", "min"),
        last_review_at=("submitted_at", "max"),
    )
    inline_human = inline[inline["user_id"].isin(human_ids)].copy()
    inline_human["issue_id"] = inline_human["pull_request_id"].map(pr_map)
    inline_human = inline_human[
        inline_human["issue_id"].notna()
        & inline_human["user_id"].ne(inline_human["issue_id"].map(author_by_issue))
    ].copy()
    inline_human["is_collab"] = inline_human["user_id"].isin(collab_ids)
    inline_collab_agg = inline_human[inline_human["is_collab"]].groupby("issue_id").agg(
        collab_inline_comments=("id", "count"),
        inline_reviewers=("user_id", "nunique"),
    )
    prs = prs.join(review_agg, on="id").join(collab_review_agg, on="id").join(inline_collab_agg, on="id")
    for column in ["human_reviews", "unique_reviewers", "change_requests", "collab_reviews", "review_rounds", "collab_inline_comments", "inline_reviewers"]:
        prs[column] = prs[column].fillna(0).astype(int)
    prs["review_span_days"] = (prs["last_review_at"] - prs["first_review_at"]).dt.total_seconds() / 86400
    prs["first_submitted_collab_review_days"] = (prs["first_review_at"] - prs["at_risk_at"]).dt.total_seconds() / 86400
    prs["submitted_collab_review_event"] = prs["first_review_at"].notna()

    file_rows = commit_pr.merge(commit_files, on="commit_sha", how="inner").drop_duplicates(["pull_request_id", "commit_sha", "filename"])
    file_rows["issue_id"] = file_rows["pull_request_id"].map(pr_map)
    file_agg = file_rows.groupby("issue_id").agg(
        commits=("commit_sha", "nunique"),
        files=("filename", "nunique"),
        cumulative_additions=("additions", "sum"),
        cumulative_deletions=("deletions", "sum"),
        cumulative_churn=("changes", "sum"),
        paths=("filename", lambda x: " ".join(sorted(set(x)))),
    )
    prs = prs.join(file_agg, on="id")
    for column in ["commits", "files", "cumulative_additions", "cumulative_deletions", "cumulative_churn"]:
        prs[column] = prs[column].fillna(0).astype(int)
    prs["paths"] = prs["paths"].fillna("")
    prs["test_touched"] = prs["paths"].str.contains(r"(?:^| )tests?/|test_", regex=True)
    prs["benchmark_touched"] = prs["paths"].str.contains(r"benchmark|eval", case=False, regex=True)
    path_tokens = prs["paths"].str.split()
    prs["docs_only"] = path_tokens.map(lambda values: bool(values) and all(v.startswith("docs/") or v.endswith((".md", ".rst")) for v in values))
    prs["size_bin"] = pd.cut(prs["cumulative_churn"], [-1, 20, 100, 500, 2000, np.inf], labels=["≤20", "21–100", "101–500", "501–2,000", ">2,000"])

    subsystem_rows, hardware_rows, topic_rows = [], [], []
    for row in prs[["id", "title", "paths"]].itertuples(index=False):
        subsystems, hardware = flag_dimensions(row.title, labels_by_issue.get(row.id, set()), row.paths)
        subsystem_rows.append(subsystems)
        hardware_rows.append(hardware)
        topic_rows.append(flag_topics(row.title, labels_by_issue.get(row.id, set()), row.paths))
    subsystem_flags = pd.DataFrame(subsystem_rows, index=prs.index)
    hardware_flags = pd.DataFrame(hardware_rows, index=prs.index)
    topic_flags = pd.DataFrame(topic_rows, index=prs.index)
    prs = pd.concat(
        [
            prs,
            subsystem_flags.add_prefix("subsystem__"),
            hardware_flags.add_prefix("hardware__"),
            topic_flags.add_prefix("topic__"),
        ],
        axis=1,
    )
    prs["hardware_specific"] = ~prs["hardware__Hardware-independent"]
    prs["large_change"] = (prs["files"] > 20) | (prs["cumulative_churn"] > 1000) | (prs["commits"] > 20)
    prs["review_intensive"] = (prs["review_rounds"] >= 3) | (prs["collab_reviews"] >= 10) | (prs["review_span_days"] >= 14)

    linked = issue_refs.merge(commit_pr, on="commit_sha", how="inner").merge(prs[["pr_id", "id", "merged"]], left_on="pull_request_id", right_on="pr_id", how="inner")
    linked = linked[linked["issue_id"] != linked["id"]]
    linked_summary = linked.groupby("issue_id").agg(linked_prs=("pr_id", "nunique"), linked_merged_pr=("merged", "max"))
    issues = issues.join(linked_summary, on="id")
    issues["linked_prs"] = issues["linked_prs"].fillna(0).astype(int)
    issues["linked_merged_pr"] = issues["linked_merged_pr"].fillna(False).astype(bool)

    issue_backlog = backlog_series(issues, closed[closed["issue_id"].isin(issues["id"])], CUTOFF).rename(columns={"opened": "issue_opened", "closed": "issue_closed", "backlog": "issue_backlog"})
    pr_backlog = backlog_series(prs.rename(columns={"artifact_closed_at": "unused"}), closed[closed["issue_id"].isin(prs["id"])], CUTOFF).rename(columns={"opened": "pr_opened", "closed": "pr_closed", "backlog": "pr_backlog"})
    monthly = issue_backlog.merge(pr_backlog, on="month", how="outer").sort_values("month")
    monthly_merges = merged.assign(month=lambda x: x["merged_at"].dt.to_period("M").dt.to_timestamp()).groupby("month").size()
    monthly["pr_merged"] = monthly["month"].map(monthly_merges).fillna(0).astype(int)

    review_human["month"] = review_human["submitted_at"].dt.to_period("M").dt.to_timestamp()
    collab_reviews = review_human[review_human["is_collab"]]
    monthly["active_collab_reviewers"] = monthly["month"].map(collab_reviews.groupby("month")["user_id"].nunique()).fillna(0).astype(int)
    monthly["collab_review_submissions"] = monthly["month"].map(collab_reviews.groupby("month").size()).fillna(0).astype(int)
    reviewer_days = (
        collab_reviews.assign(review_date=collab_reviews["submitted_at"].dt.date)
        .groupby(["month", "user_id"])["review_date"]
        .nunique()
        .groupby("month")
        .sum()
    )
    monthly["collab_reviewer_active_days"] = monthly["month"].map(reviewer_days).fillna(0).astype(int)
    monthly["pr_arrivals_per_active_reviewer"] = monthly["pr_opened"] / monthly["active_collab_reviewers"].replace(0, np.nan)
    collab_issue_comments = comments[(comments["issue_id"].isin(set(issues["id"]))) & (comments["user_id"].isin(collab_ids))].copy()
    collab_issue_comments = collab_issue_comments[
        collab_issue_comments["user_id"].ne(collab_issue_comments["issue_id"].map(author_by_issue))
    ].copy()
    collab_issue_comments["month"] = collab_issue_comments["created_at"].dt.to_period("M").dt.to_timestamp()
    monthly["active_collab_issue_commenters"] = monthly["month"].map(collab_issue_comments.groupby("month")["user_id"].nunique()).fillna(0).astype(int)
    monthly["collab_issue_comments"] = monthly["month"].map(collab_issue_comments.groupby("month").size()).fillna(0).astype(int)

    for artifact_name, frame in [("issue", issues), ("pr", prs[prs["at_risk_at"].notna()])]:
        cohort = frame[frame["age_days"] >= 7].copy()
        cohort["collaborator_response_within_7d"] = cohort["collab_event"] & (cohort["collab_days"] <= 7)
        rates = cohort.groupby("month", observed=False)["collaborator_response_within_7d"].mean()
        monthly[f"{artifact_name}_collaborator_response_7d"] = monthly["month"].map(rates)

    response_rows = fixed_horizon_rows(issues, "human_days", "human_event", "Issue: any human")
    response_rows += fixed_horizon_rows(issues, "collab_days", "collab_event", "Issue: snapshot collaborator")
    response_rows += fixed_horizon_rows(prs[prs["at_risk_at"].notna()], "human_days", "human_event", "PR: any human")
    response_rows += fixed_horizon_rows(prs[prs["at_risk_at"].notna()], "collab_days", "collab_event", "PR: snapshot collaborator")
    response_rows += fixed_horizon_rows(
        prs[prs["at_risk_at"].notna()],
        "first_submitted_collab_review_days",
        "submitted_collab_review_event",
        "PR: submitted snapshot-collaborator review",
    )
    response = pd.DataFrame(response_rows)
    response_role_rows = []
    for artifact_name, frame in [("Issue", issues), ("PR", prs[prs["at_risk_at"].notna()])]:
        for (label, author_role), group in frame.groupby(["period", "author_role"], observed=False):
            eligible = group[group["age_days"] >= 7]
            response_role_rows.append({
                "artifact": artifact_name,
                "period": str(label),
                "author_role": author_role,
                "eligible_7d": len(eligible),
                "any_human_response_7d_pct": float((eligible["human_event"] & (eligible["human_days"] <= 7)).mean()) if len(eligible) else np.nan,
                "snapshot_collaborator_response_7d_pct": float((eligible["collab_event"] & (eligible["collab_days"] <= 7)).mean()) if len(eligible) else np.nan,
            })
    response_by_author_role = pd.DataFrame(response_role_rows)

    issue_mix = aggregate_mix(issues, "intent", "Issue")
    pr_mix = aggregate_mix(prs, "work_type", "PR")
    workload_mix = pd.concat([issue_mix, pr_mix], ignore_index=True)
    classification_coverage = pd.concat(
        [
            aggregate_mix(issues, "classification_evidence", "Issue").rename(columns={"classification_evidence": "evidence"}),
            aggregate_mix(prs, "classification_evidence", "PR").rename(columns={"classification_evidence": "evidence"}),
        ],
        ignore_index=True,
    )

    dimension_rows = []
    for prefix, dimension in [("subsystem__", "subsystem"), ("hardware__", "hardware")]:
        for column in [c for c in prs if c.startswith(prefix)]:
            name = column.removeprefix(prefix)
            for label, group in prs.groupby("period", observed=False):
                dimension_rows.append({"dimension": dimension, "name": name, "period": str(label), "count": int(group[column].sum()), "share": float(group[column].mean())})
    dimensions = pd.DataFrame(dimension_rows)

    topic_rows = []
    for column in [c for c in prs if c.startswith("topic__")]:
        name = column.removeprefix("topic__")
        for label, group in prs.groupby("period", observed=False):
            topic_rows.append({
                "topic": name,
                "period": str(label),
                "prs": int(group[column].sum()),
                "share": float(group[column].mean()),
            })
    topics = pd.DataFrame(topic_rows)

    period_rows = []
    for label in PERIOD_ORDER:
        issue_group = issues[issues["period"] == label]
        pr_group = prs[prs["period"] == label]
        covered_pr_group = pr_group[pr_group["commits"] > 0]
        period_rows.append({
            "period": label,
            "issues": len(issue_group),
            "issues_open_at_snapshot": int((issue_group["state"] == "open").sum()),
            "external_issue_share": float((issue_group["author_role"] == "External human").mean()),
            "issues_linked_to_any_pr_pct": float((issue_group["linked_prs"] > 0).mean()),
            "issues_linked_to_merged_pr_pct": float(issue_group["linked_merged_pr"].mean()),
            "prs": len(pr_group),
            "prs_merged": int(pr_group["merged"].sum()),
            "prs_closed_unmerged": int(pr_group["closed_unmerged"].sum()),
            "prs_open_at_snapshot": int((pr_group["state"] == "open").sum()),
            "external_pr_share": float((pr_group["author_role"] == "External human").mean()),
            "commit_data_coverage_share": float((pr_group["commits"] > 0).mean()),
            "test_touched_among_commit_data_share": float(covered_pr_group["test_touched"].mean()) if len(covered_pr_group) else np.nan,
            "hardware_specific_share": float(pr_group["hardware_specific"].mean()),
            "large_change_among_commit_data_share": float(covered_pr_group["large_change"].mean()) if len(covered_pr_group) else np.nan,
            "review_intensive_share": float(pr_group["review_intensive"].mean()),
        })
    period_summary = pd.DataFrame(period_rows)

    concentration_rows = []
    for label in PERIOD_ORDER:
        start, end = {
            "Launch–2024": (pd.Timestamp("2023-02-09"), pd.Timestamp("2024-12-31 23:59:59")),
            "2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31 23:59:59")),
            "2026 to May 18": (pd.Timestamp("2026-01-01"), CUTOFF),
        }[label]
        actions = collab_reviews[(collab_reviews["submitted_at"] >= start) & (collab_reviews["submitted_at"] <= end)].groupby("user_id").size().sort_values(ascending=False)
        concentration_rows.append({
            "period": label,
            "active_snapshot_collab_reviewers": len(actions),
            "review_submissions": int(actions.sum()),
            "top_5_share": float(actions.head(5).sum() / actions.sum()) if actions.sum() else np.nan,
            "gini": gini(actions),
        })
    review_concentration = pd.DataFrame(concentration_rows)

    feasibility = prs[(prs["merged"]) & (prs["author_role"] != "Bot") & (prs["commits"] > 0)].copy()
    feasibility_rows = []
    for label, group in feasibility.groupby("period", observed=False):
        feasibility_rows.append({
            "period": str(label),
            "merged_human_code_prs": len(group),
            "test_touched_pct": float(group["test_touched"].mean()),
            "docs_only_pct": float(group["docs_only"].mean()),
            "hardware_specific_pct": float(group["hardware_specific"].mean()),
            "performance_pct": float((group["work_type"] == "Performance/efficiency").mean()),
            "large_change_pct": float(group["large_change"].mean()),
            "review_intensive_pct": float(group["review_intensive"].mean()),
            "commit_referenced_issue_pct": float(group["pr_id"].isin(set(linked["pr_id"])).mean()),
        })
    feasibility_summary = pd.DataFrame(feasibility_rows)
    benchmark_work_type = feasibility.groupby(["period", "work_type"], observed=False).agg(
        prs=("id", "size"),
        test_touched_pct=("test_touched", "mean"),
        benchmark_touched_pct=("benchmark_touched", "mean"),
        hardware_specific_pct=("hardware_specific", "mean"),
        large_change_pct=("large_change", "mean"),
        review_intensive_pct=("review_intensive", "mean"),
        median_files=("files", "median"),
        median_cumulative_churn=("cumulative_churn", "median"),
    ).reset_index()
    benchmark_work_type["share"] = benchmark_work_type["prs"] / benchmark_work_type.groupby("period", observed=False)["prs"].transform("sum")
    benchmark_dimension_rows = []
    for prefix, dimension in [("subsystem__", "subsystem"), ("hardware__", "hardware")]:
        for column in [c for c in feasibility if c.startswith(prefix)]:
            name = column.removeprefix(prefix)
            for label, group in feasibility.groupby("period", observed=False):
                benchmark_dimension_rows.append({
                    "dimension": dimension,
                    "name": name,
                    "period": str(label),
                    "prs": int(group[column].sum()),
                    "share": float(group[column].mean()),
                })
    benchmark_dimensions = pd.DataFrame(benchmark_dimension_rows)

    issue_outcome_rows = []
    for (label, intent), group in issues.groupby(["period", "intent"], observed=False):
        eligible_7 = group[group["age_days"] >= 7]
        eligible_30 = group[group["age_days"] >= 30]
        issue_outcome_rows.append({
            "period": str(label),
            "intent": intent,
            "issues": len(group),
            "open_at_snapshot_pct": float((group["state"] == "open").mean()) if len(group) else np.nan,
            "human_response_7d_pct": float((eligible_7["human_event"] & (eligible_7["human_days"] <= 7)).mean()) if len(eligible_7) else np.nan,
            "collaborator_response_7d_pct": float((eligible_7["collab_event"] & (eligible_7["collab_days"] <= 7)).mean()) if len(eligible_7) else np.nan,
            "human_response_30d_pct": float((eligible_30["human_event"] & (eligible_30["human_days"] <= 30)).mean()) if len(eligible_30) else np.nan,
            "commit_referenced_by_pr_pct": float((group["linked_prs"] > 0).mean()) if len(group) else np.nan,
        })
    issue_outcomes = pd.DataFrame(issue_outcome_rows)

    pr_outcome_rows = []
    for (label, work_type), group in prs.groupby(["period", "work_type"], observed=False):
        eligible_7 = group[(group["age_days"] >= 7) & group["at_risk_at"].notna()]
        eligible_90 = group[(group["age_days"] >= 90) & group["at_risk_at"].notna()]
        covered = group[group["commits"] > 0]
        pr_outcome_rows.append({
            "period": str(label),
            "work_type": work_type,
            "prs": len(group),
            "merged_at_snapshot_pct": float(group["merged"].mean()) if len(group) else np.nan,
            "merged_within_90d_pct": float((eligible_90["merged"] & (eligible_90["merge_days"] <= 90)).mean()) if len(eligible_90) else np.nan,
            "closed_unmerged_within_90d_pct": float((eligible_90["closed_unmerged"] & (eligible_90["close_days"] <= 90)).mean()) if len(eligible_90) else np.nan,
            "collaborator_response_7d_pct": float((eligible_7["collab_event"] & (eligible_7["collab_days"] <= 7)).mean()) if len(eligible_7) else np.nan,
            "submitted_collaborator_review_7d_pct": float((eligible_7["submitted_collab_review_event"] & (eligible_7["first_submitted_collab_review_days"] <= 7)).mean()) if len(eligible_7) else np.nan,
            "commit_data_coverage_pct": float((group["commits"] > 0).mean()) if len(group) else np.nan,
            "median_cumulative_churn": float(covered["cumulative_churn"].median()) if len(covered) else np.nan,
            "median_files": float(covered["files"].median()) if len(covered) else np.nan,
            "test_touched_pct": float(covered["test_touched"].mean()) if len(covered) else np.nan,
            "review_intensive_pct": float(group["review_intensive"].mean()) if len(group) else np.nan,
            "hardware_specific_pct": float(group["hardware_specific"].mean()) if len(group) else np.nan,
        })
    pr_outcomes = pd.DataFrame(pr_outcome_rows)

    hardware_outcome_rows = []
    hardware_columns = [c for c in prs if c.startswith("hardware__") and c not in {"hardware__Hardware-independent", "hardware__Cross-backend"}]
    for hardware_column in hardware_columns:
        hardware_name = hardware_column.removeprefix("hardware__")
        for label, group in prs[prs[hardware_column]].groupby("period", observed=False):
            covered = group[group["commits"] > 0]
            hardware_outcome_rows.append({
                "period": str(label),
                "hardware": hardware_name,
                "prs": len(group),
                "commit_data_coverage_pct": float((group["commits"] > 0).mean()) if len(group) else np.nan,
                "test_touched_pct": float(covered["test_touched"].mean()) if len(covered) else np.nan,
                "large_change_pct": float(covered["large_change"].mean()) if len(covered) else np.nan,
                "review_intensive_pct": float(group["review_intensive"].mean()) if len(group) else np.nan,
            })
    hardware_outcomes = pd.DataFrame(hardware_outcome_rows)

    backlog_age_rows = []
    for artifact_name, frame in [("Issue", issues), ("PR", prs)]:
        open_frame = frame[frame["state"] == "open"].copy()
        open_frame["backlog_age_days"] = (CUTOFF - open_frame["created_at"]).dt.total_seconds() / 86400
        open_frame["age_bin"] = pd.cut(open_frame["backlog_age_days"], [-1, 30, 90, 180, np.inf], labels=["≤30 days", "31–90 days", "91–180 days", ">180 days"])
        for age_bin, group in open_frame.groupby("age_bin", observed=False):
            backlog_age_rows.append({"artifact": artifact_name, "age_bin": str(age_bin), "count": len(group), "share": len(group) / len(open_frame) if len(open_frame) else np.nan})
    backlog_age = pd.DataFrame(backlog_age_rows)

    # Maintainer-facing activity composition. These are event counts, not effort
    # estimates: a label edit and a substantive review are intentionally kept apart.
    action_frames = []

    def add_actions(frame: pd.DataFrame, at: str, actor: str, action: str) -> None:
        subset = frame[frame[actor].isin(collab_ids)][[at, actor]].dropna().copy()
        subset.columns = ["at", "actor_id"]
        subset["action"] = action
        action_frames.append(subset)

    issue_comment_rows = comments[comments["issue_id"].isin(set(issues["id"]))]
    pr_comment_rows = comments[comments["issue_id"].isin(pr_issue_ids)]
    issue_comment_rows = issue_comment_rows[
        issue_comment_rows["user_id"].ne(issue_comment_rows["issue_id"].map(author_by_issue))
    ]
    pr_comment_rows = pr_comment_rows[
        pr_comment_rows["user_id"].ne(pr_comment_rows["issue_id"].map(author_by_issue))
    ]
    add_actions(issue_comment_rows, "created_at", "user_id", "Issue conversation comment")
    add_actions(pr_comment_rows, "created_at", "user_id", "PR conversation comment")
    add_actions(review_human, "submitted_at", "user_id", "Submitted review")
    add_actions(inline_human, "created_at", "user_id", "Inline review comment")
    add_actions(merged, "merged_at", "actor_id", "Merge")
    add_actions(label_history, "updated_at", "actor_id", "Label change")
    add_actions(closed, "updated_at", "actor_id", "Close or reopen")
    maintainer_events = pd.concat(action_frames, ignore_index=True)
    maintainer_events["month"] = maintainer_events["at"].dt.to_period("M").dt.to_timestamp()
    maintainer_events["period"] = period(maintainer_events["at"])
    maintainer_events["active_date"] = maintainer_events["at"].dt.date
    maintainer_actions = maintainer_events.groupby(["period", "action"], observed=False).agg(
        events=("actor_id", "size"), active_snapshot_collaborators=("actor_id", "nunique")
    ).reset_index()
    action_concentration_rows = []
    for (label, action), group in maintainer_events.groupby(["period", "action"], observed=False):
        actor_counts = group.groupby("actor_id").size().sort_values(ascending=False)
        action_concentration_rows.append({
            "period": str(label), "action": action, "active_snapshot_collaborators": len(actor_counts),
            "events": int(actor_counts.sum()),
            "top_5_share": float(actor_counts.head(5).sum() / actor_counts.sum()) if actor_counts.sum() else np.nan,
            "gini": gini(actor_counts),
        })
    maintainer_action_concentration = pd.DataFrame(action_concentration_rows)
    actor_month = maintainer_events.groupby(["month", "actor_id"], observed=False).agg(
        events=("action", "size"), active_days=("active_date", "nunique")
    ).reset_index()
    capacity_sensitivity_rows = []
    for month, group in actor_month.groupby("month", observed=False):
        capacity_sensitivity_rows.append({
            "month": month,
            "active_at_least_one_event": len(group),
            "active_at_least_five_events": int((group["events"] >= 5).sum()),
            "active_at_least_three_days": int((group["active_days"] >= 3).sum()),
            "collaborator_active_days": int(group["active_days"].sum()),
            "median_active_days_per_collaborator": float(group["active_days"].median()),
            "median_events_per_active_collaborator": float(group["events"].median()),
            "p90_events_per_active_collaborator": quantile(group["events"], 0.90),
        })
    monthly_capacity_sensitivity = pd.DataFrame(capacity_sensitivity_rows)
    monthly_maintainer_actions = maintainer_events.groupby(["month", "action"], observed=False).agg(
        events=("actor_id", "size"), active_snapshot_collaborators=("actor_id", "nunique")
    ).reset_index()
    action_month_pivot = monthly_maintainer_actions.pivot(index="month", columns="action", values="events").fillna(0)
    action_comparison_rows = []
    for action in sorted(maintainer_events["action"].unique()):
        action_comparison_rows.append({
            "action": action,
            "2025_monthly_mean": float(action_month_pivot.loc["2025-01-01":"2025-12-01", action].mean()),
            "2026_jan_apr_monthly_mean": float(action_month_pivot.loc["2026-01-01":"2026-04-01", action].mean()),
        })
    maintainer_action_comparison = pd.DataFrame(action_comparison_rows)
    maintainer_action_comparison["relative_change"] = maintainer_action_comparison["2026_jan_apr_monthly_mean"] / maintainer_action_comparison["2025_monthly_mean"] - 1
    comparison_monthly = monthly.set_index("month").join(action_month_pivot, how="left").fillna(0)
    comparison_monthly["Submitted reviews per opened PR"] = comparison_monthly["Submitted review"] / comparison_monthly["pr_opened"].replace(0, np.nan)
    comparison_monthly["Inline comments per opened PR"] = comparison_monthly["Inline review comment"] / comparison_monthly["pr_opened"].replace(0, np.nan)
    comparison_monthly["Issue comments per opened issue"] = comparison_monthly["Issue conversation comment"] / comparison_monthly["issue_opened"].replace(0, np.nan)
    comparison_metrics = {
        "Issues opened": "issue_opened",
        "PRs opened": "pr_opened",
        "PRs merged": "pr_merged",
        "Active snapshot reviewers": "active_collab_reviewers",
        "Snapshot-collaborator reviewer-days": "collab_reviewer_active_days",
        "Submitted reviews": "Submitted review",
        "Inline review comments": "Inline review comment",
        "Issue conversation comments": "Issue conversation comment",
        "PR conversation comments": "PR conversation comment",
        "Submitted reviews per opened PR": "Submitted reviews per opened PR",
        "Inline comments per opened PR": "Inline comments per opened PR",
        "Issue comments per opened issue": "Issue comments per opened issue",
    }
    recent_comparison_rows = []
    baseline_months = comparison_monthly.loc["2025-01-01":"2025-12-01"]
    recent_months = comparison_monthly.loc["2026-01-01":"2026-04-01"]
    for metric, column in comparison_metrics.items():
        baseline_mean = float(baseline_months[column].mean())
        recent_mean = float(recent_months[column].mean())
        recent_comparison_rows.append({
            "metric": metric,
            "2025_monthly_mean": baseline_mean,
            "2026_jan_apr_monthly_mean": recent_mean,
            "relative_change": recent_mean / baseline_mean - 1 if baseline_mean else np.nan,
        })
    recent_capacity_comparison = pd.DataFrame(recent_comparison_rows)

    # Review burden by disposition and work type. Review events on non-merged work
    # are reported descriptively and are not assumed to be wasted.
    prs["snapshot_outcome"] = np.select(
        [prs["merged"], prs["closed_unmerged"]],
        ["Merged", "Closed unmerged"],
        default="Open",
    )
    review_burden_outcome_rows = []
    for (label, outcome), group in prs.groupby(["period", "snapshot_outcome"], observed=False):
        review_burden_outcome_rows.append({
            "period": str(label),
            "snapshot_outcome": outcome,
            "prs": len(group),
            "prs_with_collaborator_review_pct": float((group["collab_reviews"] > 0).mean()) if len(group) else np.nan,
            "collaborator_review_submissions": int(group["collab_reviews"].sum()),
            "inline_review_comments": int(group["collab_inline_comments"].sum()),
            "median_reviews_per_pr": float(group["collab_reviews"].median()) if len(group) else np.nan,
            "p90_reviews_per_pr": quantile(group["collab_reviews"], 0.90),
            "median_inline_comments_per_pr": float(group["collab_inline_comments"].median()) if len(group) else np.nan,
            "p90_inline_comments_per_pr": quantile(group["collab_inline_comments"], 0.90),
            "median_review_rounds": float(group["review_rounds"].median()) if len(group) else np.nan,
            "p90_review_rounds": quantile(group["review_rounds"], 0.90),
        })
    review_burden_outcome = pd.DataFrame(review_burden_outcome_rows)
    review_burden_outcome["review_submission_share"] = review_burden_outcome["collaborator_review_submissions"] / review_burden_outcome.groupby("period", observed=False)["collaborator_review_submissions"].transform("sum")
    review_burden_outcome["inline_comment_share"] = review_burden_outcome["inline_review_comments"] / review_burden_outcome.groupby("period", observed=False)["inline_review_comments"].transform("sum")

    review_burden_type_rows = []
    for label, period_group in prs.groupby("period", observed=False):
        total_reviews = period_group["collab_reviews"].sum()
        total_inline = period_group["collab_inline_comments"].sum()
        for work_type, group in period_group.groupby("work_type", observed=False):
            review_burden_type_rows.append({
                "period": str(label),
                "work_type": work_type,
                "prs": len(group),
                "pr_share": len(group) / len(period_group) if len(period_group) else np.nan,
                "collaborator_review_submissions": int(group["collab_reviews"].sum()),
                "review_submission_share": float(group["collab_reviews"].sum() / total_reviews) if total_reviews else np.nan,
                "inline_review_comments": int(group["collab_inline_comments"].sum()),
                "inline_comment_share": float(group["collab_inline_comments"].sum() / total_inline) if total_inline else np.nan,
                "reviewed_pr_pct": float((group["collab_reviews"] > 0).mean()) if len(group) else np.nan,
                "median_reviews_when_reviewed": float(group.loc[group["collab_reviews"] > 0, "collab_reviews"].median()) if (group["collab_reviews"] > 0).any() else np.nan,
                "p90_reviews_per_pr": quantile(group["collab_reviews"], 0.90),
            })
    review_burden_type = pd.DataFrame(review_burden_type_rows)

    assigned_issue_ids = set(issue_assignees["issue_id"])
    open_issues = issues[issues["state"] == "open"].copy()
    open_issues["assigned"] = open_issues["id"].isin(assigned_issue_ids)
    open_issues["older_than_90d"] = open_issues["age_days"] > 90
    issue_queue_rows = []
    for intent, group in open_issues.groupby("intent", observed=False):
        issue_queue_rows.append({
            "intent": intent,
            "open_issues": len(group),
            "share_of_open_queue": len(group) / len(open_issues) if len(open_issues) else np.nan,
            "no_human_response": int((~group["human_event"]).sum()),
            "no_snapshot_collaborator_response": int((~group["collab_event"]).sum()),
            "older_than_90d": int(group["older_than_90d"].sum()),
            "older_than_90d_no_collaborator_response": int((group["older_than_90d"] & ~group["collab_event"]).sum()),
            "assigned_pct": float(group["assigned"].mean()) if len(group) else np.nan,
            "median_age_days": float(group["age_days"].median()) if len(group) else np.nan,
            "p90_age_days": quantile(group["age_days"], 0.90),
        })
    current_issue_queue = pd.DataFrame(issue_queue_rows)

    reviewer_requests = reviewer_requests.sort_values("created_at").drop_duplicates(
        ["pull_request_id", "requested_id", "requested_reviewer_type"], keep="last"
    )
    outstanding_requests = reviewer_requests[reviewer_requests["removed"] == 0]
    outstanding_request_counts = outstanding_requests.groupby("pull_request_id").size()
    prs["outstanding_review_requests"] = prs["pr_id"].map(outstanding_request_counts).fillna(0).astype(int)
    latest_collab_review = (
        review_human[review_human["is_collab"]]
        .sort_values("submitted_at")
        .drop_duplicates(["issue_id", "user_id"], keep="last")
    )
    latest_review_states = latest_collab_review.groupby("issue_id")["state"].agg(
        outstanding_change_requests=lambda values: int((values == "CHANGES_REQUESTED").sum()),
        latest_approvals=lambda values: int((values == "APPROVED").sum()),
    )
    prs = prs.join(latest_review_states, on="id")
    for column in ["outstanding_change_requests", "latest_approvals"]:
        prs[column] = prs[column].fillna(0).astype(int)

    def has_current_label(issue_id: int, pattern: str) -> bool:
        return any(re.search(pattern, value.lower()) for value in labels_by_issue.get(issue_id, set()))

    prs["queue_label_stale"] = [has_current_label(i, r"stale") for i in prs["id"]]
    prs["queue_label_rebase"] = [has_current_label(i, r"rebase|conflict") for i in prs["id"]]
    open_prs = prs[prs["state"] == "open"].copy()
    open_prs["queue_age_days"] = (CUTOFF - open_prs["created_at"]).dt.total_seconds() / 86400
    pr_queue_rows = []
    for work_type, group in open_prs.groupby("work_type", observed=False):
        pr_queue_rows.append({
            "work_type": work_type,
            "open_prs": len(group),
            "share_of_open_queue": len(group) / len(open_prs) if len(open_prs) else np.nan,
            "draft": int(group["draft"].eq(1).sum()),
            "no_snapshot_collaborator_response": int((~group["collab_event"]).sum()),
            "no_submitted_collaborator_review": int((group["collab_reviews"] == 0).sum()),
            "with_outstanding_review_request": int((group["outstanding_review_requests"] > 0).sum()),
            "with_latest_change_request": int((group["outstanding_change_requests"] > 0).sum()),
            "with_latest_approval": int((group["latest_approvals"] > 0).sum()),
            "stale_labeled": int(group["queue_label_stale"].sum()),
            "rebase_or_conflict_labeled": int(group["queue_label_rebase"].sum()),
            "median_age_days": float(group["queue_age_days"].median()) if len(group) else np.nan,
            "p90_age_days": quantile(group["queue_age_days"], 0.90),
        })
    current_pr_queue = pd.DataFrame(pr_queue_rows)

    # Issue disposition is separated from close-event volume: "completed" and
    # "not planned" have different meanings for demand resolution.
    issue_disposition = (
        issues.assign(disposition=np.select(
            [issues["state"] == "open", issues["state_reason"].eq("completed"), issues["state_reason"].eq("not_planned"), issues["state_reason"].eq("duplicate")],
            ["Open", "Completed", "Not planned", "Duplicate"],
            default="Other closed",
        ))
        .groupby(["period", "intent", "disposition"], observed=False)
        .size()
        .rename("issues")
        .reset_index()
    )
    issue_disposition["share_within_intent"] = issue_disposition["issues"] / issue_disposition.groupby(["period", "intent"], observed=False)["issues"].transform("sum")
    issues["close_days_from_creation"] = (issues["closed_at"] - issues["created_at"]).dt.total_seconds() / 86400
    issue_disposition_horizon_rows = []
    for (label, intent), group in issues.groupby(["period", "intent"], observed=False):
        for horizon in (30, 90, 180):
            eligible = group[group["age_days"] >= horizon]
            closed_within = eligible["closed_at"].notna() & (eligible["close_days_from_creation"] <= horizon)
            completed = closed_within & eligible["state_reason"].eq("completed")
            not_planned = closed_within & eligible["state_reason"].eq("not_planned")
            duplicate = closed_within & eligible["state_reason"].eq("duplicate")
            issue_disposition_horizon_rows.append({
                "period": str(label), "intent": intent, "horizon_days": horizon, "eligible": len(eligible),
                "current_completed_and_closed_within_horizon_pct": float(completed.mean()) if len(eligible) else np.nan,
                "current_not_planned_and_closed_within_horizon_pct": float(not_planned.mean()) if len(eligible) else np.nan,
                "current_duplicate_and_closed_within_horizon_pct": float(duplicate.mean()) if len(eligible) else np.nan,
                "not_in_these_terminal_dispositions_within_horizon_pct": float((~(completed | not_planned | duplicate)).mean()) if len(eligible) else np.nan,
            })
    issue_disposition_horizons = pd.DataFrame(issue_disposition_horizon_rows)

    # Test-file presence is a verifier-availability signal, not evidence that the
    # tests are sufficient. Multi-label hardware rows intentionally overlap.
    verifier_gap_rows = []
    for (label, work_type), group in feasibility.groupby(["period", "work_type"], observed=False):
        verifier_gap_rows.append({
            "period": str(label), "dimension": "work_type", "name": work_type, "prs": len(group),
            "test_touched_pct": float(group["test_touched"].mean()),
            "benchmark_or_eval_touched_pct": float(group["benchmark_touched"].mean()),
            "review_intensive_pct": float(group["review_intensive"].mean()),
            "large_change_pct": float(group["large_change"].mean()),
        })
    for column in [c for c in feasibility if c.startswith("hardware__") and c != "hardware__Hardware-independent"]:
        for label, group in feasibility[feasibility[column]].groupby("period", observed=False):
            verifier_gap_rows.append({
                "period": str(label), "dimension": "hardware", "name": column.removeprefix("hardware__"), "prs": len(group),
                "test_touched_pct": float(group["test_touched"].mean()) if len(group) else np.nan,
                "benchmark_or_eval_touched_pct": float(group["benchmark_touched"].mean()) if len(group) else np.nan,
                "review_intensive_pct": float(group["review_intensive"].mean()) if len(group) else np.nan,
                "large_change_pct": float(group["large_change"].mean()) if len(group) else np.nan,
            })
    verifier_gaps = pd.DataFrame(verifier_gap_rows)

    complexity_rows = []
    for (label, size_bin), group in feasibility.groupby(["period", "size_bin"], observed=False):
        complexity_rows.append({
            "period": str(label),
            "cumulative_churn_bin": str(size_bin),
            "prs": len(group),
            "test_touched_pct": float(group["test_touched"].mean()) if len(group) else np.nan,
            "review_intensive_pct": float(group["review_intensive"].mean()) if len(group) else np.nan,
            "median_collaborator_reviews": float(group["collab_reviews"].median()) if len(group) else np.nan,
            "p90_collaborator_reviews": quantile(group["collab_reviews"], 0.90),
            "median_files": float(group["files"].median()) if len(group) else np.nan,
        })
    pr_complexity = pd.DataFrame(complexity_rows)

    review_dimension_rows = []
    for prefix, dimension in [("subsystem__", "subsystem"), ("hardware__", "hardware"), ("topic__", "topic")]:
        for column in [c for c in prs if c.startswith(prefix) and not c.endswith("Hardware-independent")]:
            name = column.removeprefix(prefix)
            for label, group in prs[prs[column]].groupby("period", observed=False):
                review_dimension_rows.append({
                    "period": str(label), "dimension": dimension, "name": name, "prs": len(group),
                    "reviewed_pr_pct": float((group["collab_reviews"] > 0).mean()) if len(group) else np.nan,
                    "collaborator_review_submissions": int(group["collab_reviews"].sum()),
                    "inline_review_comments": int(group["collab_inline_comments"].sum()),
                    "reviews_per_pr": float(group["collab_reviews"].mean()) if len(group) else np.nan,
                    "inline_comments_per_pr": float(group["collab_inline_comments"].mean()) if len(group) else np.nan,
                    "review_intensive_pct": float(group["review_intensive"].mean()) if len(group) else np.nan,
                })
    review_burden_dimensions = pd.DataFrame(review_dimension_rows)

    contributor_rows = []
    first_pr = prs.groupby("user_id")["created_at"].min()
    prs["author_first_pr_at"] = prs["user_id"].map(first_pr)
    prs["first_time_author_pr"] = prs["created_at"].eq(prs["author_first_pr_at"])
    monthly_contributors = prs.groupby("month", observed=False).agg(
        prs=("id", "size"),
        unique_authors=("user_id", "nunique"),
        external_prs=("author_role", lambda values: int((values == "External human").sum())),
        snapshot_collaborator_prs=("author_role", lambda values: int((values == "Snapshot collaborator").sum())),
    ).reset_index()
    external_first_time = prs[(prs["author_role"] == "External human") & prs["first_time_author_pr"]].groupby("month")["user_id"].nunique()
    external_unique = prs[prs["author_role"] == "External human"].groupby("month")["user_id"].nunique()
    monthly_contributors["external_unique_authors"] = monthly_contributors["month"].map(external_unique).fillna(0).astype(int)
    monthly_contributors["first_time_external_authors"] = monthly_contributors["month"].map(external_first_time).fillna(0).astype(int)
    monthly_contributors["external_pr_share"] = monthly_contributors["external_prs"] / monthly_contributors["prs"]
    for (label, role), group in prs.groupby(["period", "author_role"], observed=False):
        eligible_7 = group[(group["age_days"] >= 7) & group["at_risk_at"].notna()]
        eligible_90 = group[(group["age_days"] >= 90) & group["at_risk_at"].notna()]
        contributor_rows.append({
            "period": str(label),
            "author_role": role,
            "prs": len(group),
            "unique_authors": int(group["user_id"].nunique()),
            "first_time_authors": int(group.loc[group["first_time_author_pr"], "user_id"].nunique()),
            "merged_within_90d_pct": float((eligible_90["merged"] & (eligible_90["merge_days"] <= 90)).mean()) if len(eligible_90) else np.nan,
            "collaborator_response_7d_pct": float((eligible_7["collab_event"] & (eligible_7["collab_days"] <= 7)).mean()) if len(eligible_7) else np.nan,
        })
    contributors = pd.DataFrame(contributor_rows)

    declared_agent_rows = []
    for agent_label in ["claude-code-assisted", "codex"]:
        labeled_ids = {issue_id for issue_id, values in labels_by_issue.items() if agent_label in values}
        agent_group = prs[prs["id"].isin(labeled_ids)]
        declared_agent_rows.append({
            "label": agent_label,
            "prs": len(agent_group),
            "merged": int(agent_group["merged"].sum()),
            "first_seen": agent_group["created_at"].min(),
            "last_seen": agent_group["created_at"].max(),
        })
    declared_agent = pd.DataFrame(declared_agent_rows)

    issue_closure_reason = (
        issues.assign(state_reason=issues["state_reason"].fillna("open/no reason"))
        .groupby(["period", "state_reason"], observed=False)
        .size()
        .rename("issues")
        .reset_index()
    )
    closure_events = closed[(closed["closed"] == 1) & closed["issue_id"].isin(issues["id"])].copy()
    closure_events["period"] = period(closure_events["updated_at"])
    closure_events["actor_type"] = closure_events["actor_id"].map(users.set_index("id")["type"]).fillna("Unknown")
    issue_closure_actor = closure_events.groupby(["period", "actor_type"], observed=False).size().rename("close_events").reset_index()
    issue_closure_actor["share"] = issue_closure_actor["close_events"] / issue_closure_actor.groupby("period", observed=False)["close_events"].transform("sum")

    competing_rows = []
    competing_curves: dict[str, pd.DataFrame] = {}
    for label in PERIOD_ORDER:
        group = prs[(prs["period"] == label) & prs["at_risk_at"].notna()].copy()
        group["outcome"] = np.select([group["merged"], group["closed_unmerged"]], ["merged", "closed_unmerged"], default="censored")
        group["outcome_days"] = group["age_days"]
        group.loc[group["merged"], "outcome_days"] = group.loc[group["merged"], "merge_days"]
        group.loc[group["closed_unmerged"], "outcome_days"] = group.loc[group["closed_unmerged"], "close_days"]
        group = group[group["outcome_days"] >= 0]
        curve = competing_curve(group["outcome_days"], group["outcome"])
        competing_curves[label] = curve
        for horizon in (30, 90, 180):
            point = curve[curve["time"] <= horizon].iloc[-1]
            competing_rows.append({
                "period": label,
                "horizon_days": horizon,
                "cif_merged": point["cif_merged"],
                "cif_closed_unmerged": point["cif_closed_unmerged"],
            })
    competing_risks = pd.DataFrame(competing_rows)

    state_latest = closed.sort_values(["updated_at", "closed"]).drop_duplicates("issue_id", keep="last").set_index("issue_id")["closed"]
    current_closed = all_issues.set_index("id")["state"].eq("closed")
    common_state_ids = current_closed.index.intersection(state_latest.index)
    state_history_mismatches = int((current_closed.loc[common_state_ids] != state_latest.loc[common_state_ids].astype(bool)).sum())

    audit = pd.DataFrame([
        {"check": "canonical issues", "value": len(issues), "note": "issue rows without a pull_request-table match"},
        {"check": "canonical PRs", "value": len(prs), "note": "pull_request table rows joined to issue"},
        {"check": "issue/PR flag conflicts", "value": len(flag_conflicts), "note": "pull_request flag disagrees with pull_request table"},
        {"check": "closed artifacts lacking close history", "value": int(((all_issues["state"] == "closed") & ~all_issues["id"].isin(set(closed["issue_id"]))).sum()), "note": "fallback to closed_at required"},
        {"check": "current-state/history mismatches", "value": state_history_mismatches, "note": "materialized state is authoritative at the snapshot boundary"},
        {"check": "snapshot collaborators with triage+", "value": len(collab_ids), "note": "current snapshot, not a historical roster"},
        {"check": "snapshot collaborators with write+", "value": len(write_ids), "note": "current snapshot, not a historical roster"},
        {"check": "bot actors", "value": len(bot_ids), "note": "GitHub user.type = Bot"},
        {"check": "main-branch commits without PR mapping", "value": int(direct_main_commits["commits"].sum()), "note": "81 of 87 occurred in 2023"},
        {"check": "snapshot checksum verified", "value": 1, "note": snapshot_sha256},
    ])

    tables = {
        "dataset_audit": audit,
        "direct_main_commits": direct_main_commits,
        "monthly_overview": monthly,
        "period_summary": period_summary,
        "response_horizons": response,
        "response_by_author_role": response_by_author_role,
        "workload_mix": workload_mix,
        "classification_coverage": classification_coverage,
        "dimensions": dimensions,
        "topics": topics,
        "review_concentration": review_concentration,
        "recent_capacity_comparison": recent_capacity_comparison,
        "maintainer_actions": maintainer_actions,
        "maintainer_action_comparison": maintainer_action_comparison,
        "maintainer_action_concentration": maintainer_action_concentration,
        "monthly_capacity_sensitivity": monthly_capacity_sensitivity,
        "monthly_maintainer_actions": monthly_maintainer_actions,
        "review_burden_by_outcome": review_burden_outcome,
        "review_burden_by_work_type": review_burden_type,
        "review_burden_by_dimension": review_burden_dimensions,
        "task_feasibility": feasibility_summary,
        "benchmark_work_type_strata": benchmark_work_type,
        "benchmark_dimension_strata": benchmark_dimensions,
        "verifier_gaps": verifier_gaps,
        "pr_complexity": pr_complexity,
        "issue_outcomes": issue_outcomes,
        "issue_disposition": issue_disposition,
        "issue_disposition_horizons": issue_disposition_horizons,
        "pr_outcomes": pr_outcomes,
        "hardware_outcomes": hardware_outcomes,
        "backlog_age": backlog_age,
        "current_issue_queue": current_issue_queue,
        "current_pr_queue": current_pr_queue,
        "contributors": contributors,
        "monthly_contributors": monthly_contributors,
        "declared_agent_assistance": declared_agent,
        "issue_closure_reason": issue_closure_reason,
        "issue_closure_actor": issue_closure_actor,
        "pr_competing_risks": competing_risks,
    }
    for name, table in tables.items():
        table.to_csv(opt.output / f"{name}.csv", index=False)

    style()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(monthly["month"], monthly["issue_opened"], label="Issues opened", color=COLORS["orange"])
    axes[0].plot(monthly["month"], monthly["pr_opened"], label="PRs opened", color=COLORS["blue"])
    axes[0].plot(monthly["month"], monthly["pr_merged"], label="PRs merged", color=COLORS["green"])
    axes[0].set_ylabel("Artifacts per month")
    axes[0].set_title("Incoming work and merge throughput")
    axes[0].legend(ncol=3)
    axes[1].plot(monthly["month"], monthly["issue_backlog"], label="Issue backlog", color=COLORS["red"])
    axes[1].plot(monthly["month"], monthly["pr_backlog"], label="PR backlog", color=COLORS["navy"])
    axes[1].set_ylabel("Open at month end")
    axes[1].set_title("Month-end backlog")
    axes[1].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(opt.figures / "activity_and_backlog.png", bbox_inches="tight")
    plt.close(fig)

    seven = response[(response["horizon_days"] == 7) & ~response["artifact"].str.contains("submitted")].copy()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(PERIOD_ORDER))
    width = 0.18
    for index, (artifact, group) in enumerate(seven.groupby("artifact", sort=False)):
        group = group.set_index("period").reindex(PERIOD_ORDER)
        ax.bar(x + (index - 1.5) * width, group["rate"] * 100, width, label=artifact)
    ax.set_xticks(x, PERIOD_ORDER)
    ax.set_ylabel("Responded within 7 days (%)")
    ax.set_title("Human and snapshot-collaborator responsiveness")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(opt.figures / "response_within_7_days.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (artifact, dim) in zip(axes, [("Issue", "intent"), ("PR", "work_type")]):
        pivot = workload_mix[workload_mix["artifact"] == artifact].pivot(index="period", columns=dim, values="share").fillna(0).reindex(PERIOD_ORDER)
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_title(f"{artifact} workload mix")
        ax.set_xlabel("")
        ax.set_ylabel("Share")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    fig.tight_layout()
    fig.savefig(opt.figures / "workload_mix.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, dimension in zip(axes, ["subsystem", "hardware"]):
        subset = dimensions[(dimensions["dimension"] == dimension) & (~dimensions["name"].isin(["Hardware-independent", "Cross-backend"]))]
        pivot = subset.pivot(index="name", columns="period", values="share").fillna(0)[PERIOD_ORDER]
        pivot.plot(kind="barh", ax=ax)
        ax.set_title(f"PR {dimension} signals")
        ax.set_xlabel("Share of PRs")
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(opt.figures / "subsystems_and_hardware.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(monthly["month"], monthly["pr_opened"], color=COLORS["blue"], label="PRs opened")
    second = axes[0].twinx()
    second.plot(monthly["month"], monthly["active_collab_reviewers"], color=COLORS["orange"], label="Active snapshot reviewers")
    axes[0].set_title("Review demand and active reviewers")
    axes[0].set_ylabel("PRs opened")
    second.set_ylabel("Reviewers")
    handles = axes[0].get_lines() + second.get_lines()
    axes[0].legend(handles, [h.get_label() for h in handles], loc="upper left")
    axes[1].plot(monthly["month"], monthly["pr_arrivals_per_active_reviewer"], color=COLORS["red"])
    axes[1].set_title("PR arrivals per active snapshot reviewer")
    axes[1].set_ylabel("PRs / reviewer / month")
    fig.tight_layout()
    fig.savefig(opt.figures / "review_capacity.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    latest_pr = pr_outcomes[pr_outcomes["period"] == "2026 to May 18"].sort_values("prs")
    axes[0].barh(latest_pr["work_type"], latest_pr["test_touched_pct"] * 100, color=COLORS["green"])
    axes[0].set_title("Test-file signal by 2026 PR type")
    axes[0].set_xlabel("PRs touching tests (%)")
    axes[1].barh(latest_pr["work_type"], latest_pr["review_intensive_pct"] * 100, color=COLORS["orange"])
    axes[1].set_title("Review-intensive signal by 2026 PR type")
    axes[1].set_xlabel("PRs meeting review-intensity rule (%)")
    fig.tight_layout()
    fig.savefig(opt.figures / "benchmark_task_signals.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for label in PERIOD_ORDER:
        issue_group = issues[issues["period"] == label]
        issue_duration = issue_group["collab_days"].where(issue_group["collab_event"], issue_group["age_days"])
        issue_curve = km_curve(issue_duration, issue_group["collab_event"])
        issue_curve = issue_curve[issue_curve["time"] <= 90]
        axes[0].step(issue_curve["time"], 1 - issue_curve["survival"], where="post", label=label)

        pr_group = prs[(prs["period"] == label) & prs["at_risk_at"].notna()]
        pr_duration = pr_group["collab_days"].where(pr_group["collab_event"], pr_group["age_days"])
        pr_curve = km_curve(pr_duration, pr_group["collab_event"])
        pr_curve = pr_curve[pr_curve["time"] <= 90]
        axes[1].step(pr_curve["time"], 1 - pr_curve["survival"], where="post", label=label)
    axes[0].set_title("Issue: snapshot-collaborator response")
    axes[1].set_title("PR: snapshot-collaborator response")
    for ax in axes:
        ax.set_xlabel("Days since ready/created")
        ax.set_ylabel("Cumulative response probability")
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(opt.figures / "response_survival.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, label in zip(axes, PERIOD_ORDER):
        curve = competing_curves[label]
        curve = curve[curve["time"] <= 180]
        ax.step(curve["time"], curve["cif_merged"], where="post", label="Merged", color=COLORS["green"])
        ax.step(curve["time"], curve["cif_closed_unmerged"], where="post", label="Closed unmerged", color=COLORS["red"])
        ax.set_title(label)
        ax.set_xlabel("Days since ready")
        ax.set_xlim(0, 180)
    axes[0].set_ylabel("Cumulative incidence")
    axes[0].legend()
    fig.suptitle("PR outcomes with competing risks")
    fig.tight_layout()
    fig.savefig(opt.figures / "pr_competing_outcomes.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    action_pivot = maintainer_action_comparison.set_index("action")[["2025_monthly_mean", "2026_jan_apr_monthly_mean"]].rename(
        columns={"2025_monthly_mean": "2025", "2026_jan_apr_monthly_mean": "2026 Jan–Apr"}
    )
    action_pivot.plot(kind="barh", ax=axes[0], color=[COLORS["gray"], COLORS["green"]])
    axes[0].set_title("Observable non-author/operational actions")
    axes[0].set_xlabel("Monthly mean (events, not effort)")
    axes[0].set_ylabel("")
    recent_burden = review_burden_type[review_burden_type["period"] == "2026 to May 18"].sort_values("review_submission_share")
    y = np.arange(len(recent_burden))
    axes[1].barh(y - 0.18, recent_burden["pr_share"] * 100, 0.36, label="Share of PRs", color=COLORS["cyan"])
    axes[1].barh(y + 0.18, recent_burden["review_submission_share"] * 100, 0.36, label="Share of submitted reviews", color=COLORS["orange"])
    axes[1].set_yticks(y, recent_burden["work_type"])
    axes[1].set_xlabel("Share (%)")
    axes[1].set_title("2026 review load versus PR volume")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(opt.figures / "maintainer_workload.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))
    issue_plot = current_issue_queue.sort_values("open_issues")
    axes[0].barh(issue_plot["intent"], issue_plot["open_issues"], color=COLORS["cyan"], label="Open")
    axes[0].barh(issue_plot["intent"], issue_plot["older_than_90d_no_collaborator_response"], color=COLORS["red"], label=">90d, no collaborator response")
    axes[0].set_title("Open issue queue at snapshot")
    axes[0].set_xlabel("Issues")
    axes[0].legend(fontsize=8)
    pr_plot = current_pr_queue.sort_values("open_prs")
    axes[1].barh(pr_plot["work_type"], pr_plot["open_prs"], color=COLORS["cyan"], label="Open")
    axes[1].barh(pr_plot["work_type"], pr_plot["no_submitted_collaborator_review"], color=COLORS["orange"], label="No submitted collaborator review")
    axes[1].set_title("Open PR queue at snapshot")
    axes[1].set_xlabel("PRs")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(opt.figures / "current_queues.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6.4))
    topic_pivot = topics[topics["period"].isin(["2025", "2026 to May 18"])].pivot(index="topic", columns="period", values="share").fillna(0)
    topic_pivot = topic_pivot.sort_values("2026 to May 18")
    topic_pivot[["2025", "2026 to May 18"]].plot(kind="barh", ax=ax, color=[COLORS["gray"], COLORS["blue"]])
    ax.set_title("vLLM engineering topic signals")
    ax.set_xlabel("Share of PRs (multi-label)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(opt.figures / "engineering_topics.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    recent_gaps = verifier_gaps[(verifier_gaps["period"] == "2026 to May 18") & (verifier_gaps["dimension"] == "work_type")].sort_values("test_touched_pct")
    axes[0].barh(recent_gaps["name"], recent_gaps["test_touched_pct"] * 100, color=COLORS["green"])
    axes[0].set_title("Test-file signal in eligible 2026 PRs")
    axes[0].set_xlabel("Touches test files (%)")
    hardware_gaps = verifier_gaps[(verifier_gaps["period"] == "2026 to May 18") & (verifier_gaps["dimension"] == "hardware") & (verifier_gaps["prs"] >= 50)].sort_values("test_touched_pct")
    axes[1].barh(hardware_gaps["name"], hardware_gaps["test_touched_pct"] * 100, color=COLORS["orange"])
    axes[1].set_title("Test-file signal by hardware, eligible 2026 PRs")
    axes[1].set_xlabel("Touches test files (%)")
    fig.tight_layout()
    fig.savefig(opt.figures / "verifier_signals.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(monthly_contributors["month"], monthly_contributors["external_pr_share"] * 100, color=COLORS["blue"])
    axes[0].set_title("External contributions became the dominant intake")
    axes[0].set_ylabel("External-human share of opened PRs (%)")
    axes[0].set_ylim(0, 100)
    axes[1].plot(monthly_contributors["month"], monthly_contributors["first_time_external_authors"], color=COLORS["orange"], label="First-time external authors")
    second = axes[1].twinx()
    second.plot(monthly["month"], monthly["active_collab_reviewers"], color=COLORS["navy"], label="Active snapshot reviewers")
    axes[1].set_title("Contributor onboarding versus review capacity")
    axes[1].set_ylabel("First-time authors / month")
    second.set_ylabel("Active reviewers / month")
    handles = axes[1].get_lines() + second.get_lines()
    axes[1].legend(handles, [line.get_label() for line in handles], loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(opt.figures / "contributor_pressure.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    recent_complexity = pr_complexity[pr_complexity["period"] == "2026 to May 18"].copy()
    size_order = ["≤20", "21–100", "101–500", "501–2,000", ">2,000"]
    recent_complexity["cumulative_churn_bin"] = pd.Categorical(recent_complexity["cumulative_churn_bin"], size_order, ordered=True)
    recent_complexity = recent_complexity.sort_values("cumulative_churn_bin")
    axes[0].bar(recent_complexity["cumulative_churn_bin"].astype(str), recent_complexity["review_intensive_pct"] * 100, color=COLORS["orange"])
    axes[0].set_title("Review intensity rises with patch size")
    axes[0].set_ylabel("Review-intensive PRs (%)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(recent_complexity["cumulative_churn_bin"].astype(str), recent_complexity["test_touched_pct"] * 100, color=COLORS["green"])
    axes[1].set_title("Test-file signal rises with patch size")
    axes[1].set_ylabel("Touches test files (%)")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(opt.figures / "pr_complexity.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "snapshot": {
            "cutoff": str(CUTOFF),
            "sha256": snapshot_sha256,
        },
        "audit": audit.to_dict(orient="records"),
        "direct_main_commits": direct_main_commits.to_dict(orient="records"),
        "period_summary": period_summary.to_dict(orient="records"),
        "response_horizons": response.to_dict(orient="records"),
        "response_by_author_role": response_by_author_role.to_dict(orient="records"),
        "review_concentration": review_concentration.to_dict(orient="records"),
        "recent_capacity_comparison": recent_capacity_comparison.to_dict(orient="records"),
        "maintainer_actions": maintainer_actions.to_dict(orient="records"),
        "maintainer_action_comparison": maintainer_action_comparison.to_dict(orient="records"),
        "maintainer_action_concentration": maintainer_action_concentration.to_dict(orient="records"),
        "monthly_capacity_sensitivity": monthly_capacity_sensitivity.to_dict(orient="records"),
        "review_burden_by_outcome": review_burden_outcome.to_dict(orient="records"),
        "review_burden_by_work_type": review_burden_type.to_dict(orient="records"),
        "review_burden_by_dimension": review_burden_dimensions.to_dict(orient="records"),
        "task_feasibility": feasibility_summary.to_dict(orient="records"),
        "topics": topics.to_dict(orient="records"),
        "classification_coverage": classification_coverage.to_dict(orient="records"),
        "verifier_gaps": verifier_gaps.to_dict(orient="records"),
        "pr_complexity": pr_complexity.to_dict(orient="records"),
        "backlog_age": backlog_age.to_dict(orient="records"),
        "current_issue_queue": current_issue_queue.to_dict(orient="records"),
        "current_pr_queue": current_pr_queue.to_dict(orient="records"),
        "issue_disposition_horizons": issue_disposition_horizons.to_dict(orient="records"),
        "contributors": contributors.to_dict(orient="records"),
        "pr_competing_risks": competing_risks.to_dict(orient="records"),
        "issue_closure_actor": issue_closure_actor.to_dict(orient="records"),
        "issue_outcomes": issue_outcomes.to_dict(orient="records"),
        "pr_outcomes": pr_outcomes.to_dict(orient="records"),
        "hardware_outcomes": hardware_outcomes.to_dict(orient="records"),
    }
    opt.summary.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
