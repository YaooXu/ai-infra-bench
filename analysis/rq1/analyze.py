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

from load_merged import load_merged_inputs


CUTOFF = pd.Timestamp("2026-07-31 23:59:59")
EXPECTED_SHA256 = "2ac86507a95f9b8785e6ce0bbf2745e3fbba67c747e37b54020a7e57ce80f8b5"
RECENT_PERIOD = "2026 Jan–Jul"
PERIOD_ORDER = ["Launch–2024", "2025", RECENT_PERIOD]
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
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True).dt.tz_localize(None)


def period(series: pd.Series) -> pd.Categorical:
    values = np.full(len(series), None, dtype=object)
    values[series.notna() & (series < pd.Timestamp("2025-01-01"))] = PERIOD_ORDER[0]
    values[series.notna() & (series >= pd.Timestamp("2025-01-01")) & (series < pd.Timestamp("2026-01-01"))] = PERIOD_ORDER[1]
    values[series.notna() & (series >= pd.Timestamp("2026-01-01")) & (series <= CUTOFF)] = PERIOD_ORDER[2]
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


def concentration(values: pd.Series) -> dict[str, float | int]:
    counts = values[values > 0].sort_values(ascending=False).astype(float)
    total = counts.sum()
    if not total:
        return {
            "active_people": 0,
            "top_1_share": float("nan"),
            "top_5_share": float("nan"),
            "contributors_for_50pct": 0,
            "contributors_for_80pct": 0,
            "hhi": float("nan"),
            "gini": float("nan"),
        }
    shares = counts / total
    cumulative = shares.cumsum()
    return {
        "active_people": len(counts),
        "top_1_share": float(shares.iloc[0]),
        "top_5_share": float(shares.head(5).sum()),
        "contributors_for_50pct": int((cumulative < 0.5).sum() + 1),
        "contributors_for_80pct": int((cumulative < 0.8).sum() + 1),
        "hhi": float((shares * shares).sum()),
        "gini": gini(counts),
    }


def path_area(filename: str) -> str:
    path = (filename or "").lower()
    rules = [
        ("V1 engine/runtime", r"^vllm/v1/"),
        ("Model executor/support", r"^vllm/model_executor/"),
        ("Distributed/executors", r"^vllm/(distributed|executor)/"),
        ("Frontend/entrypoints", r"^vllm/entrypoints/"),
        ("Kernels/native code", r"^(csrc|vllm/csrc|vllm/attention|vllm/_custom_ops)/"),
        ("Platforms/backends", r"^vllm/platforms/"),
        ("Compilation", r"^vllm/compilation/"),
        ("Legacy engine/worker", r"^vllm/(engine|worker)/"),
        ("Tests", r"^tests?/"),
        ("Benchmarks/evals", r"^(benchmarks?|evals?)/"),
        ("Documentation", r"^docs?/|\.(md|rst)$"),
        ("CI/build/packaging", r"^(\.buildkite|\.github|docker|requirements)/|^(setup\.py|pyproject\.toml|cmakelists\.txt)$"),
        ("Examples", r"^examples?/"),
        ("Other vLLM Python", r"^vllm/"),
    ]
    for name, pattern in rules:
        if re.search(pattern, path):
            return name
    return "Other repository files"


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

    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='canonical_artifact'"
    ).fetchone():
        raise ValueError("RQ1 analysis now requires the merged July 31 database")
    inputs = load_merged_inputs(conn)
    users = inputs["users"]
    collaborators = inputs["collaborators"]
    all_issues = inputs["all_issues"]
    prs_raw = inputs["prs_raw"]
    comments = inputs["comments"]
    reviews = inputs["reviews"]
    inline = inputs["inline"]
    closed = inputs["closed"]
    merged = inputs["merged"]
    ready = inputs["ready"]
    issue_labels_named = inputs["issue_labels_named"]
    commit_pr = inputs["commit_pr"]
    commit_files = inputs["commit_files"]
    pr_files = inputs["pr_files"]
    issue_refs = inputs["issue_refs"]
    label_history = inputs["label_history"]
    issue_assignees = inputs["issue_assignees"]
    reviewer_requests = inputs["reviewer_requests"]
    direct_main_commits = inputs["direct_main_commits"]
    git_commit_identity_audit = inputs["git_commit_identity_audit"]
    merged_input_audit = inputs["input_audit"]
    audit_counts = merged_input_audit.set_index("check_name")["value"].to_dict()
    observed_core_counts = {
        "canonical_artifacts": len(all_issues),
        "canonical_pull_requests": len(prs_raw),
        "canonical_comments": len(comments),
        "canonical_reviews": len(reviews),
        "canonical_inline_comments": len(inline),
        "canonical_pr_commit_associations": len(commit_pr),
        "canonical_pr_files": len(pr_files),
        "canonical_merged_prs": merged["issue_id"].nunique(),
    }
    mismatched_counts = {
        name: (audit_counts.get(name), observed)
        for name, observed in observed_core_counts.items()
        if audit_counts.get(name) != observed
    }
    if mismatched_counts:
        raise ValueError(f"Merged input count mismatch: {mismatched_counts}")
    if audit_counts.get("release_validation_failures") != 0:
        raise ValueError("Merged database contains failed release validations")
    if merged["issue_id"].duplicated().any():
        raise ValueError("Duplicate merged-PR records after event/materialized reconciliation")
    for name, frame, key in [
        ("artifacts", all_issues, "id"),
        ("pull requests", prs_raw, "id"),
        ("comments", comments, "id"),
        ("reviews", reviews, "id"),
        ("inline comments", inline, "id"),
    ]:
        if frame[key].duplicated().any():
            raise ValueError(f"Duplicate canonical {name} IDs")
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

    post_cutoff_times = {
        "artifacts": int((all_issues["created_at"] > CUTOFF).sum()),
        "comments": int((comments["created_at"] > CUTOFF).sum()),
        "reviews": int((reviews["submitted_at"] > CUTOFF).sum()),
        "inline comments": int((inline["created_at"] > CUTOFF).sum()),
        "close/reopen events": int((closed["updated_at"] > CUTOFF).sum()),
        "merge events": int((merged["merged_at"] > CUTOFF).sum()),
        "ready/draft events": int((ready["created_at"] > CUTOFF).sum()),
        "reference events": int((issue_refs["referenced_at"] > CUTOFF).sum()),
        "label events": int((label_history["updated_at"] > CUTOFF).sum()),
        "review-request events": int((reviewer_requests["created_at"] > CUTOFF).sum()),
    }
    if any(post_cutoff_times.values()):
        raise ValueError(f"Post-cutoff analytical rows detected: {post_cutoff_times}")

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

    labels_by_issue = (
        issue_labels_named.groupby("issue_id")["name"].agg(lambda values: set(values)).to_dict()
    )
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
    prs["author_permission"] = np.select(
        [prs["user_id"].isin(bot_ids), prs["user_id"].isin(write_ids), prs["user_id"].isin(collab_ids)],
        ["Bot", "Snapshot write+", "Snapshot triage-only"],
        default="External human",
    )

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
    prs["merger_id"] = prs["id"].map(merge_by_issue["actor_id"])
    prs["merger_role"] = np.select(
        [prs["merger_id"].isin(bot_ids), prs["merger_id"].isin(write_ids), prs["merger_id"].isin(collab_ids)],
        ["Bot", "Snapshot write+", "Snapshot triage-only"],
        default="Other/unknown",
    )
    prs["merged"] = prs["merged_at"].notna()
    prs["author_merged_own_pr"] = prs["merged"] & prs["merger_id"].eq(prs["user_id"])
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

    delta_file_pr_ids = set(prs_raw.loc[prs_raw["source_layer"] == "delta", "id"])
    base_file_rows = (
        commit_pr[~commit_pr["pull_request_id"].isin(delta_file_pr_ids)]
        .merge(commit_files, on="commit_sha", how="inner")
        .drop_duplicates(["pull_request_id", "commit_sha", "filename"])
    )
    delta_file_rows = pr_files.copy()
    delta_file_rows["commit_sha"] = pd.NA
    file_rows = pd.concat(
        [
            base_file_rows[
                ["pull_request_id", "commit_sha", "filename", "additions", "deletions", "changes"]
            ],
            delta_file_rows[
                ["pull_request_id", "commit_sha", "filename", "additions", "deletions", "changes"]
            ],
        ],
        ignore_index=True,
    )
    file_rows["issue_id"] = file_rows["pull_request_id"].map(pr_map)
    file_rows["path_area"] = file_rows["filename"].map(path_area)
    file_agg = file_rows.groupby("issue_id").agg(
        files=("filename", "nunique"),
        cumulative_additions=("additions", "sum"),
        cumulative_deletions=("deletions", "sum"),
        cumulative_churn=("changes", "sum"),
        paths=("filename", lambda x: " ".join(sorted(set(x)))),
    )
    commit_agg = (
        commit_pr.groupby("pull_request_id")["commit_sha"]
        .nunique()
        .rename("commits")
        .to_frame()
    )
    commit_agg["issue_id"] = commit_agg.index.map(pr_map)
    commit_agg = commit_agg.dropna(subset=["issue_id"]).groupby("issue_id")["commits"].max()
    prs = prs.join(file_agg, on="id").join(commit_agg, on="id")
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
    prs["cutoff_stable_classification_inputs"] = (
        prs["files_cutoff_stable"].eq(1)
        & prs["issue_representation_may_postdate_cutoff"].eq(0)
    )
    issues["cutoff_stable_classification_inputs"] = issues[
        "representation_may_postdate_cutoff"
    ].eq(0)
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

    classification_sensitivity_rows = []
    for population, frame, dimension, prefix in [
        ("Issue", issues, "intent", None),
        ("PR", prs, "work_type", None),
        ("PR", prs, "hardware", "hardware__"),
        ("PR", prs, "topic", "topic__"),
    ]:
        recent = frame[frame["period"] == RECENT_PERIOD]
        stable = recent[recent["cutoff_stable_classification_inputs"]]
        if prefix is None:
            names = sorted(recent[dimension].dropna().unique())
            for name in names:
                all_share = float(recent[dimension].eq(name).mean())
                stable_share = float(stable[dimension].eq(name).mean())
                classification_sensitivity_rows.append({
                    "artifact": population,
                    "dimension": dimension,
                    "name": name,
                    "all_records": len(recent),
                    "stable_records": len(stable),
                    "all_share": all_share,
                    "stable_only_share": stable_share,
                    "absolute_difference": stable_share - all_share,
                })
        else:
            for column in sorted(c for c in recent if c.startswith(prefix)):
                name = column.removeprefix(prefix)
                all_share = float(recent[column].mean())
                stable_share = float(stable[column].mean())
                classification_sensitivity_rows.append({
                    "artifact": population,
                    "dimension": dimension,
                    "name": name,
                    "all_records": len(recent),
                    "stable_records": len(stable),
                    "all_share": all_share,
                    "stable_only_share": stable_share,
                    "absolute_difference": stable_share - all_share,
                })
    classification_stability_sensitivity = pd.DataFrame(classification_sensitivity_rows)

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
            RECENT_PERIOD: (pd.Timestamp("2026-01-01"), CUTOFF),
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
            "2026_jan_jul_monthly_mean": float(action_month_pivot.loc["2026-01-01":"2026-07-01", action].mean()),
        })
    maintainer_action_comparison = pd.DataFrame(action_comparison_rows)
    maintainer_action_comparison["relative_change"] = maintainer_action_comparison["2026_jan_jul_monthly_mean"] / maintainer_action_comparison["2025_monthly_mean"] - 1
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
    recent_months = comparison_monthly.loc["2026-01-01":"2026-07-01"]
    for metric, column in comparison_metrics.items():
        baseline_mean = float(baseline_months[column].mean())
        recent_mean = float(recent_months[column].mean())
        recent_comparison_rows.append({
            "metric": metric,
            "2025_monthly_mean": baseline_mean,
            "2026_jan_jul_monthly_mean": recent_mean,
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

    # Engineering ownership: distinguish code authored by current write-capable
    # collaborators from community intake and from review/merge gatekeeping.
    human_prs = prs[prs["author_role"] != "Bot"].copy()
    pr_author_role_rows = []
    for label, period_group in human_prs.groupby("period", observed=False):
        for permission, role_group in period_group.groupby("author_permission", observed=False):
            for work_type, group in role_group.groupby("work_type", observed=False):
                eligible_90 = group[(group["age_days"] >= 90) & group["at_risk_at"].notna()]
                covered = group[group["commits"] > 0]
                type_total = int((period_group["work_type"] == work_type).sum())
                pr_author_role_rows.append({
                    "period": str(label),
                    "author_permission": permission,
                    "work_type": work_type,
                    "prs": len(group),
                    "unique_authors": int(group["user_id"].nunique()),
                    "share_within_author_role": len(group) / len(role_group) if len(role_group) else np.nan,
                    "share_of_work_type": len(group) / type_total if type_total else np.nan,
                    "open_at_snapshot_pct": float((group["state"] == "open").mean()) if len(group) else np.nan,
                    "merged_at_snapshot_pct": float(group["merged"].mean()) if len(group) else np.nan,
                    "eligible_for_90d_outcome": len(eligible_90),
                    "merged_within_90d": int((eligible_90["merged"] & (eligible_90["merge_days"] <= 90)).sum()),
                    "closed_unmerged_within_90d": int((eligible_90["closed_unmerged"] & (eligible_90["close_days"] <= 90)).sum()),
                    "merged_within_90d_pct": float((eligible_90["merged"] & (eligible_90["merge_days"] <= 90)).mean()) if len(eligible_90) else np.nan,
                    "closed_unmerged_within_90d_pct": float((eligible_90["closed_unmerged"] & (eligible_90["close_days"] <= 90)).mean()) if len(eligible_90) else np.nan,
                    "reviewed_by_snapshot_collaborator_pct": float((group["collab_reviews"] > 0).mean()) if len(group) else np.nan,
                    "collaborator_reviews": int(group["collab_reviews"].sum()),
                    "median_collaborator_reviews": float(group["collab_reviews"].median()) if len(group) else np.nan,
                    "commit_data_coverage_pct": float((group["commits"] > 0).mean()) if len(group) else np.nan,
                    "median_cumulative_churn_when_covered": float(covered["cumulative_churn"].median()) if len(covered) else np.nan,
                    "test_touched_when_covered_pct": float(covered["test_touched"].mean()) if len(covered) else np.nan,
                    "hardware_specific_pct": float(group["hardware_specific"].mean()) if len(group) else np.nan,
                    "review_intensive_pct": float(group["review_intensive"].mean()) if len(group) else np.nan,
                })
    pr_by_author_role_and_type = pd.DataFrame(pr_author_role_rows)
    pr_author_role_summary = pr_by_author_role_and_type.groupby(["period", "author_permission"], observed=False).agg(
        prs=("prs", "sum"),
        unique_authors=("unique_authors", "max"),
        eligible_for_90d_outcome=("eligible_for_90d_outcome", "sum"),
        merged_within_90d=("merged_within_90d", "sum"),
        closed_unmerged_within_90d=("closed_unmerged_within_90d", "sum"),
        collaborator_reviews=("collaborator_reviews", "sum"),
    ).reset_index()
    # Unique authors above cannot be summed across work types; recompute exactly.
    exact_unique_authors = human_prs.groupby(["period", "author_permission"], observed=False)["user_id"].nunique()
    pr_author_role_summary["unique_authors"] = [
        exact_unique_authors.get((row.period, row.author_permission), 0)
        for row in pr_author_role_summary.itertuples(index=False)
    ]
    pr_author_role_summary["merged_within_90d_pct"] = pr_author_role_summary["merged_within_90d"] / pr_author_role_summary["eligible_for_90d_outcome"].replace(0, np.nan)
    pr_author_role_summary["closed_unmerged_within_90d_pct"] = pr_author_role_summary["closed_unmerged_within_90d"] / pr_author_role_summary["eligible_for_90d_outcome"].replace(0, np.nan)
    role_metrics = human_prs.groupby(["period", "author_permission"], observed=False).agg(
        open_at_snapshot_pct=("state", lambda values: float((values == "open").mean())),
        reviewed_by_snapshot_collaborator_pct=("collab_reviews", lambda values: float((values > 0).mean())),
        hardware_specific_pct=("hardware_specific", "mean"),
        review_intensive_pct=("review_intensive", "mean"),
    ).reset_index()
    pr_author_role_summary = pr_author_role_summary.merge(role_metrics, on=["period", "author_permission"], how="left")
    pr_author_role_summary["reviews_per_pr"] = pr_author_role_summary["collaborator_reviews"] / pr_author_role_summary["prs"]

    pr_role_dimension_rows = []
    for prefix, dimension in [("subsystem__", "subsystem"), ("hardware__", "hardware"), ("topic__", "topic")]:
        for column in [c for c in human_prs if c.startswith(prefix)]:
            name = column.removeprefix(prefix)
            for label, period_group in human_prs.groupby("period", observed=False):
                dimension_total = int(period_group[column].sum())
                for permission, role_group in period_group.groupby("author_permission", observed=False):
                    group = role_group[role_group[column]]
                    if not len(group):
                        continue
                    pr_role_dimension_rows.append({
                        "period": str(label), "dimension": dimension, "name": name,
                        "author_permission": permission, "prs": len(group),
                        "unique_authors": int(group["user_id"].nunique()),
                        "share_within_author_role": len(group) / len(role_group),
                        "share_of_dimension": len(group) / dimension_total if dimension_total else np.nan,
                        "merged_at_snapshot_pct": float(group["merged"].mean()),
                        "reviewed_by_snapshot_collaborator_pct": float((group["collab_reviews"] > 0).mean()),
                        "collaborator_reviews": int(group["collab_reviews"].sum()),
                        "reviews_per_pr": float(group["collab_reviews"].mean()),
                    })
    pr_by_author_role_and_dimension = pd.DataFrame(pr_role_dimension_rows)

    ownership_rows = []

    def add_ownership(group: pd.DataFrame, period_name: str, population: str, dimension: str, name: str) -> None:
        if not len(group):
            return
        stats = concentration(group.groupby("user_id").size())
        ownership_rows.append({
            "period": period_name, "population": population, "dimension": dimension,
            "name": name, "prs": len(group), **stats,
        })

    ownership_populations = {
        "All human authors": human_prs,
        "Snapshot collaborators": human_prs[human_prs["author_role"] == "Snapshot collaborator"],
        "Snapshot write+": human_prs[human_prs["author_permission"] == "Snapshot write+"],
        "External humans": human_prs[human_prs["author_role"] == "External human"],
    }
    for population, population_frame in ownership_populations.items():
        for label, period_group in population_frame.groupby("period", observed=False):
            add_ownership(period_group, str(label), population, "overall", "All PRs")
            for name, group in period_group.groupby("work_type", observed=False):
                add_ownership(group, str(label), population, "work_type", name)
            for prefix, dimension in [("subsystem__", "subsystem"), ("hardware__", "hardware"), ("topic__", "topic")]:
                for column in [c for c in period_group if c.startswith(prefix)]:
                    add_ownership(period_group[period_group[column]], str(label), population, dimension, column.removeprefix(prefix))
    engineering_ownership = pd.DataFrame(ownership_rows)

    merged_human = human_prs[human_prs["merged"]].copy()
    merge_gatekeeping_rows = []
    for (label, work_type), group in merged_human.groupby(["period", "work_type"], observed=False):
        actor_counts = group.dropna(subset=["merger_id"]).groupby("merger_id").size()
        stats = concentration(actor_counts)
        merge_gatekeeping_rows.append({
            "period": str(label), "work_type": work_type, "merged_prs": len(group),
            "external_authored_pct": float((group["author_role"] == "External human").mean()),
            "snapshot_write_authored_pct": float((group["author_permission"] == "Snapshot write+").mean()),
            "author_merged_own_pr_pct": float(group["author_merged_own_pr"].mean()),
            "median_days_to_merge": float(group["merge_days"].median()),
            **{f"merge_actor_{key}": value for key, value in stats.items()},
        })
    merge_gatekeeping = pd.DataFrame(merge_gatekeeping_rows)
    merge_actor_roles = (
        merged_human.groupby(["period", "merger_role"], observed=False)
        .size().rename("merged_prs").reset_index()
    )
    merge_actor_roles["share"] = merge_actor_roles["merged_prs"] / merge_actor_roles.groupby("period", observed=False)["merged_prs"].transform("sum")

    # Changed-path areas are single-label per file and multi-label per PR. They
    # expose concrete code ownership below the broad topic taxonomy.
    pr_area = (
        file_rows.dropna(subset=["issue_id"])
        .groupby(["issue_id", "path_area"], observed=False)
        .agg(area_cumulative_changes=("changes", "sum"), area_files=("filename", "nunique"))
        .reset_index()
        .merge(
            prs[["id", "period", "author_role", "author_permission", "user_id", "merged", "collab_reviews", "review_intensive"]],
            left_on="issue_id", right_on="id", how="inner",
        )
    )
    path_area_rows = []
    for (label, area), group in pr_area.groupby(["period", "path_area"], observed=False):
        total = len(group)
        for permission, role_group in group.groupby("author_permission", observed=False):
            path_area_rows.append({
                "period": str(label), "path_area": area, "author_permission": permission,
                "prs": len(role_group), "share_of_area_prs": len(role_group) / total,
                "unique_authors": int(role_group["user_id"].nunique()),
                "merged_at_snapshot_pct": float(role_group["merged"].mean()),
                "median_area_cumulative_changes": float(role_group["area_cumulative_changes"].median()),
                "reviewed_by_snapshot_collaborator_pct": float((role_group["collab_reviews"] > 0).mean()),
                "reviews_per_pr": float(role_group["collab_reviews"].mean()),
            })
    path_area_by_author_role = pd.DataFrame(path_area_rows)

    path_ownership_rows = []
    for population, population_frame in {
        "All human authors": pr_area[pr_area["author_role"] != "Bot"],
        "Snapshot collaborators": pr_area[pr_area["author_role"] == "Snapshot collaborator"],
        "External humans": pr_area[pr_area["author_role"] == "External human"],
    }.items():
        for (label, area), group in population_frame.groupby(["period", "path_area"], observed=False):
            path_ownership_rows.append({
                "period": str(label), "population": population, "path_area": area,
                "prs": len(group), **concentration(group.groupby("user_id").size()),
            })
    path_area_ownership = pd.DataFrame(path_ownership_rows)

    # Review ownership and reviewer specialization use review-event time, not PR
    # creation cohort. They describe who carried gatekeeping work in each period.
    review_facts = collab_reviews.merge(
        prs[["id", "work_type"] + [c for c in prs if c.startswith(("subsystem__", "hardware__", "topic__"))]],
        left_on="issue_id", right_on="id", how="inner",
    )
    review_facts["review_period"] = period(review_facts["submitted_at"])
    review_ownership_rows = []

    def add_review_ownership(group: pd.DataFrame, period_name: str, dimension: str, name: str) -> None:
        if not len(group):
            return
        review_ownership_rows.append({
            "period": period_name, "dimension": dimension, "name": name,
            "review_submissions": len(group), "reviewed_prs": int(group["issue_id"].nunique()),
            **concentration(group.groupby("user_id").size()),
        })

    for label, period_group in review_facts.groupby("review_period", observed=False):
        add_review_ownership(period_group, str(label), "overall", "All reviews")
        for name, group in period_group.groupby("work_type", observed=False):
            add_review_ownership(group, str(label), "work_type", name)
        for prefix, dimension in [("subsystem__", "subsystem"), ("hardware__", "hardware"), ("topic__", "topic")]:
            for column in [c for c in period_group if c.startswith(prefix)]:
                add_review_ownership(period_group[period_group[column]], str(label), dimension, column.removeprefix(prefix))
    review_ownership = pd.DataFrame(review_ownership_rows)

    reviewer_specialization_rows = []
    reviewer_primary_rows = []
    for label, period_group in review_facts.groupby("review_period", observed=False):
        profiles = []
        for reviewer_id, reviewer_group in period_group.groupby("user_id"):
            counts = reviewer_group.groupby("work_type").size().sort_values(ascending=False)
            profiles.append({
                "reviews": len(reviewer_group),
                "reviewed_prs": reviewer_group["issue_id"].nunique(),
                "distinct_work_types": len(counts),
                "primary_work_type": counts.index[0],
                "primary_work_type_share": counts.iloc[0] / counts.sum(),
            })
        profile = pd.DataFrame(profiles)
        reviewer_specialization_rows.append({
            "period": str(label), "active_reviewers": len(profile),
            "median_review_submissions": float(profile["reviews"].median()),
            "p90_review_submissions": quantile(profile["reviews"], 0.90),
            "median_distinct_work_types": float(profile["distinct_work_types"].median()),
            "median_primary_work_type_share": float(profile["primary_work_type_share"].median()),
            "reviewers_with_majority_specialty_pct": float((profile["primary_work_type_share"] >= 0.5).mean()),
        })
        for name, count in profile["primary_work_type"].value_counts().items():
            reviewer_primary_rows.append({"period": str(label), "primary_work_type": name, "reviewers": int(count), "share": count / len(profile)})
    reviewer_specialization = pd.DataFrame(reviewer_specialization_rows)
    reviewer_primary_work_type = pd.DataFrame(reviewer_primary_rows)

    # Anonymized collaborator portfolio overlap separates engineering from
    # gatekeeping without publishing individual rankings or identifiers.
    portfolio_frames = []
    authored_actions = prs[prs["user_id"].isin(collab_ids)][["created_at", "user_id"]].rename(columns={"created_at": "at", "user_id": "actor_id"})
    authored_actions["portfolio_action"] = "Authored PR"
    portfolio_frames.append(authored_actions)
    for frame, at, actor, action in [
        (collab_reviews, "submitted_at", "user_id", "Submitted review"),
        (collab_issue_comments, "created_at", "user_id", "Issue response"),
        (merged[merged["actor_id"].isin(collab_ids)], "merged_at", "actor_id", "Merge"),
    ]:
        subset = frame[[at, actor]].rename(columns={at: "at", actor: "actor_id"}).copy()
        subset["portfolio_action"] = action
        portfolio_frames.append(subset)
    portfolio_events = pd.concat(portfolio_frames, ignore_index=True).dropna(subset=["at", "actor_id"])
    portfolio_events["period"] = period(portfolio_events["at"])
    portfolio = portfolio_events.groupby(["period", "actor_id", "portfolio_action"], observed=True).size().unstack(fill_value=0).reset_index()
    for column in ["Authored PR", "Submitted review", "Issue response", "Merge"]:
        if column not in portfolio:
            portfolio[column] = 0
    portfolio["engineering"] = portfolio["Authored PR"] > 0
    portfolio["gatekeeping"] = (portfolio[["Submitted review", "Issue response", "Merge"]].sum(axis=1) > 0)
    portfolio["portfolio_type"] = np.select(
        [portfolio["engineering"] & portfolio["gatekeeping"], portfolio["engineering"], portfolio["gatekeeping"]],
        ["Engineering and gatekeeping", "Engineering only", "Gatekeeping only"],
        default="No observed action",
    )
    collaborator_portfolio = portfolio.groupby(["period", "portfolio_type"], observed=True).agg(
        snapshot_collaborators=("actor_id", "nunique"),
        authored_prs=("Authored PR", "sum"),
        submitted_reviews=("Submitted review", "sum"),
        issue_responses=("Issue response", "sum"),
        merges=("Merge", "sum"),
    ).reset_index()
    inactive_rows = []
    for label in PERIOD_ORDER:
        observed_people = int(portfolio.loc[portfolio["period"] == label, "actor_id"].nunique())
        inactive_rows.append({
            "period": label, "portfolio_type": "No observed public action",
            "snapshot_collaborators": len(collab_ids) - observed_people,
            "authored_prs": 0, "submitted_reviews": 0, "issue_responses": 0, "merges": 0,
        })
    collaborator_portfolio = pd.concat([collaborator_portfolio, pd.DataFrame(inactive_rows)], ignore_index=True)
    collaborator_portfolio["share_of_snapshot_roster"] = collaborator_portfolio["snapshot_collaborators"] / len(collab_ids)

    contributor_rows = []
    first_pr = prs.groupby("user_id")["created_at"].min()
    prs["author_first_pr_at"] = prs["user_id"].map(first_pr)
    prs["first_time_author_pr"] = prs["created_at"].eq(prs["author_first_pr_at"])
    ordered_prs = prs.sort_values(["user_id", "created_at", "id"])
    observed_pr_number = ordered_prs.groupby("user_id").cumcount().add(1)
    prs.loc[ordered_prs.index, "author_observed_pr_number"] = observed_pr_number.to_numpy()
    prs["author_observed_pr_number"] = prs["author_observed_pr_number"].astype(int)
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

    # Contributor lifecycle metrics expose onboarding demand hidden by raw PR
    # volume. "External" remains a snapshot-roster definition throughout.
    external_prs = prs[prs["author_role"] == "External human"].copy()
    external_prs["experience"] = pd.cut(
        external_prs["author_observed_pr_number"],
        bins=[0, 1, 5, np.inf],
        labels=["First observed PR", "2nd–5th observed PR", "6th+ observed PR"],
    )
    external_experience_rows = []
    for (label, experience), group in external_prs.groupby(["period", "experience"], observed=True):
        eligible_7 = group[(group["age_days"] >= 7) & group["at_risk_at"].notna()]
        eligible_90 = group[(group["age_days"] >= 90) & group["at_risk_at"].notna()]
        external_experience_rows.append({
            "period": str(label), "experience": str(experience),
            "prs": len(group), "unique_authors": int(group["user_id"].nunique()),
            "share_of_external_prs": len(group) / len(external_prs[external_prs["period"] == label]),
            "collaborator_response_within_7d_pct": float((eligible_7["collab_event"] & (eligible_7["collab_days"] <= 7)).mean()) if len(eligible_7) else np.nan,
            "reviewed_by_snapshot_collaborator_pct": float((group["collab_reviews"] > 0).mean()),
            "collaborator_reviews_per_pr": float(group["collab_reviews"].mean()),
            "eligible_for_90d_outcome": len(eligible_90),
            "merged_within_90d_pct": float((eligible_90["merged"] & (eligible_90["merge_days"] <= 90)).mean()) if len(eligible_90) else np.nan,
            "closed_unmerged_within_90d_pct": float((eligible_90["closed_unmerged"] & (eligible_90["close_days"] <= 90)).mean()) if len(eligible_90) else np.nan,
            "hardware_specific_pct": float(group["hardware_specific"].mean()),
            "review_intensive_pct": float(group["review_intensive"].mean()),
        })
    external_contributor_experience = pd.DataFrame(external_experience_rows)

    external_experience_by_type = (
        external_prs.groupby(["period", "experience", "work_type"], observed=True)
        .size().rename("prs").reset_index()
    )
    external_experience_by_type["share_within_experience"] = (
        external_experience_by_type["prs"]
        / external_experience_by_type.groupby(["period", "experience"], observed=True)["prs"].transform("sum")
    )

    frequency = (
        external_prs.groupby(["period", "user_id"], observed=True)
        .size().rename("prs").reset_index()
    )
    frequency["frequency_band"] = pd.cut(
        frequency["prs"], bins=[0, 1, 4, np.inf],
        labels=["One PR in period", "2–4 PRs in period", "5+ PRs in period"],
    )
    external_contributor_frequency = (
        frequency.groupby(["period", "frequency_band"], observed=True)
        .agg(authors=("user_id", "nunique"), prs=("prs", "sum"))
        .reset_index()
    )
    external_contributor_frequency["share_of_external_authors"] = (
        external_contributor_frequency["authors"]
        / external_contributor_frequency.groupby("period", observed=True)["authors"].transform("sum")
    )
    external_contributor_frequency["share_of_external_prs"] = (
        external_contributor_frequency["prs"]
        / external_contributor_frequency.groupby("period", observed=True)["prs"].transform("sum")
    )

    external_author_dates = (
        external_prs.sort_values(["user_id", "created_at", "id"])
        .groupby("user_id")["created_at"]
        .agg(
            first_pr_at="first",
            second_pr_at=lambda values: values.iloc[1] if len(values) > 1 else pd.NaT,
            observed_prs="size",
        )
        .reset_index()
    )
    external_author_dates["first_pr_period"] = period(external_author_dates["first_pr_at"])
    external_author_dates["days_to_second_pr"] = (
        external_author_dates["second_pr_at"] - external_author_dates["first_pr_at"]
    ).dt.total_seconds() / 86400
    retention_rows = []
    for label, group in external_author_dates.groupby("first_pr_period", observed=True):
        for horizon in (90, 180, 365):
            eligible = group[(CUTOFF - group["first_pr_at"]).dt.total_seconds() / 86400 >= horizon]
            returned = eligible["days_to_second_pr"].notna() & (eligible["days_to_second_pr"] <= horizon)
            retention_rows.append({
                "first_pr_period": str(label), "horizon_days": horizon,
                "first_time_external_authors": len(group), "eligible_authors": len(eligible),
                "returned_within_horizon": int(returned.sum()),
                "return_rate": float(returned.mean()) if len(eligible) else np.nan,
            })
    external_contributor_retention = pd.DataFrame(retention_rows)

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
        {"check": "May-18 snapshot collaborators with triage+", "value": len(collab_ids), "note": "May 18 permission roster, not a July or historical roster"},
        {"check": "May-18 snapshot collaborators with write+", "value": len(write_ids), "note": "May 18 permission roster, not a July or historical roster"},
        {"check": "bot actors", "value": len(bot_ids), "note": "base GitHub user.type plus conservative login inference for delta-only actors"},
        {"check": "submitted reviews missing event time", "value": int(reviews["submitted_at"].isna().sum()), "note": "retained in artifact-level burden; excluded from event-period ownership"},
        {"check": "main-branch commits without associated PR", "value": int(direct_main_commits["commits"].sum()), "note": "canonical default-branch history through July 31"},
        {"check": "snapshot checksum verified", "value": 1, "note": snapshot_sha256},
    ])
    merged_audit_rows = merged_input_audit.rename(columns={"check_name": "check"}).copy()
    merged_audit_rows["note"] = "merged-input compatibility audit"
    audit = pd.concat([audit, merged_audit_rows[["check", "value", "note"]]], ignore_index=True)

    tables = {
        "dataset_audit": audit,
        "direct_main_commits": direct_main_commits,
        "git_commit_identity_audit": git_commit_identity_audit,
        "monthly_overview": monthly,
        "period_summary": period_summary,
        "response_horizons": response,
        "response_by_author_role": response_by_author_role,
        "workload_mix": workload_mix,
        "classification_coverage": classification_coverage,
        "classification_stability_sensitivity": classification_stability_sensitivity,
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
        "pr_by_author_role_and_type": pr_by_author_role_and_type,
        "pr_author_role_summary": pr_author_role_summary,
        "pr_by_author_role_and_dimension": pr_by_author_role_and_dimension,
        "engineering_ownership": engineering_ownership,
        "merge_gatekeeping": merge_gatekeeping,
        "merge_actor_roles": merge_actor_roles,
        "path_area_by_author_role": path_area_by_author_role,
        "path_area_ownership": path_area_ownership,
        "review_ownership": review_ownership,
        "reviewer_specialization": reviewer_specialization,
        "reviewer_primary_work_type": reviewer_primary_work_type,
        "collaborator_portfolio": collaborator_portfolio,
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
        "external_contributor_experience": external_contributor_experience,
        "external_experience_by_type": external_experience_by_type,
        "external_contributor_frequency": external_contributor_frequency,
        "external_contributor_retention": external_contributor_retention,
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
    latest_pr = pr_outcomes[pr_outcomes["period"] == RECENT_PERIOD].sort_values("prs")
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
    action_pivot = maintainer_action_comparison.set_index("action")[["2025_monthly_mean", "2026_jan_jul_monthly_mean"]].rename(
        columns={"2025_monthly_mean": "2025", "2026_jan_jul_monthly_mean": "2026 Jan–Jul"}
    )
    action_pivot.plot(kind="barh", ax=axes[0], color=[COLORS["gray"], COLORS["green"]])
    axes[0].set_title("Observable non-author/operational actions")
    axes[0].set_xlabel("Monthly mean (events, not effort)")
    axes[0].set_ylabel("")
    recent_burden = review_burden_type[review_burden_type["period"] == RECENT_PERIOD].sort_values("review_submission_share")
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
    topic_pivot = topics[topics["period"].isin(["2025", RECENT_PERIOD])].pivot(index="topic", columns="period", values="share").fillna(0)
    topic_pivot = topic_pivot.sort_values(RECENT_PERIOD)
    topic_pivot[["2025", RECENT_PERIOD]].plot(kind="barh", ax=ax, color=[COLORS["gray"], COLORS["blue"]])
    ax.set_title("vLLM engineering topic signals")
    ax.set_xlabel("Share of PRs (multi-label)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(opt.figures / "engineering_topics.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    recent_gaps = verifier_gaps[(verifier_gaps["period"] == RECENT_PERIOD) & (verifier_gaps["dimension"] == "work_type")].sort_values("test_touched_pct")
    axes[0].barh(recent_gaps["name"], recent_gaps["test_touched_pct"] * 100, color=COLORS["green"])
    axes[0].set_title("Test-file signal in eligible 2026 PRs")
    axes[0].set_xlabel("Touches test files (%)")
    hardware_gaps = verifier_gaps[(verifier_gaps["period"] == RECENT_PERIOD) & (verifier_gaps["dimension"] == "hardware") & (verifier_gaps["prs"] >= 50)].sort_values("test_touched_pct")
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    recent_frequency = external_contributor_frequency[external_contributor_frequency["period"] == RECENT_PERIOD].copy()
    frequency_order = ["One PR in period", "2–4 PRs in period", "5+ PRs in period"]
    recent_frequency["frequency_band"] = pd.Categorical(recent_frequency["frequency_band"], frequency_order, ordered=True)
    recent_frequency = recent_frequency.sort_values("frequency_band")
    x = np.arange(len(recent_frequency))
    axes[0].bar(x - 0.18, recent_frequency["share_of_external_authors"] * 100, 0.36, label="Authors", color=COLORS["cyan"])
    axes[0].bar(x + 0.18, recent_frequency["share_of_external_prs"] * 100, 0.36, label="PRs", color=COLORS["navy"])
    axes[0].set_xticks(x, ["1", "2–4", "5+"])
    axes[0].set_title("External contribution frequency, 2026")
    axes[0].set_xlabel("PRs per author during period")
    axes[0].set_ylabel("Share (%)")
    axes[0].legend(fontsize=8)

    recent_experience = external_contributor_experience[external_contributor_experience["period"] == RECENT_PERIOD].copy()
    experience_order = ["First observed PR", "2nd–5th observed PR", "6th+ observed PR"]
    recent_experience["experience"] = pd.Categorical(recent_experience["experience"], experience_order, ordered=True)
    recent_experience = recent_experience.sort_values("experience")
    x = np.arange(len(recent_experience))
    axes[1].bar(x - 0.18, recent_experience["collaborator_response_within_7d_pct"] * 100, 0.36, label="Collaborator response ≤7d", color=COLORS["orange"])
    axes[1].bar(x + 0.18, recent_experience["merged_within_90d_pct"] * 100, 0.36, label="Merged ≤90d", color=COLORS["green"])
    axes[1].set_xticks(x, ["First", "2nd–5th", "6th+"])
    axes[1].set_title("Experience and external PR outcomes")
    axes[1].set_ylabel("Eligible PRs (%)")
    axes[1].legend(fontsize=8)

    retention = external_contributor_retention[external_contributor_retention["horizon_days"] == 90].copy()
    retention["first_pr_period"] = pd.Categorical(retention["first_pr_period"], PERIOD_ORDER, ordered=True)
    retention = retention.sort_values("first_pr_period")
    axes[2].bar(retention["first_pr_period"].astype(str), retention["return_rate"] * 100, color=COLORS["blue"])
    axes[2].set_title("Return after a first external PR")
    axes[2].set_ylabel("Second PR within 90 days (%)")
    axes[2].tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(opt.figures / "external_contributor_lifecycle.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    recent_complexity = pr_complexity[pr_complexity["period"] == RECENT_PERIOD].copy()
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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    recent_roles = pr_by_author_role_and_type[
        (pr_by_author_role_and_type["period"] == RECENT_PERIOD)
        & (pr_by_author_role_and_type["author_permission"] != "Bot")
    ]
    role_order = ["External human", "Snapshot triage-only", "Snapshot write+"]
    contribution = recent_roles.pivot(index="work_type", columns="author_permission", values="share_of_work_type").fillna(0)
    contribution = contribution.reindex(columns=[value for value in role_order if value in contribution.columns]).sort_values("External human")
    contribution.plot(kind="barh", stacked=True, ax=axes[0], color=[COLORS["cyan"], COLORS["orange"], COLORS["navy"]])
    axes[0].set_title("Who authored each 2026 PR type?")
    axes[0].set_xlabel("Share of PR type")
    axes[0].set_ylabel("")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8)
    composition = recent_roles.pivot(index="author_permission", columns="work_type", values="share_within_author_role").fillna(0)
    composition = composition.reindex([value for value in role_order if value in composition.index])
    composition.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20")
    axes[1].set_title("What does each author group work on?")
    axes[1].set_ylabel("Share of group PRs")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    fig.tight_layout()
    fig.savefig(opt.figures / "pr_authorship_by_type.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    recent_ownership = engineering_ownership[
        (engineering_ownership["period"] == RECENT_PERIOD)
        & (engineering_ownership["population"] == "Snapshot write+")
        & (engineering_ownership["dimension"] == "work_type")
        & (engineering_ownership["prs"] >= 20)
    ].sort_values("top_5_share")
    axes[0].barh(recent_ownership["name"], recent_ownership["top_5_share"] * 100, color=COLORS["orange"])
    axes[0].set_title("2026 write+ engineering concentration")
    axes[0].set_xlabel("Share authored by top five (%)")
    recent_review_ownership = review_ownership[
        (review_ownership["period"] == RECENT_PERIOD)
        & (review_ownership["dimension"] == "work_type")
        & (review_ownership["review_submissions"] >= 20)
    ].sort_values("top_5_share")
    axes[1].barh(recent_review_ownership["name"], recent_review_ownership["top_5_share"] * 100, color=COLORS["green"])
    axes[1].set_title("2026 review ownership concentration")
    axes[1].set_xlabel("Share submitted by top five reviewers (%)")
    fig.tight_layout()
    fig.savefig(opt.figures / "engineering_and_review_ownership.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    recent_area = path_area_by_author_role[
        (path_area_by_author_role["period"] == RECENT_PERIOD)
        & (path_area_by_author_role["author_permission"] != "Bot")
    ]
    area_counts = recent_area.groupby("path_area")["prs"].sum().sort_values()
    axes[0].barh(area_counts.index, area_counts.values, color=COLORS["blue"])
    axes[0].set_title("2026 PRs touching each path area")
    axes[0].set_xlabel("PR-area records (multi-label per PR)")
    area_roles = recent_area.pivot(index="path_area", columns="author_permission", values="share_of_area_prs").fillna(0).reindex(area_counts.index)
    area_roles = area_roles.reindex(columns=[value for value in role_order if value in area_roles.columns])
    area_roles.plot(kind="barh", stacked=True, ax=axes[1], color=[COLORS["cyan"], COLORS["orange"], COLORS["navy"]])
    axes[1].set_title("Who authored work in each path area?")
    axes[1].set_xlabel("Share of PR-area records")
    axes[1].set_ylabel("")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(opt.figures / "path_area_ownership.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    portfolio_pivot = collaborator_portfolio.pivot(index="period", columns="portfolio_type", values="share_of_snapshot_roster").fillna(0).reindex(PERIOD_ORDER)
    portfolio_pivot.plot(kind="bar", stacked=True, ax=axes[0], color=[COLORS["navy"], COLORS["cyan"], COLORS["orange"], COLORS["gray"]])
    axes[0].set_title("Snapshot-collaborator portfolio overlap")
    axes[0].set_ylabel("Share of snapshot roster")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(fontsize=8)
    specialization = reviewer_specialization.set_index("period").reindex(PERIOD_ORDER)
    axes[1].bar(PERIOD_ORDER, specialization["median_primary_work_type_share"] * 100, color=COLORS["green"])
    axes[1].set_title("Reviewer work-type specialization")
    axes[1].set_ylabel("Median share in primary PR type (%)")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(opt.figures / "collaborator_portfolios.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "snapshot": {
            "cutoff": str(CUTOFF),
            "sha256": snapshot_sha256,
        },
        "audit": audit.to_dict(orient="records"),
        "direct_main_commits": direct_main_commits.to_dict(orient="records"),
        "git_commit_identity_audit": git_commit_identity_audit.to_dict(orient="records"),
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
        "pr_by_author_role_and_type": pr_by_author_role_and_type.to_dict(orient="records"),
        "pr_author_role_summary": pr_author_role_summary.to_dict(orient="records"),
        "pr_by_author_role_and_dimension": pr_by_author_role_and_dimension.to_dict(orient="records"),
        "engineering_ownership": engineering_ownership.to_dict(orient="records"),
        "merge_gatekeeping": merge_gatekeeping.to_dict(orient="records"),
        "merge_actor_roles": merge_actor_roles.to_dict(orient="records"),
        "path_area_by_author_role": path_area_by_author_role.to_dict(orient="records"),
        "path_area_ownership": path_area_ownership.to_dict(orient="records"),
        "review_ownership": review_ownership.to_dict(orient="records"),
        "reviewer_specialization": reviewer_specialization.to_dict(orient="records"),
        "reviewer_primary_work_type": reviewer_primary_work_type.to_dict(orient="records"),
        "collaborator_portfolio": collaborator_portfolio.to_dict(orient="records"),
        "task_feasibility": feasibility_summary.to_dict(orient="records"),
        "topics": topics.to_dict(orient="records"),
        "classification_coverage": classification_coverage.to_dict(orient="records"),
        "classification_stability_sensitivity": classification_stability_sensitivity.to_dict(orient="records"),
        "verifier_gaps": verifier_gaps.to_dict(orient="records"),
        "pr_complexity": pr_complexity.to_dict(orient="records"),
        "backlog_age": backlog_age.to_dict(orient="records"),
        "current_issue_queue": current_issue_queue.to_dict(orient="records"),
        "current_pr_queue": current_pr_queue.to_dict(orient="records"),
        "issue_disposition_horizons": issue_disposition_horizons.to_dict(orient="records"),
        "contributors": contributors.to_dict(orient="records"),
        "external_contributor_experience": external_contributor_experience.to_dict(orient="records"),
        "external_experience_by_type": external_experience_by_type.to_dict(orient="records"),
        "external_contributor_frequency": external_contributor_frequency.to_dict(orient="records"),
        "external_contributor_retention": external_contributor_retention.to_dict(orient="records"),
        "pr_competing_risks": competing_risks.to_dict(orient="records"),
        "issue_closure_actor": issue_closure_actor.to_dict(orient="records"),
        "issue_outcomes": issue_outcomes.to_dict(orient="records"),
        "pr_outcomes": pr_outcomes.to_dict(orient="records"),
        "hardware_outcomes": hardware_outcomes.to_dict(orient="records"),
    }
    opt.summary.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
