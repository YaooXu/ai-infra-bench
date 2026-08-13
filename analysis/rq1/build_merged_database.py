#!/usr/bin/env python3
"""Build a provenance-preserving vLLM GitHub database through a fixed cutoff.

The output keeps the maintainer-provided Fivetran snapshot intact, copies the
raw normalized delta tables under a ``delta_`` prefix, and materializes a set
of cutoff-consistent ``canonical_`` tables for common analysis.  It never
mutates either input database and replaces the output atomically only after
all validation checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_CUTOFF = "2026-07-31T23:59:59Z"
DEFAULT_BASE_CUTOFF = "2026-05-18T20:02:21Z"
BASE_SHA256 = "1992a9f7011ebe35ba6f62511d5ccc727b233e21d7279db3d3496f9f4892c44d"
SOURCE_GIST = "https://gist.github.com/simon-mo/2b0f4e9f872d479a08ae53edac51ecb1"
REPOSITORY = "vllm-project/vllm"
SCHEMA_VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="May 18 Fivetran SQLite snapshot")
    parser.add_argument("--delta", type=Path, required=True, help="July 31 normalized delta SQLite")
    parser.add_argument("--output", type=Path, required=True, help="Merged SQLite output")
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--base-cutoff", default=DEFAULT_BASE_CUTOFF)
    parser.add_argument(
        "--skip-base-checksum",
        action="store_true",
        help="Allow a base snapshot other than the cited May 18 artifact",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_obj(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


def nested(obj: dict[str, Any], *keys: str) -> Any:
    value: Any = obj
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def before_or_at(value: str | None, cutoff: str) -> bool:
    return bool(value and value <= cutoff)


def issue_number_from_url(url: str | None) -> int | None:
    if not url:
        return None
    try:
        return int(url.rstrip("/").rsplit("/", 1)[-1])
    except ValueError:
        return None


def batched(rows: Iterable[tuple[Any, ...]], size: int = 2_000) -> Iterator[list[tuple[Any, ...]]]:
    batch: list[tuple[Any, ...]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def insert_many(
    conn: sqlite3.Connection,
    sql: str,
    rows: Iterable[tuple[Any, ...]],
    size: int = 2_000,
) -> None:
    for batch in batched(rows, size):
        conn.executemany(sql, batch)


def copy_delta_tables(conn: sqlite3.Connection) -> None:
    delta_tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM delta.sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    if not delta_tables:
        raise RuntimeError("delta database has no tables")
    for table in delta_tables:
        target = f"delta_{table}"
        conn.execute(f'DROP TABLE IF EXISTS "{target}"')
        conn.execute(f'CREATE TABLE "{target}" AS SELECT * FROM delta."{table}"')

    indexes = [
        ("delta_artifact_raw", "database_id"),
        ("delta_pull_request_raw", "database_id"),
        ("delta_issue_comment_raw", "id"),
        ("delta_review_comment_raw", "id"),
        ("delta_pull_request_review_raw", "database_id"),
        ("delta_timeline_event_raw", "event_id"),
        ("delta_default_branch_commit_raw", "commit_sha"),
    ]
    for table, column in indexes:
        conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "idx_{table}_{column}" ON "{table}"("{column}")')
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_delta_pr_commit_key "
        "ON delta_pull_request_commit_raw(pull_request_id, commit_sha)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_delta_pr_file_key "
        "ON delta_pull_request_file_raw(pull_request_id, path)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_delta_timeline_artifact_time "
        "ON delta_timeline_event_raw(artifact_id, created_at)"
    )


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS dataset_metadata;
        DROP TABLE IF EXISTS dataset_source;
        DROP TABLE IF EXISTS dataset_table_inventory;
        DROP TABLE IF EXISTS dataset_validation;
        DROP TABLE IF EXISTS canonical_artifact;
        DROP TABLE IF EXISTS canonical_pull_request;
        DROP TABLE IF EXISTS canonical_artifact_label;
        DROP TABLE IF EXISTS canonical_artifact_assignee;
        DROP TABLE IF EXISTS canonical_issue_comment;
        DROP TABLE IF EXISTS canonical_pull_request_review;
        DROP TABLE IF EXISTS canonical_review_comment;
        DROP TABLE IF EXISTS canonical_pull_request_commit;
        DROP TABLE IF EXISTS canonical_pull_request_file;
        DROP TABLE IF EXISTS canonical_maintenance_event;
        DROP TABLE IF EXISTS canonical_default_branch_commit;
        DROP VIEW IF EXISTS canonical_issue;
        DROP VIEW IF EXISTS canonical_pr_artifact;

        CREATE TABLE dataset_metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE dataset_source (
          source_id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          author TEXT,
          url TEXT NOT NULL,
          observed_cutoff TEXT,
          sha256 TEXT,
          citation TEXT NOT NULL,
          notes TEXT
        );
        CREATE TABLE dataset_table_inventory (
          table_name TEXT PRIMARY KEY,
          layer TEXT NOT NULL,
          row_count INTEGER NOT NULL,
          notes TEXT
        );
        CREATE TABLE dataset_validation (
          check_name TEXT PRIMARY KEY,
          expected_value TEXT,
          observed_value TEXT NOT NULL,
          passed INTEGER NOT NULL,
          notes TEXT
        );

        CREATE TABLE canonical_artifact (
          database_id INTEGER PRIMARY KEY,
          node_id TEXT,
          number INTEGER NOT NULL UNIQUE,
          artifact_type TEXT NOT NULL CHECK (artifact_type IN ('Issue', 'PullRequest')),
          repository TEXT NOT NULL,
          author_id INTEGER,
          author_login TEXT,
          author_association TEXT,
          title TEXT,
          body TEXT,
          created_at TEXT NOT NULL,
          updated_at_observed TEXT,
          state_at_cutoff TEXT NOT NULL,
          state_reason_at_cutoff TEXT,
          closed_at_cutoff TEXT,
          merged_at_cutoff TEXT,
          is_draft_at_cutoff INTEGER,
          locked_observed INTEGER,
          source_layer TEXT NOT NULL,
          representation_may_postdate_cutoff INTEGER NOT NULL,
          state_reconstructed_from_timeline INTEGER NOT NULL,
          last_maintenance_event_at_cutoff TEXT
        );
        CREATE INDEX idx_canonical_artifact_created ON canonical_artifact(created_at);
        CREATE INDEX idx_canonical_artifact_type_state ON canonical_artifact(artifact_type, state_at_cutoff);

        CREATE TABLE canonical_pull_request (
          database_id INTEGER PRIMARY KEY,
          artifact_id INTEGER NOT NULL UNIQUE REFERENCES canonical_artifact(database_id),
          created_at TEXT NOT NULL,
          updated_at_observed TEXT,
          closed_at_cutoff TEXT,
          merged_at_cutoff TEXT,
          is_draft_at_cutoff INTEGER,
          merge_commit_sha TEXT,
          base_ref TEXT,
          base_sha TEXT,
          head_ref TEXT,
          head_sha TEXT,
          source_layer TEXT NOT NULL,
          files_cutoff_stable INTEGER NOT NULL
        );

        CREATE TABLE canonical_artifact_label (
          artifact_id INTEGER NOT NULL REFERENCES canonical_artifact(database_id),
          label_name TEXT NOT NULL,
          label_color TEXT,
          PRIMARY KEY (artifact_id, label_name)
        );
        CREATE INDEX idx_canonical_label_name ON canonical_artifact_label(label_name);

        CREATE TABLE canonical_artifact_assignee (
          artifact_id INTEGER NOT NULL REFERENCES canonical_artifact(database_id),
          user_id INTEGER,
          login TEXT NOT NULL,
          PRIMARY KEY (artifact_id, login)
        );

        CREATE TABLE canonical_issue_comment (
          database_id INTEGER PRIMARY KEY,
          artifact_id INTEGER REFERENCES canonical_artifact(database_id),
          artifact_number INTEGER,
          author_id INTEGER,
          author_login TEXT,
          author_association TEXT,
          body TEXT,
          created_at TEXT NOT NULL,
          updated_at_observed TEXT,
          source_layer TEXT NOT NULL,
          representation_may_postdate_cutoff INTEGER NOT NULL
        );
        CREATE INDEX idx_canonical_issue_comment_artifact_time
          ON canonical_issue_comment(artifact_id, created_at);

        CREATE TABLE canonical_pull_request_review (
          database_id INTEGER PRIMARY KEY,
          pull_request_id INTEGER NOT NULL,
          author_id INTEGER,
          author_login TEXT,
          body TEXT,
          submitted_at TEXT,
          updated_at_observed TEXT,
          state TEXT,
          commit_sha TEXT,
          source_layer TEXT NOT NULL,
          representation_may_postdate_cutoff INTEGER NOT NULL,
          pull_request_present INTEGER NOT NULL
        );
        CREATE INDEX idx_canonical_review_pr_time
          ON canonical_pull_request_review(pull_request_id, submitted_at);

        CREATE TABLE canonical_review_comment (
          database_id INTEGER PRIMARY KEY,
          pull_request_id INTEGER REFERENCES canonical_pull_request(database_id),
          review_id INTEGER,
          parent_comment_id INTEGER,
          author_id INTEGER,
          author_login TEXT,
          author_association TEXT,
          body TEXT,
          path TEXT,
          created_at TEXT NOT NULL,
          updated_at_observed TEXT,
          source_layer TEXT NOT NULL,
          representation_may_postdate_cutoff INTEGER NOT NULL
        );
        CREATE INDEX idx_canonical_review_comment_pr_time
          ON canonical_review_comment(pull_request_id, created_at);

        CREATE TABLE canonical_pull_request_commit (
          pull_request_id INTEGER NOT NULL REFERENCES canonical_pull_request(database_id),
          commit_sha TEXT NOT NULL,
          authored_at TEXT,
          committed_at TEXT,
          author_name TEXT,
          author_email TEXT,
          author_login TEXT,
          committer_name TEXT,
          committer_email TEXT,
          committer_login TEXT,
          message TEXT,
          additions INTEGER,
          deletions INTEGER,
          changed_files INTEGER,
          source_layer TEXT NOT NULL,
          PRIMARY KEY (pull_request_id, commit_sha)
        );
        CREATE INDEX idx_canonical_pr_commit_sha ON canonical_pull_request_commit(commit_sha);

        CREATE TABLE canonical_pull_request_file (
          pull_request_id INTEGER NOT NULL REFERENCES canonical_pull_request(database_id),
          path TEXT NOT NULL,
          status TEXT,
          previous_path TEXT,
          additions INTEGER,
          deletions INTEGER,
          changes INTEGER,
          patch_available INTEGER NOT NULL,
          cutoff_stable INTEGER NOT NULL,
          PRIMARY KEY (pull_request_id, path)
        );
        CREATE INDEX idx_canonical_pr_file_path ON canonical_pull_request_file(path);

        CREATE TABLE canonical_maintenance_event (
          event_id TEXT PRIMARY KEY,
          artifact_id INTEGER NOT NULL REFERENCES canonical_artifact(database_id),
          event_type TEXT NOT NULL,
          created_at TEXT NOT NULL,
          actor_id INTEGER,
          actor_login TEXT,
          subject_id INTEGER,
          subject_login TEXT,
          label_name TEXT,
          commit_sha TEXT
        );
        CREATE INDEX idx_canonical_event_artifact_time
          ON canonical_maintenance_event(artifact_id, created_at);
        CREATE INDEX idx_canonical_event_type_time
          ON canonical_maintenance_event(event_type, created_at);

        CREATE TABLE canonical_default_branch_commit (
          commit_sha TEXT PRIMARY KEY,
          authored_at TEXT,
          committed_at TEXT NOT NULL,
          author_name TEXT,
          author_email TEXT,
          author_login TEXT,
          committer_name TEXT,
          committer_email TEXT,
          committer_login TEXT,
          message TEXT,
          additions INTEGER,
          deletions INTEGER,
          changed_files INTEGER,
          associated_pull_requests_json TEXT NOT NULL,
          is_direct INTEGER NOT NULL,
          source_layer TEXT NOT NULL
        );

        CREATE VIEW canonical_issue AS
          SELECT * FROM canonical_artifact WHERE artifact_type = 'Issue';
        CREATE VIEW canonical_pr_artifact AS
          SELECT * FROM canonical_artifact WHERE artifact_type = 'PullRequest';
        """
    )


def load_events(
    conn: sqlite3.Connection, cutoff: str
) -> tuple[dict[int, list[dict[str, Any]]], list[tuple[Any, ...]]]:
    by_artifact: dict[int, list[dict[str, Any]]] = defaultdict(list)
    canonical_rows: list[tuple[Any, ...]] = []
    pull_request_to_artifact = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT pr.database_id, a.database_id "
            "FROM delta_pull_request_raw pr JOIN delta_artifact_raw a USING(number)"
        )
    }
    for row in conn.execute(
        "SELECT event_id, artifact_id, event_type, created_at, raw_json "
        "FROM delta_timeline_event_raw WHERE created_at <= ? ORDER BY artifact_id, created_at, event_id",
        (cutoff,),
    ):
        obj = json_obj(row[4])
        obj.setdefault("__typename", row[2])
        obj.setdefault("id", row[0])
        obj.setdefault("createdAt", row[3])
        artifact_id = pull_request_to_artifact.get(row[1], row[1])
        by_artifact[artifact_id].append(obj)
        actor = obj.get("actor") or {}
        requested = obj.get("requestedReviewer") or obj.get("assignee") or {}
        label = obj.get("label") or {}
        commit = obj.get("commit") or {}
        canonical_rows.append(
            (
                row[0],
                artifact_id,
                row[2],
                row[3],
                as_int(actor.get("databaseId")),
                actor.get("login"),
                as_int(requested.get("databaseId")),
                requested.get("login"),
                label.get("name"),
                commit.get("oid"),
            )
        )
    return by_artifact, canonical_rows


def reconstruct_lifecycle(
    obj: dict[str, Any],
    artifact_type: str,
    events: list[dict[str, Any]],
    cutoff: str,
) -> tuple[str, str | None, str | None, int | None, str | None, int]:
    state = "OPEN"
    closed_at: str | None = None
    merged_at: str | None = None
    lifecycle_seen = False
    last_event: str | None = None
    priority = {"ClosedEvent": 1, "MergedEvent": 2, "ReopenedEvent": 3}
    ordered = sorted(
        events,
        key=lambda event: (
            event.get("createdAt") or "",
            priority.get(event.get("__typename") or "", 0),
            event.get("id") or "",
        ),
    )
    for event in ordered:
        event_type = event.get("__typename")
        created_at = event.get("createdAt")
        if created_at and created_at <= cutoff:
            last_event = max(last_event or created_at, created_at)
        if event_type == "ClosedEvent":
            lifecycle_seen = True
            state, closed_at = "CLOSED", created_at
        elif event_type == "MergedEvent":
            lifecycle_seen = True
            state, closed_at, merged_at = "MERGED", created_at, created_at
        elif event_type == "ReopenedEvent":
            lifecycle_seen = True
            state, closed_at = "OPEN", None

    readiness = [
        event
        for event in ordered
        if event.get("__typename") in {"ReadyForReviewEvent", "ConvertToDraftEvent"}
    ]
    is_draft: int | None = None
    if artifact_type == "PullRequest":
        if readiness:
            is_draft = int(readiness[0].get("__typename") == "ReadyForReviewEvent")
            for event in readiness:
                is_draft = int(event.get("__typename") == "ConvertToDraftEvent")
        else:
            is_draft = bool_int(obj.get("isDraft", obj.get("draft")))

    if not lifecycle_seen:
        raw_updated = obj.get("updatedAt") or obj.get("updated_at")
        if before_or_at(raw_updated, cutoff):
            raw_state = str(obj.get("state") or "OPEN").upper()
            state = "MERGED" if raw_state == "MERGED" else raw_state
            raw_closed = obj.get("closedAt") or obj.get("closed_at")
            raw_merged = obj.get("mergedAt") or nested(obj, "pull_request", "merged_at")
            closed_at = raw_closed if before_or_at(raw_closed, cutoff) else None
            merged_at = raw_merged if before_or_at(raw_merged, cutoff) else None
            if merged_at:
                state = "MERGED"
    return state, closed_at, merged_at, is_draft, last_event, int(lifecycle_seen)


def stateful_people_and_labels(
    obj: dict[str, Any], events: list[dict[str, Any]], cutoff: str
) -> tuple[dict[str, str | None], dict[str, int | None]]:
    updated = obj.get("updatedAt") or obj.get("updated_at")
    if before_or_at(updated, cutoff):
        raw_labels = obj.get("labels") or []
        label_nodes = raw_labels.get("nodes") or [] if isinstance(raw_labels, dict) else raw_labels
        raw_assignees = obj.get("assignees") or []
        assignee_nodes = (
            raw_assignees.get("nodes") or []
            if isinstance(raw_assignees, dict)
            else raw_assignees
        )
        labels = {
            item.get("name"): item.get("color")
            for item in label_nodes
            if isinstance(item, dict) and item.get("name")
        }
        assignees = {
            item.get("login"): as_int(item.get("databaseId", item.get("id")))
            for item in assignee_nodes
            if isinstance(item, dict) and item.get("login")
        }
        return labels, assignees

    labels: dict[str, str | None] = {}
    assignees: dict[str, int | None] = {}
    for event in sorted(events, key=lambda item: (item.get("createdAt") or "", item.get("id") or "")):
        event_type = event.get("__typename")
        if event_type in {"LabeledEvent", "UnlabeledEvent"}:
            label = event.get("label") or {}
            name = label.get("name")
            if name:
                if event_type == "LabeledEvent":
                    labels[name] = label.get("color")
                else:
                    labels.pop(name, None)
        elif event_type in {"AssignedEvent", "UnassignedEvent"}:
            assignee = event.get("assignee") or {}
            login = assignee.get("login")
            if login:
                if event_type == "AssignedEvent":
                    assignees[login] = as_int(assignee.get("databaseId"))
                else:
                    assignees.pop(login, None)
    return labels, assignees


def build_artifacts(
    conn: sqlite3.Connection,
    cutoff: str,
    events: dict[int, list[dict[str, Any]]],
) -> None:
    base_rows: dict[int, tuple[Any, ...]] = {}
    base_labels: dict[int, list[tuple[str, str | None]]] = defaultdict(list)
    base_assignees: dict[int, list[tuple[str, int | None]]] = defaultdict(list)

    for row in conn.execute(
        """
        SELECT i.id, i.number, CASE WHEN pr.id IS NOT NULL THEN 'PullRequest' ELSE 'Issue' END,
               i.user_id, u.login, i.title, i.body, i.created_at, i.updated_at,
               UPPER(i.state), i.state_reason, i.closed_at, i.locked,
               pr.id, pr.draft, im.merged_at
        FROM issue i
        LEFT JOIN user u ON u.id = i.user_id
        LEFT JOIN pull_request pr ON pr.issue_id = i.id
        LEFT JOIN (
          SELECT issue_id, MAX(merged_at) AS merged_at FROM issue_merged GROUP BY issue_id
        ) im ON im.issue_id = i.id
        WHERE i.created_at <= ?
        """,
        (cutoff,),
    ):
        state = "MERGED" if row[15] and row[15] <= cutoff else row[9]
        merged = row[15] if row[15] and row[15] <= cutoff else None
        closed = row[11] if row[11] and row[11] <= cutoff else None
        base_rows[row[0]] = (
            row[0], None, row[1], row[2], REPOSITORY, row[3], row[4], None,
            row[5], row[6], row[7], row[8], state, row[10], closed, merged,
            row[14] if row[2] == "PullRequest" else None, row[12], "base", 0, 0, None,
        )

    for row in conn.execute(
        """
        SELECT il.issue_id, COALESCE(il.label, l.name), l.color
        FROM issue_label il LEFT JOIN label l ON l.id = il.label_id
        WHERE COALESCE(il.label, l.name) IS NOT NULL
        """
    ):
        base_labels[row[0]].append((row[1], row[2]))
    for row in conn.execute(
        "SELECT ia.issue_id, u.login, ia.user_id FROM issue_assignee ia JOIN user u ON u.id=ia.user_id"
    ):
        base_assignees[row[0]].append((row[1], row[2]))

    delta_rows: dict[int, tuple[Any, ...]] = {}
    delta_labels: dict[int, list[tuple[str, str | None]]] = defaultdict(list)
    delta_assignees: dict[int, list[tuple[str, int | None]]] = defaultdict(list)
    for row in conn.execute(
        "SELECT database_id, node_id, number, artifact_type, created_at, updated_at, raw_json "
        "FROM delta_artifact_raw WHERE created_at <= ?",
        (cutoff,),
    ):
        obj = json_obj(row[6])
        artifact_events = events.get(row[0], [])
        state, closed, merged, is_draft, last_event, reconstructed = reconstruct_lifecycle(
            obj, row[3], artifact_events, cutoff
        )
        author = obj.get("author") or obj.get("user") or {}
        postdates = int(bool(row[5] and row[5] > cutoff))
        state_reason = obj.get("stateReason") or obj.get("state_reason")
        if postdates:
            state_reason = None
        delta_rows[row[0]] = (
            row[0], row[1], row[2], row[3], REPOSITORY,
            as_int(author.get("databaseId", author.get("id"))), author.get("login"),
            obj.get("authorAssociation") or obj.get("author_association"),
            obj.get("title"), obj.get("body"), row[4], row[5], state, state_reason,
            closed, merged, is_draft, bool_int(obj.get("locked")), "delta", postdates,
            reconstructed, last_event,
        )
        labels, assignees = stateful_people_and_labels(obj, artifact_events, cutoff)
        delta_labels[row[0]] = sorted(labels.items())
        delta_assignees[row[0]] = sorted((login, user_id) for login, user_id in assignees.items())

    merged_rows = {**base_rows, **delta_rows}
    insert_many(
        conn,
        "INSERT INTO canonical_artifact VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        merged_rows.values(),
    )
    label_rows: list[tuple[Any, ...]] = []
    assignee_rows: list[tuple[Any, ...]] = []
    for artifact_id in merged_rows:
        labels = delta_labels.get(artifact_id, base_labels.get(artifact_id, []))
        assignees = delta_assignees.get(artifact_id, base_assignees.get(artifact_id, []))
        label_rows.extend((artifact_id, name, color) for name, color in labels)
        assignee_rows.extend((artifact_id, user_id, login) for login, user_id in assignees)
    insert_many(conn, "INSERT OR IGNORE INTO canonical_artifact_label VALUES (?,?,?)", label_rows)
    insert_many(conn, "INSERT OR IGNORE INTO canonical_artifact_assignee VALUES (?,?,?)", assignee_rows)


def build_pull_requests(conn: sqlite3.Connection, cutoff: str) -> None:
    base: dict[int, tuple[Any, ...]] = {}
    for row in conn.execute(
        """
        SELECT pr.id, pr.issue_id, pr.created_at, pr.updated_at, ca.closed_at_cutoff,
               ca.merged_at_cutoff, ca.is_draft_at_cutoff, pr.merge_commit_sha,
               pr.base_ref, pr.base_sha, pr.head_ref, pr.head_sha
        FROM pull_request pr JOIN canonical_artifact ca ON ca.database_id=pr.issue_id
        WHERE pr.created_at <= ?
        """,
        (cutoff,),
    ):
        base[row[0]] = (*row, "base", 1)

    delta: dict[int, tuple[Any, ...]] = {}
    for row in conn.execute(
        "SELECT database_id, number, created_at, updated_at, raw_json "
        "FROM delta_pull_request_raw WHERE created_at <= ?",
        (cutoff,),
    ):
        obj = json_obj(row[4])
        artifact = conn.execute(
            "SELECT database_id, closed_at_cutoff, merged_at_cutoff, is_draft_at_cutoff "
            "FROM canonical_artifact WHERE number=? AND artifact_type='PullRequest'",
            (row[1],),
        ).fetchone()
        if not artifact:
            raise RuntimeError(f"PR #{row[1]} has no canonical artifact")
        stable = int(bool(row[3] and row[3] <= cutoff) or bool(artifact[2]))
        delta[row[0]] = (
            row[0], artifact[0], row[2], row[3], artifact[1], artifact[2], artifact[3],
            nested(obj, "mergeCommit", "oid"), obj.get("baseRefName"), obj.get("baseRefOid"),
            obj.get("headRefName"), obj.get("headRefOid"), "delta", stable,
        )
    rows = {**base, **delta}
    insert_many(conn, "INSERT INTO canonical_pull_request VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows.values())


def build_comments(conn: sqlite3.Connection, cutoff: str) -> None:
    issue_number_to_id = {
        row[0]: row[1] for row in conn.execute("SELECT number,database_id FROM canonical_artifact")
    }
    base_comments: dict[int, tuple[Any, ...]] = {}
    for row in conn.execute(
        """
        SELECT c.id,c.issue_id,i.number,c.user_id,u.login,c.body,c.created_at,c.updated_at
        FROM issue_comment c JOIN issue i ON i.id=c.issue_id LEFT JOIN user u ON u.id=c.user_id
        WHERE c.created_at <= ?
        """,
        (cutoff,),
    ):
        base_comments[row[0]] = (
            row[0], row[1], row[2], row[3], row[4], None, row[5], row[6], row[7], "base", 0
        )
    for row in conn.execute(
        "SELECT id, issue_url, created_at, updated_at, raw_json "
        "FROM delta_issue_comment_raw WHERE created_at <= ?",
        (cutoff,),
    ):
        obj = json_obj(row[4])
        number = issue_number_from_url(row[1])
        user = obj.get("user") or {}
        base_comments[row[0]] = (
            row[0], issue_number_to_id.get(number), number, as_int(user.get("id")), user.get("login"),
            obj.get("author_association"), obj.get("body"), row[2], row[3], "delta",
            int(bool(row[3] and row[3] > cutoff)),
        )
    insert_many(
        conn,
        "INSERT INTO canonical_issue_comment VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        base_comments.values(),
    )

    reviews: dict[int, tuple[Any, ...]] = {}
    for row in conn.execute(
        """
        SELECT r.id,r.pull_request_id,r.user_id,u.login,r.body,r.submitted_at,
               r.submitted_at,r.state,r.commit_sha,
               CASE WHEN pr.database_id IS NULL THEN 0 ELSE 1 END
        FROM pull_request_review r LEFT JOIN user u ON u.id=r.user_id
        LEFT JOIN canonical_pull_request pr ON pr.database_id=r.pull_request_id
        WHERE r.submitted_at IS NULL OR r.submitted_at <= ?
        """,
        (cutoff,),
    ):
        reviews[row[0]] = (*row[:9], "base", 0, row[9])
    for row in conn.execute(
        "SELECT d.database_id,d.pull_request_id,d.submitted_at,d.updated_at,d.state,d.raw_json "
        "FROM delta_pull_request_review_raw d "
        "JOIN canonical_pull_request pr ON pr.database_id=d.pull_request_id "
        "WHERE d.submitted_at IS NULL OR d.submitted_at <= ?",
        (cutoff,),
    ):
        obj = json_obj(row[5])
        author = obj.get("author") or {}
        reviews[row[0]] = (
            row[0], row[1], as_int(author.get("databaseId")), author.get("login"), obj.get("body"),
            row[2], row[3], row[4], nested(obj, "commit", "oid"), "delta",
            int(bool(row[3] and row[3] > cutoff)), 1,
        )
    insert_many(
        conn,
        "INSERT INTO canonical_pull_request_review VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        reviews.values(),
    )

    review_comments: dict[int, tuple[Any, ...]] = {}
    for row in conn.execute(
        """
        SELECT c.id,c.pull_request_id,c.pull_request_review_id,c.parent_comment_id,
               c.user_id,u.login,NULL,c.body,c.path,c.created_at,c.updated_at
        FROM pull_request_review_comments c LEFT JOIN user u ON u.id=c.user_id
        JOIN canonical_pull_request pr ON pr.database_id=c.pull_request_id
        WHERE c.created_at <= ?
        """,
        (cutoff,),
    ):
        review_comments[row[0]] = (*row, "base", 0)
    for row in conn.execute(
        "SELECT id,pull_request_url,created_at,updated_at,raw_json "
        "FROM delta_review_comment_raw WHERE created_at <= ?",
        (cutoff,),
    ):
        obj = json_obj(row[4])
        user = obj.get("user") or {}
        number = issue_number_from_url(row[1])
        pr = conn.execute(
            "SELECT pr.database_id FROM canonical_pull_request pr "
            "JOIN canonical_artifact a ON a.database_id=pr.artifact_id WHERE a.number=?",
            (number,),
        ).fetchone()
        review_comments[row[0]] = (
            row[0], pr[0] if pr else None, as_int(obj.get("pull_request_review_id")),
            as_int(obj.get("in_reply_to_id")), as_int(user.get("id")), user.get("login"),
            obj.get("author_association"), obj.get("body"), obj.get("path"), row[2], row[3],
            "delta", int(bool(row[3] and row[3] > cutoff)),
        )
    insert_many(
        conn,
        "INSERT INTO canonical_review_comment VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        review_comments.values(),
    )


def commit_fields(obj: dict[str, Any]) -> tuple[Any, ...]:
    commit = obj.get("commit") if isinstance(obj.get("commit"), dict) else obj
    graphql_author = commit.get("author") or {}
    graphql_committer = commit.get("committer") or {}
    rest_commit = commit.get("commit") or {}
    rest_author = rest_commit.get("author") or {}
    rest_committer = rest_commit.get("committer") or {}
    author_user = obj.get("author") if isinstance(obj.get("author"), dict) else {}
    committer_user = obj.get("committer") if isinstance(obj.get("committer"), dict) else {}
    return (
        commit.get("authoredDate") or rest_author.get("date"),
        commit.get("committedDate") or rest_committer.get("date"),
        graphql_author.get("name") or rest_author.get("name"),
        graphql_author.get("email") or rest_author.get("email"),
        nested(graphql_author, "user", "login") or author_user.get("login"),
        graphql_committer.get("name") or rest_committer.get("name"),
        graphql_committer.get("email") or rest_committer.get("email"),
        nested(graphql_committer, "user", "login") or committer_user.get("login"),
        commit.get("message") or rest_commit.get("message"),
        as_int(commit.get("additions") or nested(obj, "stats", "additions")),
        as_int(commit.get("deletions") or nested(obj, "stats", "deletions")),
        as_int(commit.get("changedFiles") or len(obj.get("files") or [])) or None,
    )


def build_commits_and_files(conn: sqlite3.Connection, cutoff: str) -> None:
    commits: dict[tuple[int, str], tuple[Any, ...]] = {}
    for row in conn.execute(
        """
        SELECT cp.pull_request_id,cp.commit_sha,c.author_date,c.committer_date,c.author_name,
               c.author_email,ua.login,c.committer_name,c.committer_email,uc.login,c.message
        FROM commit_pull_request cp LEFT JOIN "commit" c ON c.sha=cp.commit_sha
        JOIN canonical_pull_request pr ON pr.database_id=cp.pull_request_id
        LEFT JOIN user_email uea ON uea.email=c.author_email
        LEFT JOIN user ua ON ua.id=uea.user_id
        LEFT JOIN user_email uec ON uec.email=c.committer_email
        LEFT JOIN user uc ON uc.id=uec.user_id
        WHERE c.sha IS NULL OR c.committer_date <= ?
        """,
        (cutoff,),
    ):
        commits[(row[0], row[1])] = (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9],
            row[10], None, None, None, "base",
        )
    for row in conn.execute(
        "SELECT d.pull_request_id,d.commit_sha,d.committed_at,d.raw_json "
        "FROM delta_pull_request_commit_raw d "
        "JOIN canonical_pull_request pr ON pr.database_id=d.pull_request_id "
        "WHERE d.committed_at IS NULL OR d.committed_at <= ?",
        (cutoff,),
    ):
        fields = commit_fields(json_obj(row[3]))
        commits[(row[0], row[1])] = (
            row[0], row[1], fields[0], fields[1] or row[2], *fields[2:], "delta"
        )
    insert_many(
        conn,
        "INSERT INTO canonical_pull_request_commit VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        commits.values(),
    )

    stable = {
        row[0]: row[1] for row in conn.execute("SELECT database_id,files_cutoff_stable FROM canonical_pull_request")
    }
    file_rows: list[tuple[Any, ...]] = []
    for row in conn.execute(
        "SELECT d.pull_request_id,d.path,d.raw_json FROM delta_pull_request_file_raw d "
        "JOIN canonical_pull_request pr ON pr.database_id=d.pull_request_id"
    ):
        obj = json_obj(row[2])
        file_rows.append(
            (
                row[0], row[1], obj.get("status"), obj.get("previous_filename"),
                as_int(obj.get("additions")), as_int(obj.get("deletions")), as_int(obj.get("changes")),
                int(bool(obj.get("patch"))), stable.get(row[0], 0),
            )
        )
    insert_many(conn, "INSERT INTO canonical_pull_request_file VALUES (?,?,?,?,?,?,?,?,?)", file_rows)


def build_default_branch(conn: sqlite3.Connection, cutoff: str) -> None:
    commits: dict[str, tuple[Any, ...]] = {}
    for row in conn.execute(
        """
        SELECT c.sha,c.author_date,c.committer_date,c.author_name,c.author_email,ua.login,
               c.committer_name,c.committer_email,uc.login,c.message
        FROM branch_commit_relation b JOIN "commit" c ON c.sha=b.commit_sha
        LEFT JOIN user_email uea ON uea.email=c.author_email LEFT JOIN user ua ON ua.id=uea.user_id
        LEFT JOIN user_email uec ON uec.email=c.committer_email LEFT JOIN user uc ON uc.id=uec.user_id
        WHERE b.branch_name='main' AND c.committer_date <= ?
        """,
        (cutoff,),
    ):
        prs = [
            item[0]
            for item in conn.execute(
                "SELECT pull_request_id FROM commit_pull_request WHERE commit_sha=? ORDER BY pull_request_id",
                (row[0],),
            )
        ]
        commits[row[0]] = (*row, None, None, None, json.dumps(prs), int(not prs), "base")
    for row in conn.execute(
        "SELECT commit_sha,committed_at,is_direct,raw_json FROM delta_default_branch_commit_raw "
        "WHERE committed_at <= ?",
        (cutoff,),
    ):
        obj = json_obj(row[3])
        fields = commit_fields(obj)
        associated = nested(obj, "associatedPullRequests", "nodes") or []
        pr_numbers = [item.get("number") for item in associated if isinstance(item, dict)]
        commits[row[0]] = (
            row[0], fields[0], fields[1] or row[1], *fields[2:9], fields[9], fields[10], fields[11],
            json.dumps(pr_numbers), row[2], "delta",
        )
    insert_many(
        conn,
        "INSERT INTO canonical_default_branch_commit VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        commits.values(),
    )


def add_provenance(
    conn: sqlite3.Connection,
    base_sha: str,
    delta_sha: str,
    cutoff: str,
    base_cutoff: str,
) -> None:
    source_retrieved_through = conn.execute(
        "SELECT MAX(retrieved_at) FROM delta_request_file"
    ).fetchone()[0]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "repository": REPOSITORY,
        "base_cutoff": base_cutoff,
        "analysis_cutoff": cutoff,
        "source_retrieved_through": source_retrieved_through,
        "cutoff_semantics": "inclusive UTC",
        "text_semantics": (
            "For delta-refreshed records, title/body text is the representation observed during "
            "the August 2026 collection and may include edits after the analysis cutoff; use "
            "representation_may_postdate_cutoff to identify these rows."
        ),
        "file_semantics": (
            "canonical_pull_request_file contains PR-level aggregate diffs from the delta API. "
            "files_cutoff_stable is conservative: true only when the PR representation was not "
            "updated after cutoff or the PR had merged by cutoff. Base per-commit paths remain in commit_file."
        ),
        "maintainer_roster_semantics": (
            "repo_collaborator is the May 18 snapshot roster, not a historical roster and not refreshed by the public API."
        ),
    }
    conn.executemany("INSERT INTO dataset_metadata VALUES (?,?)", sorted(metadata.items()))
    conn.executemany(
        "INSERT INTO dataset_source VALUES (?,?,?,?,?,?,?,?)",
        [
            (
                "base-fivetran-2026-05-18",
                "vLLM GitHub Gym: vLLM GitHub Snapshot (Fivetran)",
                "Simon Mo",
                SOURCE_GIST,
                base_cutoff,
                base_sha,
                (
                    "Simon Mo. “vLLM GitHub Gym: vLLM GitHub Snapshot (Fivetran).” "
                    "GitHub Gist, May 18, 2026. " + SOURCE_GIST
                ),
                "Maintainer-provided base snapshot; original Fivetran tables are preserved unchanged.",
            ),
            (
                "github-api-delta-2026-07-31",
                "vLLM GitHub API delta through 2026-07-31",
                "AI Infra Bench",
                "https://github.com/ai-infra-bench/ai-infra-bench/tree/main/analysis/rq1",
                cutoff,
                delta_sha,
                (
                    "AI Infra Bench. “vLLM GitHub API delta through 2026-07-31.” "
                    "GitHub repository, 2026."
                ),
                "Collected from GitHub REST and GraphQL APIs; delta raw tables are prefixed delta_.",
            ),
        ],
    )


def inventory(conn: sqlite3.Connection) -> None:
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    rows = []
    for table in tables:
        if table.startswith("delta_"):
            layer = "delta_raw"
        elif table.startswith("canonical_"):
            layer = "canonical_cutoff"
        elif table.startswith("dataset_"):
            layer = "metadata"
        else:
            layer = "base_fivetran"
        count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        rows.append((table, layer, count, None))
    conn.executemany("INSERT OR REPLACE INTO dataset_table_inventory VALUES (?,?,?,?)", rows)
    inventory_rows = conn.execute("SELECT COUNT(*) FROM dataset_table_inventory").fetchone()[0]
    conn.execute(
        "UPDATE dataset_table_inventory SET row_count=? WHERE table_name='dataset_table_inventory'",
        (inventory_rows,),
    )


def validate(conn: sqlite3.Connection, cutoff: str) -> None:
    checks: list[tuple[str, str | None, str, int, str | None]] = []

    def check(name: str, expected: int | str | None, observed: int | str, notes: str | None = None) -> None:
        passed = expected is None or str(expected) == str(observed)
        checks.append((name, None if expected is None else str(expected), str(observed), int(passed), notes))

    base_artifacts = conn.execute("SELECT COUNT(*) FROM issue WHERE created_at <= ?", (cutoff,)).fetchone()[0]
    delta_artifacts = conn.execute(
        "SELECT COUNT(*) FROM delta_artifact_raw WHERE created_at <= ?", (cutoff,)
    ).fetchone()[0]
    overlap = conn.execute(
        "SELECT COUNT(*) FROM issue i JOIN delta_artifact_raw d ON d.database_id=i.id "
        "WHERE i.created_at <= ? AND d.created_at <= ?",
        (cutoff, cutoff),
    ).fetchone()[0]
    expected_artifacts = base_artifacts + delta_artifacts - overlap
    check(
        "canonical_artifact_union",
        expected_artifacts,
        conn.execute("SELECT COUNT(*) FROM canonical_artifact").fetchone()[0],
        "Deduplicated by GitHub issue database ID.",
    )
    expected_prs = conn.execute(
        "SELECT COUNT(*) FROM canonical_artifact WHERE artifact_type='PullRequest'"
    ).fetchone()[0]
    check("canonical_pull_request_coverage", expected_prs, conn.execute("SELECT COUNT(*) FROM canonical_pull_request").fetchone()[0])
    check(
        "base_pull_request_flag_mismatches",
        None,
        conn.execute(
            "SELECT COUNT(*) FROM pull_request pr JOIN issue i ON i.id=pr.issue_id "
            "WHERE COALESCE(i.pull_request,0)<>1"
        ).fetchone()[0],
        "Preserved and classified from pull_request table membership rather than the inconsistent flag.",
    )
    check(
        "artifacts_recovered_from_base_gap",
        None,
        conn.execute(
            "SELECT COUNT(*) FROM canonical_artifact a WHERE a.source_layer='delta' "
            "AND a.created_at <= (SELECT value FROM dataset_metadata WHERE key='base_cutoff') "
            "AND NOT EXISTS (SELECT 1 FROM issue i WHERE i.id=a.database_id)"
        ).fetchone()[0],
        "Artifacts created before the base cutoff but absent from the base snapshot.",
    )
    check(
        "orphan_canonical_pull_requests",
        0,
        conn.execute(
            "SELECT COUNT(*) FROM canonical_pull_request p LEFT JOIN canonical_artifact a "
            "ON a.database_id=p.artifact_id WHERE a.database_id IS NULL"
        ).fetchone()[0],
    )
    base_comments = conn.execute(
        "SELECT COUNT(*) FROM issue_comment WHERE created_at <= ?", (cutoff,)
    ).fetchone()[0]
    delta_comments = conn.execute(
        "SELECT COUNT(*) FROM delta_issue_comment_raw WHERE created_at <= ?", (cutoff,)
    ).fetchone()[0]
    comment_overlap = conn.execute(
        "SELECT COUNT(*) FROM issue_comment b JOIN delta_issue_comment_raw d ON d.id=b.id "
        "WHERE b.created_at <= ? AND d.created_at <= ?",
        (cutoff, cutoff),
    ).fetchone()[0]
    check(
        "canonical_issue_comment_union",
        base_comments + delta_comments - comment_overlap,
        conn.execute("SELECT COUNT(*) FROM canonical_issue_comment").fetchone()[0],
    )

    base_reviews = conn.execute(
        "SELECT COUNT(*) FROM pull_request_review WHERE submitted_at IS NULL OR submitted_at <= ?",
        (cutoff,),
    ).fetchone()[0]
    delta_reviews = conn.execute(
        "SELECT COUNT(*) FROM delta_pull_request_review_raw d "
        "JOIN canonical_pull_request p ON p.database_id=d.pull_request_id "
        "WHERE d.submitted_at IS NULL OR d.submitted_at <= ?",
        (cutoff,),
    ).fetchone()[0]
    review_overlap = conn.execute(
        "SELECT COUNT(*) FROM pull_request_review b JOIN delta_pull_request_review_raw d "
        "ON d.database_id=b.id JOIN canonical_pull_request p ON p.database_id=d.pull_request_id "
        "WHERE (b.submitted_at IS NULL OR b.submitted_at <= ?) "
        "AND (d.submitted_at IS NULL OR d.submitted_at <= ?)",
        (cutoff, cutoff),
    ).fetchone()[0]
    check(
        "canonical_pull_request_review_union",
        base_reviews + delta_reviews - review_overlap,
        conn.execute("SELECT COUNT(*) FROM canonical_pull_request_review").fetchone()[0],
        "Includes two base reviews whose referenced PR rows are absent; pull_request_present=0.",
    )

    base_review_comments = conn.execute(
        "SELECT COUNT(*) FROM pull_request_review_comments WHERE created_at <= ?", (cutoff,)
    ).fetchone()[0]
    delta_review_comments = conn.execute(
        "SELECT COUNT(*) FROM delta_review_comment_raw WHERE created_at <= ?", (cutoff,)
    ).fetchone()[0]
    review_comment_overlap = conn.execute(
        "SELECT COUNT(*) FROM pull_request_review_comments b JOIN delta_review_comment_raw d ON d.id=b.id "
        "WHERE b.created_at <= ? AND d.created_at <= ?",
        (cutoff, cutoff),
    ).fetchone()[0]
    check(
        "canonical_review_comment_union",
        base_review_comments + delta_review_comments - review_comment_overlap,
        conn.execute("SELECT COUNT(*) FROM canonical_review_comment").fetchone()[0],
    )

    base_pr_commits = conn.execute(
        "SELECT COUNT(*) FROM commit_pull_request cp "
        "JOIN canonical_pull_request p ON p.database_id=cp.pull_request_id "
        "LEFT JOIN \"commit\" c ON c.sha=cp.commit_sha "
        "WHERE c.sha IS NULL OR c.committer_date <= ?",
        (cutoff,),
    ).fetchone()[0]
    delta_pr_commits = conn.execute(
        "SELECT COUNT(*) FROM delta_pull_request_commit_raw d "
        "JOIN canonical_pull_request p ON p.database_id=d.pull_request_id "
        "WHERE d.committed_at IS NULL OR d.committed_at <= ?",
        (cutoff,),
    ).fetchone()[0]
    pr_commit_overlap = conn.execute(
        "SELECT COUNT(*) FROM commit_pull_request b JOIN delta_pull_request_commit_raw d "
        "ON d.pull_request_id=b.pull_request_id AND d.commit_sha=b.commit_sha "
        "JOIN canonical_pull_request p ON p.database_id=d.pull_request_id "
        "LEFT JOIN \"commit\" c ON c.sha=b.commit_sha "
        "WHERE (c.sha IS NULL OR c.committer_date <= ?) "
        "AND (d.committed_at IS NULL OR d.committed_at <= ?)",
        (cutoff, cutoff),
    ).fetchone()[0]
    check(
        "canonical_pull_request_commit_union",
        base_pr_commits + delta_pr_commits - pr_commit_overlap,
        conn.execute("SELECT COUNT(*) FROM canonical_pull_request_commit").fetchone()[0],
        "Includes one base PR-commit association whose commit metadata row is absent.",
    )
    check(
        "canonical_pull_request_file_coverage",
        conn.execute(
            "SELECT COUNT(*) FROM delta_pull_request_file_raw d "
            "JOIN canonical_pull_request p ON p.database_id=d.pull_request_id"
        ).fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM canonical_pull_request_file").fetchone()[0],
        "PR-level aggregate files exist only for delta-refreshed PRs; base per-commit files remain in commit_file.",
    )
    check(
        "canonical_maintenance_event_coverage",
        conn.execute(
            "SELECT COUNT(*) FROM delta_timeline_event_raw WHERE created_at <= ?", (cutoff,)
        ).fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM canonical_maintenance_event").fetchone()[0],
    )

    base_main_commits = conn.execute(
        "SELECT COUNT(DISTINCT b.commit_sha) FROM branch_commit_relation b JOIN \"commit\" c "
        "ON c.sha=b.commit_sha WHERE b.branch_name='main' AND c.committer_date <= ?",
        (cutoff,),
    ).fetchone()[0]
    delta_main_commits = conn.execute(
        "SELECT COUNT(*) FROM delta_default_branch_commit_raw WHERE committed_at <= ?", (cutoff,)
    ).fetchone()[0]
    main_overlap = conn.execute(
        "SELECT COUNT(DISTINCT b.commit_sha) FROM branch_commit_relation b JOIN \"commit\" c "
        "ON c.sha=b.commit_sha JOIN delta_default_branch_commit_raw d ON d.commit_sha=b.commit_sha "
        "WHERE b.branch_name='main' AND c.committer_date <= ? AND d.committed_at <= ?",
        (cutoff, cutoff),
    ).fetchone()[0]
    check(
        "canonical_default_branch_commit_union",
        base_main_commits + delta_main_commits - main_overlap,
        conn.execute("SELECT COUNT(*) FROM canonical_default_branch_commit").fetchone()[0],
    )
    check(
        "post_cutoff_artifacts",
        0,
        conn.execute("SELECT COUNT(*) FROM canonical_artifact WHERE created_at > ?", (cutoff,)).fetchone()[0],
    )
    check(
        "post_cutoff_comments",
        0,
        conn.execute("SELECT COUNT(*) FROM canonical_issue_comment WHERE created_at > ?", (cutoff,)).fetchone()[0],
    )
    check(
        "post_cutoff_reviews",
        0,
        conn.execute(
            "SELECT COUNT(*) FROM canonical_pull_request_review WHERE submitted_at > ?", (cutoff,)
        ).fetchone()[0],
    )
    check(
        "post_cutoff_review_comments",
        0,
        conn.execute("SELECT COUNT(*) FROM canonical_review_comment WHERE created_at > ?", (cutoff,)).fetchone()[0],
    )
    check(
        "post_cutoff_events",
        0,
        conn.execute("SELECT COUNT(*) FROM canonical_maintenance_event WHERE created_at > ?", (cutoff,)).fetchone()[0],
    )
    check(
        "post_cutoff_default_branch_commits",
        0,
        conn.execute(
            "SELECT COUNT(*) FROM canonical_default_branch_commit WHERE committed_at > ?", (cutoff,)
        ).fetchone()[0],
    )
    foreign_key_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    check(
        "sqlite_foreign_key_check",
        0,
        len(foreign_key_violations),
        json.dumps([tuple(row) for row in foreign_key_violations[:10]]),
    )
    conn.executemany("INSERT INTO dataset_validation VALUES (?,?,?,?,?)", checks)
    failed = [item for item in checks if not item[3]]
    if failed:
        details = "; ".join(
            f"{item[0]} expected={item[1]} observed={item[2]} notes={item[4]}"
            for item in failed
        )
        raise RuntimeError("validation failed: " + details)


def build(base: Path, delta: Path, output: Path, cutoff: str, base_cutoff: str, skip_checksum: bool) -> None:
    if not base.is_file() or not delta.is_file():
        raise FileNotFoundError("both --base and --delta must exist")
    base_sha = sha256_file(base)
    if not skip_checksum and base_sha != BASE_SHA256:
        raise RuntimeError(f"unexpected base SHA-256: {base_sha}; expected {BASE_SHA256}")
    delta_sha = sha256_file(delta)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temp = Path(temp_name)
    temp.unlink()
    try:
        shutil.copy2(base, temp)
        conn = sqlite3.connect(temp)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("ATTACH DATABASE ? AS delta", (str(delta),))
        copy_delta_tables(conn)
        conn.execute("DETACH DATABASE delta")
        create_schema(conn)
        events, canonical_event_rows = load_events(conn, cutoff)
        build_artifacts(conn, cutoff, events)
        build_pull_requests(conn, cutoff)
        build_comments(conn, cutoff)
        build_commits_and_files(conn, cutoff)
        insert_many(
            conn,
            "INSERT INTO canonical_maintenance_event VALUES (?,?,?,?,?,?,?,?,?,?)",
            canonical_event_rows,
        )
        build_default_branch(conn, cutoff)
        add_provenance(conn, base_sha, delta_sha, cutoff, base_cutoff)
        validate(conn, cutoff)
        inventory(conn)
        conn.commit()
        conn.execute("PRAGMA foreign_keys=ON")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity_check failed: {integrity}")
        conn.close()
        temp.replace(output)
    except Exception:
        temp.unlink(missing_ok=True)
        raise

    output_sha = sha256_file(output)
    print(json.dumps({
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": output_sha,
        "base_sha256": base_sha,
        "delta_sha256": delta_sha,
        "cutoff": cutoff,
    }, indent=2))


def main() -> int:
    args = parse_args()
    try:
        build(
            args.base.resolve(),
            args.delta.resolve(),
            args.output.resolve(),
            args.cutoff,
            args.base_cutoff,
            args.skip_base_checksum,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
