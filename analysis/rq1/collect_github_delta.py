#!/usr/bin/env python3
"""Collect a resumable, raw-preserving GitHub delta for the vLLM RQ1 study.

The collector intentionally stores every successful API response before it
normalizes anything. Authentication is read from ``gh auth token`` and is never
written to disk or logs. Raw output can contain public issue/PR text, commit
emails, and GitHub actor identifiers; keep the output directory untracked.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests


API = "https://api.github.com"
GRAPHQL = f"{API}/graphql"
PER_PAGE = 100
API_VERSION = "2022-11-28"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="vllm-project/vllm")
    parser.add_argument("--base-cutoff", default="2026-05-18T20:02:21Z")
    parser.add_argument("--cutoff", default="2026-07-31T23:59:59Z")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pause", type=float, default=0.08)
    parser.add_argument("--materialize-only", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def public_headers(headers: requests.structures.CaseInsensitiveDict[str]) -> dict[str, str]:
    keep = {
        "content-type", "date", "etag", "last-modified", "link",
        "x-github-api-version-selected", "x-github-request-id",
        "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset",
        "x-ratelimit-resource", "x-ratelimit-used",
    }
    return {key.lower(): value for key, value in headers.items() if key.lower() in keep}


def link_next(value: str | None) -> bool:
    return bool(value and re.search(r'rel="next"', value))


@dataclass
class Client:
    token: str
    output: Path
    pause: float

    def __post_init__(self) -> None:
        self.thread_local = threading.local()
        self.rate_lock = threading.Lock()
        self.block_until: dict[str, float] = {}

    def session(self) -> requests.Session:
        if hasattr(self.thread_local, "session"):
            return self.thread_local.session
        session = requests.Session()
        session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "ai-infra-bench-rq1-collector",
        })
        self.thread_local.session = session
        return session

    def request(
        self,
        method: str,
        url: str,
        destination: Path,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if destination.exists():
            return read_gzip_json(destination)
        attempts = 0
        while True:
            resource = "graphql" if url == GRAPHQL else "core"
            with self.rate_lock:
                delay = self.block_until.get(resource, 0.0) - time.time()
            if delay > 0:
                time.sleep(delay)
            attempts += 1
            try:
                response = self.session().request(method, url, params=params, json=payload, timeout=300)
            except (requests.ConnectionError, requests.Timeout):
                if attempts >= 10:
                    raise
                time.sleep(min(120.0, (2 ** attempts) + random.random() * 2))
                continue
            remaining = int(response.headers.get("X-RateLimit-Remaining", "1"))
            reset = int(response.headers.get("X-RateLimit-Reset", "0"))
            if response.status_code in {429, 502, 503, 504} or (
                response.status_code == 403
                and (remaining == 0 or "secondary rate limit" in response.text.lower())
            ):
                if attempts >= 10:
                    response.raise_for_status()
                if remaining == 0 and reset:
                    delay = max(1.0, reset - time.time() + 2.0)
                else:
                    delay = min(120.0, (2 ** attempts) + random.random() * 2)
                time.sleep(delay)
                continue
            response.raise_for_status()
            body = response.json()
            envelope = {
                "request": {
                    "method": method,
                    "url": response.url,
                    "payload": payload,
                },
                "response": {
                    "status": response.status_code,
                    "headers": public_headers(response.headers),
                    "body": body,
                },
                "retrieved_at": utc_now(),
            }
            atomic_gzip_json(destination, envelope)
            # Start waiting before exhaustion. This keeps resumable, multi-hour
            # crawls from persisting a rate-limit error at the canonical path.
            if remaining <= 100 and reset > time.time():
                with self.rate_lock:
                    self.block_until[response.headers.get("X-RateLimit-Resource", resource)] = reset + 2.0
            time.sleep(self.pause)
            return envelope

    def rest_pages(
        self,
        endpoint: str,
        directory: Path,
        params: dict[str, Any],
        max_pages: int = 99,
    ) -> tuple[list[Path], bool]:
        paths: list[Path] = []
        page = 1
        while True:
            path = directory / f"page_{page:05d}.json.gz"
            query = {**params, "per_page": PER_PAGE, "page": page}
            envelope = self.request("GET", f"{API}/{endpoint}", path, params=query)
            paths.append(path)
            body = envelope["response"]["body"]
            has_more = bool(body and link_next(envelope["response"]["headers"].get("link")))
            if not has_more:
                break
            if page >= max_pages:
                return paths, True
            page += 1
        return paths, False

    def rest_time_segments(
        self,
        endpoint: str,
        directory: Path,
        params: dict[str, Any],
        time_field: str = "updated_at",
    ) -> list[Path]:
        all_paths: list[Path] = []
        since = params["since"]
        segment = 1
        while True:
            segment_dir = directory / f"segment_{segment:05d}"
            paths, has_more = self.rest_pages(
                endpoint, segment_dir, {**params, "since": since}, max_pages=99,
            )
            all_paths.extend(paths)
            if not has_more:
                return all_paths
            last_rows = response_body(paths[-1])
            observed = [row.get(time_field) for row in last_rows if row.get(time_field)]
            if not observed:
                raise RuntimeError(f"Cannot segment {endpoint}: no {time_field} on final page")
            next_since = max(observed)
            if next_since <= since:
                raise RuntimeError(f"Cannot advance {endpoint}: {next_since} <= {since}")
            since = next_since
            segment += 1

    def graphql(self, query: str, variables: dict[str, Any], destination: Path) -> dict[str, Any]:
        envelope = self.request(
            "POST", GRAPHQL, destination,
            payload={"query": query, "variables": variables},
        )
        body = envelope["response"]["body"]
        if body.get("errors"):
            raise RuntimeError(f"GraphQL errors in {destination}: {body['errors']}")
        return envelope


ACTOR = """actor{login ... on User{databaseId} ... on Bot{databaseId} ... on Organization{databaseId}}"""
ASSIGNEE = """assignee{__typename ... on User{databaseId login} ... on Bot{databaseId login} ... on Mannequin{databaseId login}}"""
REVIEWER = """requestedReviewer{__typename ... on User{databaseId login} ... on Team{databaseId slug}}"""

COMMON_TIMELINE_FRAGMENTS = f"""
__typename
... on ClosedEvent{{id createdAt {ACTOR}}}
... on ReopenedEvent{{id createdAt {ACTOR}}}
... on LabeledEvent{{id createdAt {ACTOR} label{{id name}}}}
... on UnlabeledEvent{{id createdAt {ACTOR} label{{id name}}}}
... on AssignedEvent{{id createdAt {ACTOR} {ASSIGNEE}}}
... on UnassignedEvent{{id createdAt {ACTOR} {ASSIGNEE}}}
... on RenamedTitleEvent{{id createdAt {ACTOR} previousTitle currentTitle}}
... on ReferencedEvent{{id createdAt {ACTOR} commit{{oid}}}}
... on CrossReferencedEvent{{id createdAt {ACTOR} source{{__typename ... on Issue{{databaseId number}} ... on PullRequest{{databaseId number}}}}}}
... on MilestonedEvent{{id createdAt {ACTOR} milestoneTitle}}
... on DemilestonedEvent{{id createdAt {ACTOR} milestoneTitle}}
... on LockedEvent{{id createdAt {ACTOR} lockReason}}
... on UnlockedEvent{{id createdAt {ACTOR}}}
... on CommentDeletedEvent{{id createdAt {ACTOR}}}
... on MarkedAsDuplicateEvent{{id createdAt {ACTOR} isCrossRepository canonical{{__typename ... on Issue{{databaseId number}} ... on PullRequest{{databaseId number}}}} duplicate{{__typename ... on Issue{{databaseId number}} ... on PullRequest{{databaseId number}}}}}}
... on UnmarkedAsDuplicateEvent{{id createdAt {ACTOR} isCrossRepository canonical{{__typename ... on Issue{{databaseId number}} ... on PullRequest{{databaseId number}}}} duplicate{{__typename ... on Issue{{databaseId number}} ... on PullRequest{{databaseId number}}}}}}
... on TransferredEvent{{id createdAt {ACTOR} fromRepository{{nameWithOwner}}}}
... on ConvertedToDiscussionEvent{{id createdAt {ACTOR} discussion{{databaseId number}}}}
... on ConnectedEvent{{id createdAt {ACTOR} isCrossRepository source{{__typename ... on Issue{{databaseId number}} ... on PullRequest{{databaseId number}}}} subject{{__typename ... on Issue{{databaseId number}} ... on PullRequest{{databaseId number}}}}}}
... on DisconnectedEvent{{id createdAt {ACTOR} isCrossRepository source{{__typename ... on Issue{{databaseId number}} ... on PullRequest{{databaseId number}}}} subject{{__typename ... on Issue{{databaseId number}} ... on PullRequest{{databaseId number}}}}}}
... on IssueTypeChangedEvent{{id createdAt {ACTOR} issueType{{id name}} prevIssueType{{id name}}}}
... on BlockedByAddedEvent{{id createdAt {ACTOR} blockingIssue{{databaseId number}}}}
... on BlockedByRemovedEvent{{id createdAt {ACTOR} blockingIssue{{databaseId number}}}}
... on BlockingAddedEvent{{id createdAt {ACTOR} blockedIssue{{databaseId number}}}}
... on BlockingRemovedEvent{{id createdAt {ACTOR} blockedIssue{{databaseId number}}}}
... on SubIssueAddedEvent{{id createdAt {ACTOR} subIssue{{databaseId number}}}}
... on SubIssueRemovedEvent{{id createdAt {ACTOR} subIssue{{databaseId number}}}}
... on ParentIssueAddedEvent{{id createdAt {ACTOR} parent{{databaseId number}}}}
... on ParentIssueRemovedEvent{{id createdAt {ACTOR} parent{{databaseId number}}}}
"""

PR_TIMELINE_FRAGMENTS = COMMON_TIMELINE_FRAGMENTS + f"""
... on MergedEvent{{id createdAt {ACTOR} commit{{oid}}}}
... on ReadyForReviewEvent{{id createdAt {ACTOR}}}
... on ConvertToDraftEvent{{id createdAt {ACTOR}}}
... on ReviewRequestedEvent{{id createdAt {ACTOR} {REVIEWER}}}
... on ReviewRequestRemovedEvent{{id createdAt {ACTOR} {REVIEWER}}}
... on HeadRefForcePushedEvent{{id createdAt {ACTOR} beforeCommit{{oid}} afterCommit{{oid}}}}
... on BaseRefForcePushedEvent{{id createdAt {ACTOR} beforeCommit{{oid}} afterCommit{{oid}}}}
... on ReviewDismissedEvent{{id createdAt {ACTOR} previousReviewState dismissalMessage review{{databaseId id submittedAt author{{login ... on User{{databaseId}} ... on Bot{{databaseId}}}}}}}}
... on BaseRefChangedEvent{{id createdAt {ACTOR} previousRefName currentRefName}}
... on HeadRefDeletedEvent{{id createdAt {ACTOR} headRefName}}
... on HeadRefRestoredEvent{{id createdAt {ACTOR}}}
... on BaseRefDeletedEvent{{id createdAt {ACTOR} baseRefName}}
... on AutoMergeEnabledEvent{{id createdAt {ACTOR}}}
... on AutoMergeDisabledEvent{{id createdAt {ACTOR} reason reasonCode}}
... on AddedToMergeQueueEvent{{id createdAt {ACTOR}}}
... on RemovedFromMergeQueueEvent{{id createdAt {ACTOR} reason}}
"""

COMMON_TIMELINE_MINIMAL = """
__typename
... on ClosedEvent{id createdAt}
... on ReopenedEvent{id createdAt}
... on LabeledEvent{id createdAt}
... on UnlabeledEvent{id createdAt}
... on AssignedEvent{id createdAt}
... on UnassignedEvent{id createdAt}
... on RenamedTitleEvent{id createdAt}
... on ReferencedEvent{id createdAt}
... on CrossReferencedEvent{id createdAt}
... on MilestonedEvent{id createdAt}
... on DemilestonedEvent{id createdAt}
... on LockedEvent{id createdAt}
... on UnlockedEvent{id createdAt}
"""

PR_TIMELINE_MINIMAL = COMMON_TIMELINE_MINIMAL + """
... on MergedEvent{id createdAt}
... on ReadyForReviewEvent{id createdAt}
... on ConvertToDraftEvent{id createdAt}
... on ReviewRequestedEvent{id createdAt}
... on ReviewRequestRemovedEvent{id createdAt}
... on HeadRefForcePushedEvent{id createdAt}
... on BaseRefForcePushedEvent{id createdAt}
"""

ISSUE_TIMELINE_TYPES = """
CLOSED_EVENT,REOPENED_EVENT,LABELED_EVENT,UNLABELED_EVENT,ASSIGNED_EVENT,
UNASSIGNED_EVENT,RENAMED_TITLE_EVENT,REFERENCED_EVENT,CROSS_REFERENCED_EVENT,
MILESTONED_EVENT,DEMILESTONED_EVENT,LOCKED_EVENT,UNLOCKED_EVENT
,COMMENT_DELETED_EVENT,MARKED_AS_DUPLICATE_EVENT,UNMARKED_AS_DUPLICATE_EVENT,
TRANSFERRED_EVENT,CONVERTED_TO_DISCUSSION_EVENT,CONNECTED_EVENT,DISCONNECTED_EVENT,
ISSUE_TYPE_CHANGED_EVENT,BLOCKED_BY_ADDED_EVENT,BLOCKED_BY_REMOVED_EVENT,
BLOCKING_ADDED_EVENT,BLOCKING_REMOVED_EVENT,SUB_ISSUE_ADDED_EVENT,
SUB_ISSUE_REMOVED_EVENT,PARENT_ISSUE_ADDED_EVENT,PARENT_ISSUE_REMOVED_EVENT
"""

PR_TIMELINE_TYPES = ISSUE_TIMELINE_TYPES + """
,MERGED_EVENT,READY_FOR_REVIEW_EVENT,CONVERT_TO_DRAFT_EVENT,
REVIEW_REQUESTED_EVENT,REVIEW_REQUEST_REMOVED_EVENT,HEAD_REF_FORCE_PUSHED_EVENT,
BASE_REF_FORCE_PUSHED_EVENT
,REVIEW_DISMISSED_EVENT,BASE_REF_CHANGED_EVENT,
HEAD_REF_DELETED_EVENT,HEAD_REF_RESTORED_EVENT,BASE_REF_DELETED_EVENT,
AUTO_MERGE_ENABLED_EVENT,AUTO_MERGE_DISABLED_EVENT,ADDED_TO_MERGE_QUEUE_EVENT,
REMOVED_FROM_MERGE_QUEUE_EVENT
"""

REVIEW_NODE = """
databaseId id submittedAt updatedAt state body
commit{oid}
author{login ... on User{databaseId} ... on Bot{databaseId} ... on Organization{databaseId}}
"""

COMMIT_NODE = """
commit{
  oid authoredDate committedDate additions deletions changedFiles message
  author{name email date user{databaseId login}}
  committer{name email date user{databaseId login}}
  parents(first:20){nodes{oid}}
}
"""

FILE_NODE = """path additions deletions changeType"""

ARTIFACT_QUERY = f"""
query($ids:[ID!]!){{
  nodes(ids:$ids){{
    __typename
    ... on Issue{{
      id databaseId number createdAt updatedAt closedAt state stateReason title body locked
      author{{login ... on User{{databaseId}} ... on Bot{{databaseId}}}}
      labels(first:100){{nodes{{id name color description}}}}
      assignees(first:100){{nodes{{databaseId login}}}}
      timelineItems(first:100,itemTypes:[{ISSUE_TIMELINE_TYPES}]){{
        pageInfo{{hasNextPage endCursor}} nodes{{{COMMON_TIMELINE_FRAGMENTS}}}
      }}
    }}
    ... on PullRequest{{
      id databaseId number createdAt updatedAt closedAt mergedAt state isDraft title body locked
      baseRefName baseRefOid headRefName headRefOid mergeCommit{{oid}}
      author{{login ... on User{{databaseId}} ... on Bot{{databaseId}}}}
      labels(first:100){{nodes{{id name color description}}}}
      assignees(first:100){{nodes{{databaseId login}}}}
      reviewRequests(first:100){{nodes{{{REVIEWER}}}}}
      timelineItems(first:100,itemTypes:[{PR_TIMELINE_TYPES}]){{
        pageInfo{{hasNextPage endCursor}} nodes{{{PR_TIMELINE_FRAGMENTS}}}
      }}
    }}
  }}
  rateLimit{{cost remaining resetAt}}
}}
"""


def pull_request_connection_query(kind: str) -> str:
    if kind == "reviews":
        fragment = REVIEW_NODE
    elif kind == "commits":
        fragment = COMMIT_NODE
    elif kind == "files":
        fragment = FILE_NODE
    else:
        raise ValueError(kind)
    return f"""
query($ids:[ID!]!){{
  nodes(ids:$ids){{
    __typename
    ... on PullRequest{{
      id databaseId number
      {kind}(first:100){{pageInfo{{hasNextPage endCursor}} nodes{{{fragment}}}}}
    }}
  }}
  rateLimit{{cost remaining resetAt}}
}}
"""


def tail_query(kind: str, typename: str) -> str:
    if kind == "reviews":
        connection = f"reviews(first:100,after:$cursor){{pageInfo{{hasNextPage endCursor}} nodes{{{REVIEW_NODE}}}}}"
    elif kind == "commits":
        connection = f"commits(first:100,after:$cursor){{pageInfo{{hasNextPage endCursor}} nodes{{{COMMIT_NODE}}}}}"
    elif kind == "files":
        connection = f"files(first:100,after:$cursor){{pageInfo{{hasNextPage endCursor}} nodes{{{FILE_NODE}}}}}"
    elif kind == "timelineItems":
        types = PR_TIMELINE_TYPES if typename == "PullRequest" else ISSUE_TIMELINE_TYPES
        fragments = PR_TIMELINE_FRAGMENTS if typename == "PullRequest" else COMMON_TIMELINE_FRAGMENTS
        connection = f"timelineItems(first:100,after:$cursor,itemTypes:[{types}]){{pageInfo{{hasNextPage endCursor}} nodes{{{fragments}}}}}"
    else:
        raise ValueError(kind)
    return (
        "query($id:ID!,$cursor:String!){node(id:$id){... on %s{%s}} "
        "rateLimit{cost remaining resetAt}}" % (typename, connection)
    )


def response_body(path: Path) -> Any:
    return read_gzip_json(path)["response"]["body"]


def all_rest_rows(directory: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(directory.rglob("page_*.json.gz")):
        yield from response_body(path)


def collect_rest(client: Client, repo: str, base_cutoff: str) -> None:
    client.rest_time_segments(
        f"repos/{repo}/issues",
        client.output / "raw/rest/issues",
        {"state": "all", "since": base_cutoff, "sort": "updated", "direction": "asc"},
    )
    client.rest_time_segments(
        f"repos/{repo}/issues/comments",
        client.output / "raw/rest/issue_comments",
        {"since": base_cutoff, "sort": "updated", "direction": "asc"},
    )
    client.rest_time_segments(
        f"repos/{repo}/pulls/comments",
        client.output / "raw/rest/review_comments",
        {"since": base_cutoff, "sort": "updated", "direction": "asc"},
    )
    client.rest_pages(
        f"repos/{repo}/labels",
        client.output / "raw/rest/repository_labels",
        {},
        max_pages=1000,
    )


def artifact_index(output: Path, cutoff: str) -> list[dict[str, Any]]:
    end = parse_time(cutoff)
    rows_by_id: dict[int, dict[str, Any]] = {}
    for item in all_rest_rows(output / "raw/rest/issues"):
        created = parse_time(item.get("created_at"))
        if created and end and created <= end:
            rows_by_id[item["id"]] = {
                "node_id": item["node_id"],
                "database_id": item["id"],
                "number": item["number"],
                "typename": "PullRequest" if "pull_request" in item else "Issue",
            }
    rows = list(rows_by_id.values())
    rows.sort(key=lambda row: (row["number"], row["database_id"]))
    return rows


def collect_pull_request_files_rest(
    client: Client,
    repo: str,
    artifacts: list[dict[str, Any]],
    workers: int,
) -> None:
    pull_requests = [row for row in artifacts if row["typename"] == "PullRequest"]
    root = client.output / "raw/rest/pull_request_files"

    def fetch(row: dict[str, Any]) -> None:
        client.rest_pages(
            f"repos/{repo}/pulls/{row['number']}/files",
            root / f"pr_{row['number']:08d}",
            {},
            max_pages=1000,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, row) for row in pull_requests]
        for future in as_completed(futures):
            future.result()


def collect_capped_file_fallbacks(client: Client, repo: str) -> None:
    nodes, _ = flatten_graphql(client.output)
    by_number = {node["number"]: node for node in nodes if node["__typename"] == "PullRequest"}
    for directory in sorted((client.output / "raw/rest/pull_request_files").glob("pr_*")):
        rows = list(all_rest_rows(directory))
        if len(rows) < 3000:
            continue
        number = int(directory.name.removeprefix("pr_"))
        node = by_number[number]
        base, head = node["baseRefOid"], node["headRefOid"]
        destination = client.output / f"raw/git/pr_{number}_files.json.gz"
        if destination.exists():
            continue
        git_dir = client.output / f"raw/git/pr_{number}.git"
        if not git_dir.exists():
            subprocess.run(["git", "init", "--bare", str(git_dir)], check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", str(git_dir), "remote", "add", "origin", f"https://github.com/{repo}.git"],
                check=True, capture_output=True,
            )
        subprocess.run(
            ["git", "-C", str(git_dir), "-c", "protocol.version=2", "fetch", "--filter=blob:none",
             "--no-tags", "--depth=1", "origin", base, head],
            check=True, capture_output=True,
        )
        numstat_raw = subprocess.run(
            ["git", "-C", str(git_dir), "diff", "--numstat", "--no-renames", "-z", base, head],
            check=True, capture_output=True,
        ).stdout.decode("utf-8", errors="surrogateescape")
        status_raw = subprocess.run(
            ["git", "-C", str(git_dir), "diff", "--name-status", "--no-renames", "-z", base, head],
            check=True, capture_output=True,
        ).stdout.decode("utf-8", errors="surrogateescape")
        numstat: dict[str, dict[str, int | None]] = {}
        for record in numstat_raw.rstrip("\0").split("\0") if numstat_raw else []:
            additions, deletions, path = record.split("\t", 2)
            numstat[path] = {
                "additions": int(additions) if additions != "-" else None,
                "deletions": int(deletions) if deletions != "-" else None,
            }
        status_parts = status_raw.rstrip("\0").split("\0") if status_raw else []
        status = {status_parts[i + 1]: status_parts[i] for i in range(0, len(status_parts), 2)}
        files = [
            {"path": path, **stats, "status": status.get(path)}
            for path, stats in sorted(numstat.items())
        ]
        envelope = {
            "request": {
                "method": "git-diff",
                "url": f"https://github.com/{repo}.git",
                "payload": {"pull_request": number, "base": base, "head": head, "no_renames": True},
            },
            "response": {
                "status": 200,
                "headers": {},
                "body": {
                    "files": files,
                    "file_count": len(files),
                    "rest_endpoint_rows": len(rows),
                    "reason": "GitHub pull-files endpoint reached its 3000-file cap",
                },
            },
            "retrieved_at": utc_now(),
        }
        atomic_gzip_json(destination, envelope)


def collect_graphql(
    client: Client,
    artifacts: list[dict[str, Any]],
    batch_size: int,
    workers: int,
) -> None:
    def fetch_batches(
        rows: list[dict[str, Any]], directory: Path, query: str, size: int,
        recovery_directory: Path | None = None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        jobs = []
        for offset in range(0, len(rows), size):
            batch = rows[offset:offset + size]
            destination = directory / f"batch_{offset // size + 1:05d}.json.gz"
            if recovery_directory and all(
                (
                    (recovery_directory / f"pr_{row['number']:08d}.json.gz").exists()
                    or (
                        client.output / "raw/rest/pull_request_commits_recovery"
                        / f"pr_{row['number']:08d}" / "page_00001.json.gz"
                    ).exists()
                )
                for row in batch
            ):
                continue
            jobs.append((destination, {"ids": [row["node_id"] for row in batch]}))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(client.graphql, query, variables, destination) for destination, variables in jobs]
            for future in as_completed(futures):
                future.result()

    initial = client.output / "raw/graphql/artifacts"
    fetch_batches(artifacts, initial, ARTIFACT_QUERY, min(batch_size, 25))

    pull_requests = [row for row in artifacts if row["typename"] == "PullRequest"]
    # Reviews are sparse enough to batch broadly. Commit and file connections
    # are intentionally smaller to avoid GitHub query timeouts on large PRs.
    fetch_batches(
        pull_requests, client.output / "raw/graphql/reviews",
        pull_request_connection_query("reviews"), batch_size,
    )
    fetch_batches(
        pull_requests, client.output / "raw/graphql/commits",
        pull_request_connection_query("commits"), min(batch_size, 10),
        client.output / "raw/graphql/commits_recovery",
    )

    tails = client.output / "raw/graphql/tails"
    tails.mkdir(parents=True, exist_ok=True)
    tail_jobs = []
    connection_directories = {
        "timelineItems": initial,
        "reviews": client.output / "raw/graphql/reviews",
        "commits": client.output / "raw/graphql/commits",
    }
    for kind, directory in connection_directories.items():
        paths = list(directory.glob("batch_*.json.gz"))
        if kind == "commits":
            paths.extend((client.output / "raw/graphql/commits_recovery").glob("pr_*.json.gz"))
        for batch_path in sorted(paths):
            nodes = response_body(batch_path)["data"]["nodes"]
            for node in nodes:
                if not node:
                    continue
                connection = node.get(kind)
                if connection and connection["pageInfo"].get("hasNextPage", False):
                    tail_jobs.append((node, kind, connection["pageInfo"].get("endCursor")))

    def fetch_tail(node: dict[str, Any], kind: str, cursor: str) -> None:
        page = 2
        has_next = True
        while has_next:
            destination = tails / f"{node['__typename'].lower()}_{node['number']}_{kind}_{page:04d}.json.gz"
            envelope = client.graphql(
                tail_query(kind, node["__typename"]),
                {"id": node["id"], "cursor": cursor},
                destination,
            )
            tail_node = envelope["response"]["body"]["data"]["node"]
            connection = tail_node[kind]
            cursor = connection["pageInfo"].get("endCursor")
            has_next = connection["pageInfo"].get("hasNextPage", False)
            page += 1

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_tail, node, kind, cursor) for node, kind, cursor in tail_jobs]
        for future in as_completed(futures):
            future.result()


DEFAULT_BRANCH_HISTORY_QUERY = """
query($owner:String!,$name:String!,$since:GitTimestamp!,$until:GitTimestamp!,$cursor:String){
  repository(owner:$owner,name:$name){
    defaultBranchRef{
      name
      target{
        ... on Commit{
          history(first:100,after:$cursor,since:$since,until:$until){
            pageInfo{hasNextPage endCursor}
            nodes{
              oid authoredDate committedDate additions deletions changedFiles message
              author{name email date user{databaseId login}}
              committer{name email date user{databaseId login}}
              parents(first:20){nodes{oid}}
              associatedPullRequests(first:20){nodes{databaseId number mergedAt}}
            }
          }
        }
      }
    }
  }
  rateLimit{cost remaining resetAt}
}
"""


def collect_default_branch_history(
    client: Client,
    repo: str,
    base_cutoff: str,
    cutoff: str,
    workers: int,
) -> None:
    owner, name = repo.split("/", 1)
    directory = client.output / "raw/graphql/default_branch_history"
    cursor: str | None = None
    page = 1
    direct_commits: list[str] = []
    while True:
        destination = directory / f"page_{page:05d}.json.gz"
        envelope = client.graphql(
            DEFAULT_BRANCH_HISTORY_QUERY,
            {"owner": owner, "name": name, "since": base_cutoff, "until": cutoff, "cursor": cursor},
            destination,
        )
        history = envelope["response"]["body"]["data"]["repository"]["defaultBranchRef"]["target"]["history"]
        for commit in history["nodes"]:
            if not commit["associatedPullRequests"]["nodes"]:
                direct_commits.append(commit["oid"])
        if not history["pageInfo"]["hasNextPage"]:
            break
        cursor = history["pageInfo"]["endCursor"]
        page += 1

    detail_dir = client.output / "raw/rest/default_branch_direct_commits"
    def fetch(sha: str) -> None:
        client.request("GET", f"{API}/repos/{repo}/commits/{sha}", detail_dir / f"{sha}.json.gz")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, sha) for sha in sorted(set(direct_commits))]
        for future in as_completed(futures):
            future.result()


def collect_repo_state(client: Client, repo: str, cutoff: str) -> None:
    directory = client.output / "raw/rest/repository_state"
    commit = client.request(
        "GET", f"{API}/repos/{repo}/commits", directory / "default_branch_commit.json.gz",
        params={"until": cutoff, "per_page": 1},
    )["response"]["body"][0]
    tree_sha = commit["sha"]
    client.request(
        "GET", f"{API}/repos/{repo}/git/trees/{tree_sha}", directory / "tree_recursive.json.gz",
        params={"recursive": 1},
    )
    client.request(
        "GET", f"{API}/repos/{repo}", directory / "repository.json.gz",
    )
    client.request(
        "GET", f"{API}/gists/2b0f4e9f872d479a08ae53edac51ecb1",
        client.output / "raw/rest/source_gist/gist.json.gz",
    )


def flatten_graphql(output: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    nodes: list[dict[str, Any]] = []
    tails: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in sorted((output / "raw/graphql/artifacts").glob("batch_*.json.gz")):
        nodes.extend(node for node in response_body(path)["data"]["nodes"] if node)
    by_id = {node["id"]: node for node in nodes}
    for kind in ["reviews", "commits"]:
        paths = list((output / f"raw/graphql/{kind}").glob("batch_*.json.gz"))
        if kind == "commits":
            paths.extend((output / "raw/graphql/commits_recovery").glob("pr_*.json.gz"))
        for path in sorted(paths):
            for connection_node in response_body(path)["data"]["nodes"]:
                if connection_node and connection_node["id"] in by_id:
                    by_id[connection_node["id"]][kind] = connection_node[kind]
    by_number = {node["number"]: node for node in nodes if node["__typename"] == "PullRequest"}
    for directory in sorted((output / "raw/rest/pull_request_commits_recovery").glob("pr_*")):
        number = int(directory.name.removeprefix("pr_"))
        if number not in by_number:
            continue
        commits = []
        for row in all_rest_rows(directory):
            raw_commit = row["commit"]
            commits.append({
                "commit": {
                    "oid": row["sha"],
                    "authoredDate": raw_commit["author"].get("date"),
                    "committedDate": raw_commit["committer"].get("date"),
                    "message": raw_commit.get("message"),
                    "author": {
                        **raw_commit["author"],
                        "user": ({"databaseId": row["author"]["id"], "login": row["author"]["login"]} if row.get("author") else None),
                    },
                    "committer": {
                        **raw_commit["committer"],
                        "user": ({"databaseId": row["committer"]["id"], "login": row["committer"]["login"]} if row.get("committer") else None),
                    },
                    "parents": {"nodes": [{"oid": parent["sha"]} for parent in row.get("parents", [])]},
                    "source": "rest_fallback",
                }
            })
        by_number[number]["commits"] = {
            "nodes": commits,
            "pageInfo": {"hasNextPage": False, "endCursor": None},
        }
    for path in sorted((output / "raw/graphql/tails").glob("*.json.gz")):
        match = re.match(r"(issue|pullrequest)_(\d+)_(\w+)_(\d+)\.json\.gz", path.name)
        if not match:
            continue
        typename = "PullRequest" if match.group(1) == "pullrequest" else "Issue"
        number, kind = match.group(2), match.group(3)
        node = response_body(path)["data"]["node"]
        tails.setdefault((f"{typename}:{number}", kind), []).extend(node[kind]["nodes"])
    return nodes, tails


def validate_collection(output: Path, cutoff: str) -> dict[str, Any]:
    expected = artifact_index(output, cutoff)
    expected_nodes = {row["node_id"] for row in expected}
    expected_pr_nodes = {row["node_id"] for row in expected if row["typename"] == "PullRequest"}
    expected_pr_numbers = {row["number"] for row in expected if row["typename"] == "PullRequest"}

    observed_nodes: set[str] = set()
    for path in sorted((output / "raw/graphql/artifacts").glob("batch_*.json.gz")):
        observed_nodes.update(node["id"] for node in response_body(path)["data"]["nodes"] if node)

    connection_nodes: dict[str, set[str]] = {}
    for kind in ["reviews", "commits"]:
        ids: set[str] = set()
        paths = list((output / f"raw/graphql/{kind}").glob("batch_*.json.gz"))
        if kind == "commits":
            paths.extend((output / "raw/graphql/commits_recovery").glob("pr_*.json.gz"))
        for path in sorted(paths):
            ids.update(node["id"] for node in response_body(path)["data"]["nodes"] if node)
        connection_nodes[kind] = ids
    commit_recovery_profiles = {"full": 0, "lightweight": 0, "minimal": 0}
    for path in (output / "raw/graphql/commits_recovery").glob("pr_*.json.gz"):
        node = next((node for node in response_body(path)["data"]["nodes"] if node), None)
        commits = node.get("commits", {}).get("nodes", []) if node else []
        fields = set(commits[0]["commit"]) if commits else set()
        if "additions" in fields:
            commit_recovery_profiles["full"] += 1
        elif "message" in fields:
            commit_recovery_profiles["lightweight"] += 1
        else:
            commit_recovery_profiles["minimal"] += 1
    rest_commit_fallback_numbers = {
        int(path.parent.name.removeprefix("pr_"))
        for path in (output / "raw/rest/pull_request_commits_recovery").glob("pr_*/page_00001.json.gz")
    }
    rest_commit_fallback_nodes = {
        row["node_id"] for row in expected
        if row["number"] in rest_commit_fallback_numbers and row["typename"] == "PullRequest"
    }
    connection_nodes["commits"].update(rest_commit_fallback_nodes)

    file_numbers = {
        int(path.parent.name.removeprefix("pr_"))
        for path in (output / "raw/rest/pull_request_files").glob("pr_*/page_00001.json.gz")
    }
    incomplete_file_connections = 0
    capped_file_numbers: set[int] = set()
    for directory in (output / "raw/rest/pull_request_files").glob("pr_*"):
        pages = sorted(directory.glob("page_*.json.gz"))
        row_count = sum(len(response_body(page)) for page in pages)
        if row_count >= 3000:
            capped_file_numbers.add(int(directory.name.removeprefix("pr_")))
        if pages and link_next(read_gzip_json(pages[-1])["response"]["headers"].get("link")):
            incomplete_file_connections += 1
    git_file_fallback_numbers = {
        int(path.name.removeprefix("pr_").removesuffix("_files.json.gz"))
        for path in (output / "raw/git").glob("pr_*_files.json.gz")
    }

    incomplete_tails: list[str] = []
    tail_groups: dict[str, list[Path]] = {}
    for path in (output / "raw/graphql/tails").glob("*.json.gz"):
        group = path.name.rsplit("_", 1)[0]
        tail_groups.setdefault(group, []).append(path)
    expected_tail_groups: set[str] = set()
    connection_sources: list[tuple[str, list[Path]]] = [
        ("timelineItems", list((output / "raw/graphql/artifacts").glob("batch_*.json.gz"))),
        ("reviews", list((output / "raw/graphql/reviews").glob("batch_*.json.gz"))),
        ("commits", list((output / "raw/graphql/commits").glob("batch_*.json.gz"))
         + list((output / "raw/graphql/commits_recovery").glob("pr_*.json.gz"))),
    ]
    for kind, paths in connection_sources:
        for path in paths:
            for node in response_body(path)["data"]["nodes"]:
                if node and node.get(kind, {}).get("pageInfo", {}).get("hasNextPage"):
                    expected_tail_groups.add(f"{node['__typename'].lower()}_{node['number']}_{kind}")
    for group, paths in tail_groups.items():
        last = sorted(paths)[-1]
        match = re.match(r"(issue|pullrequest)_(\d+)_(\w+)_", last.name)
        if not match:
            incomplete_tails.append(group)
            continue
        kind = match.group(3)
        node = response_body(last)["data"]["node"]
        if node[kind]["pageInfo"].get("hasNextPage"):
            incomplete_tails.append(group)

    direct_commit_shas: set[str] = set()
    history_pages = sorted((output / "raw/graphql/default_branch_history").glob("page_*.json.gz"))
    for path in history_pages:
        history = response_body(path)["data"]["repository"]["defaultBranchRef"]["target"]["history"]
        direct_commit_shas.update(
            commit["oid"] for commit in history["nodes"]
            if not commit["associatedPullRequests"]["nodes"]
        )
    observed_direct_commit_shas = {
        path.name.removesuffix(".json.gz")
        for path in (output / "raw/rest/default_branch_direct_commits").glob("*.json.gz")
    }
    incomplete_default_branch_history = 0
    if history_pages:
        history = response_body(history_pages[-1])["data"]["repository"]["defaultBranchRef"]["target"]["history"]
        incomplete_default_branch_history = int(history["pageInfo"].get("hasNextPage", False))
    repository_state_files = {
        path.name for path in (output / "raw/rest/repository_state").glob("*.json.gz")
    }
    expected_repository_state_files = {
        "default_branch_commit.json.gz", "tree_recursive.json.gz", "repository.json.gz",
    }
    label_pages = sorted((output / "raw/rest/repository_labels").glob("page_*.json.gz"))
    incomplete_label_pages = int(bool(
        label_pages and link_next(read_gzip_json(label_pages[-1])["response"]["headers"].get("link"))
    ))

    canonical_files = [
        path for path in (output / "raw").rglob("*.json.gz")
        if not any(part.startswith("audit") or part == "failed" for part in path.parts)
    ]
    invalid_responses: list[str] = []
    for path in canonical_files:
        envelope = read_gzip_json(path)
        response = envelope.get("response", {})
        if not 200 <= int(response.get("status", 0)) < 300:
            invalid_responses.append(str(path.relative_to(output)))
        body = response.get("body")
        if isinstance(body, dict) and body.get("errors"):
            invalid_responses.append(str(path.relative_to(output)))

    checks = {
        "expected_artifacts": len(expected_nodes),
        "observed_artifacts": len(observed_nodes),
        "missing_artifacts": len(expected_nodes - observed_nodes),
        "extra_artifacts": len(observed_nodes - expected_nodes),
        "expected_pull_requests": len(expected_pr_nodes),
        "review_connection_pull_requests": len(connection_nodes["reviews"]),
        "commit_connection_pull_requests": len(connection_nodes["commits"]),
        "file_connection_pull_requests": len(file_numbers),
        "missing_review_connections": len(expected_pr_nodes - connection_nodes["reviews"]),
        "missing_commit_connections": len(expected_pr_nodes - connection_nodes["commits"]),
        "rest_commit_fallback_pull_requests": len(rest_commit_fallback_nodes),
        "graphql_commit_recovery_profiles": commit_recovery_profiles,
        "missing_file_connections": len(expected_pr_numbers - file_numbers),
        "incomplete_file_connections": incomplete_file_connections,
        "capped_file_connections": len(capped_file_numbers),
        "git_file_fallback_connections": len(git_file_fallback_numbers),
        "missing_capped_file_fallbacks": len(capped_file_numbers - git_file_fallback_numbers),
        "tail_connections": len(tail_groups),
        "expected_tail_connections": len(expected_tail_groups),
        "missing_tail_connections": len(expected_tail_groups - set(tail_groups)),
        "incomplete_tail_connections": len(incomplete_tails),
        "default_branch_history_pages": len(history_pages),
        "missing_default_branch_history": int(not history_pages),
        "incomplete_default_branch_history": incomplete_default_branch_history,
        "direct_default_branch_commits": len(direct_commit_shas),
        "missing_direct_commit_details": len(direct_commit_shas - observed_direct_commit_shas),
        "missing_repository_state_files": len(expected_repository_state_files - repository_state_files),
        "missing_repository_label_pages": int(not label_pages),
        "incomplete_repository_label_pages": incomplete_label_pages,
        "missing_source_gist_metadata": int(not (output / "raw/rest/source_gist/gist.json.gz").exists()),
        "canonical_response_files": len(canonical_files),
        "invalid_canonical_responses": len(set(invalid_responses)),
    }
    failures = {
        key: value for key, value in checks.items()
        if key.startswith(("missing_", "extra_", "incomplete_", "invalid_")) and value
    }
    if failures:
        raise RuntimeError(f"Collection validation failed: {failures}")
    return checks


def materialize(output: Path, base_cutoff: str, cutoff: str, repo: str) -> Path:
    validation = validate_collection(output, cutoff)
    derived = output / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    database = derived / "github_delta.sqlite"
    if database.exists():
        database.unlink()
    conn = sqlite3.connect(database)
    conn.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE artifact_raw (
      database_id INTEGER PRIMARY KEY, node_id TEXT, number INTEGER, artifact_type TEXT,
      created_at TEXT, updated_at TEXT, raw_json TEXT NOT NULL
    );
    CREATE TABLE issue_comment_raw (
      id INTEGER PRIMARY KEY, issue_url TEXT, created_at TEXT, updated_at TEXT, raw_json TEXT NOT NULL
    );
    CREATE TABLE review_comment_raw (
      id INTEGER PRIMARY KEY, pull_request_url TEXT, created_at TEXT, updated_at TEXT, raw_json TEXT NOT NULL
    );
    CREATE TABLE pull_request_raw (
      database_id INTEGER PRIMARY KEY, node_id TEXT, number INTEGER, created_at TEXT,
      updated_at TEXT, closed_at TEXT, merged_at TEXT, raw_json TEXT NOT NULL
    );
    CREATE TABLE pull_request_review_raw (
      database_id INTEGER PRIMARY KEY, pull_request_id INTEGER, submitted_at TEXT,
      updated_at TEXT, state TEXT, raw_json TEXT NOT NULL
    );
    CREATE TABLE pull_request_commit_raw (
      pull_request_id INTEGER, commit_sha TEXT, committed_at TEXT, raw_json TEXT NOT NULL,
      PRIMARY KEY (pull_request_id, commit_sha)
    );
    CREATE TABLE pull_request_file_raw (
      pull_request_id INTEGER, path TEXT, raw_json TEXT NOT NULL,
      PRIMARY KEY (pull_request_id, path)
    );
    CREATE TABLE timeline_event_raw (
      event_id TEXT PRIMARY KEY, artifact_id INTEGER, event_type TEXT, created_at TEXT, raw_json TEXT NOT NULL
    );
    CREATE TABLE default_branch_commit_raw (
      commit_sha TEXT PRIMARY KEY, committed_at TEXT, is_direct INTEGER, raw_json TEXT NOT NULL
    );
    CREATE TABLE request_file (
      path TEXT PRIMARY KEY, bytes INTEGER, sha256 TEXT, retrieved_at TEXT, request_url TEXT
    );
    CREATE INDEX timeline_artifact_at ON timeline_event_raw(artifact_id, created_at);
    CREATE INDEX review_pr_at ON pull_request_review_raw(pull_request_id, submitted_at);
    """)

    artifact_rows = list(all_rest_rows(output / "raw/rest/issues"))
    conn.executemany(
        "INSERT OR REPLACE INTO artifact_raw VALUES (?,?,?,?,?,?,?)",
        [(
            row["id"], row["node_id"], row["number"],
            "PullRequest" if "pull_request" in row else "Issue",
            row.get("created_at"), row.get("updated_at"), json.dumps(row, ensure_ascii=False),
        ) for row in artifact_rows],
    )
    comments = list(all_rest_rows(output / "raw/rest/issue_comments"))
    conn.executemany(
        "INSERT OR REPLACE INTO issue_comment_raw VALUES (?,?,?,?,?)",
        [(row["id"], row.get("issue_url"), row.get("created_at"), row.get("updated_at"), json.dumps(row, ensure_ascii=False)) for row in comments],
    )
    review_comments = list(all_rest_rows(output / "raw/rest/review_comments"))
    conn.executemany(
        "INSERT OR REPLACE INTO review_comment_raw VALUES (?,?,?,?,?)",
        [(row["id"], row.get("pull_request_url"), row.get("created_at"), row.get("updated_at"), json.dumps(row, ensure_ascii=False)) for row in review_comments],
    )

    nodes, tails = flatten_graphql(output)
    for node in nodes:
        if node["__typename"] != "PullRequest":
            continue
        pr_id = node["databaseId"]
        conn.execute(
            "INSERT OR REPLACE INTO pull_request_raw VALUES (?,?,?,?,?,?,?,?)",
            (pr_id, node["id"], node["number"], node.get("createdAt"), node.get("updatedAt"),
             node.get("closedAt"), node.get("mergedAt"), json.dumps(node, ensure_ascii=False)),
        )
        key = f"PullRequest:{node['number']}"
        reviews = node["reviews"]["nodes"] + tails.get((key, "reviews"), [])
        commits = node["commits"]["nodes"] + tails.get((key, "commits"), [])
        conn.executemany(
            "INSERT OR REPLACE INTO pull_request_review_raw VALUES (?,?,?,?,?,?)",
            [(r["databaseId"], pr_id, r.get("submittedAt"), r.get("updatedAt"), r.get("state"), json.dumps(r, ensure_ascii=False)) for r in reviews],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO pull_request_commit_raw VALUES (?,?,?,?)",
            [(pr_id, c["commit"]["oid"], c["commit"].get("committedDate"), json.dumps(c, ensure_ascii=False)) for c in commits],
        )

    number_to_id = {node["number"]: node["databaseId"] for node in nodes if node["__typename"] == "PullRequest"}
    for directory in sorted((output / "raw/rest/pull_request_files").glob("pr_*")):
        number = int(directory.name.removeprefix("pr_"))
        pr_id = number_to_id.get(number)
        if pr_id is None:
            continue
        conn.executemany(
            "INSERT OR REPLACE INTO pull_request_file_raw VALUES (?,?,?)",
            [(pr_id, row["filename"], json.dumps(row, ensure_ascii=False)) for row in all_rest_rows(directory)],
        )
    for path in sorted((output / "raw/git").glob("pr_*_files.json.gz")):
        fallback = read_gzip_json(path)["response"]["body"]
        number = int(path.name.removeprefix("pr_").removesuffix("_files.json.gz"))
        pr_id = number_to_id[number]
        conn.execute("DELETE FROM pull_request_file_raw WHERE pull_request_id = ?", (pr_id,))
        conn.executemany(
            "INSERT OR REPLACE INTO pull_request_file_raw VALUES (?,?,?)",
            [(pr_id, row["path"], json.dumps({**row, "source": "git_diff_fallback"}, ensure_ascii=False)) for row in fallback["files"]],
        )

    artifact_id = {node["id"]: node["databaseId"] for node in nodes}
    for node in nodes:
        key = f"{node['__typename']}:{node['number']}"
        events = node["timelineItems"]["nodes"] + tails.get((key, "timelineItems"), [])
        for event in events:
            if not event.get("id"):
                continue
            conn.execute(
                "INSERT OR REPLACE INTO timeline_event_raw VALUES (?,?,?,?,?)",
                (event["id"], artifact_id[node["id"]], event["__typename"], event.get("createdAt"), json.dumps(event, ensure_ascii=False)),
            )

    for path in sorted((output / "raw/graphql/default_branch_history").glob("page_*.json.gz")):
        history = response_body(path)["data"]["repository"]["defaultBranchRef"]["target"]["history"]
        for commit in history["nodes"]:
            is_direct = not bool(commit["associatedPullRequests"]["nodes"])
            conn.execute(
                "INSERT OR REPLACE INTO default_branch_commit_raw VALUES (?,?,?,?)",
                (commit["oid"], commit.get("committedDate"), int(is_direct), json.dumps(commit, ensure_ascii=False)),
            )

    raw_files = sorted((output / "raw").rglob("*.json.gz"))
    for path in raw_files:
        envelope = read_gzip_json(path)
        conn.execute(
            "INSERT INTO request_file VALUES (?,?,?,?,?)",
            (str(path.relative_to(output)), path.stat().st_size, sha256(path), envelope.get("retrieved_at"), envelope["request"].get("url")),
        )
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in [
            "artifact_raw", "issue_comment_raw", "review_comment_raw", "pull_request_raw",
            "pull_request_review_raw", "pull_request_commit_raw", "pull_request_file_raw",
            "timeline_event_raw", "default_branch_commit_raw", "request_file",
        ]
    }
    cutoff_counts = {
        "artifact_raw": conn.execute("SELECT COUNT(*) FROM artifact_raw WHERE created_at <= ?", (cutoff,)).fetchone()[0],
        "issue_comment_raw": conn.execute("SELECT COUNT(*) FROM issue_comment_raw WHERE created_at <= ?", (cutoff,)).fetchone()[0],
        "review_comment_raw": conn.execute("SELECT COUNT(*) FROM review_comment_raw WHERE created_at <= ?", (cutoff,)).fetchone()[0],
        "pull_request_raw": conn.execute("SELECT COUNT(*) FROM pull_request_raw WHERE created_at <= ?", (cutoff,)).fetchone()[0],
        "pull_request_review_raw": conn.execute("SELECT COUNT(*) FROM pull_request_review_raw WHERE submitted_at <= ?", (cutoff,)).fetchone()[0],
        "pull_request_commit_raw": conn.execute("SELECT COUNT(*) FROM pull_request_commit_raw WHERE committed_at <= ?", (cutoff,)).fetchone()[0],
        "timeline_event_raw": conn.execute("SELECT COUNT(*) FROM timeline_event_raw WHERE created_at <= ?", (cutoff,)).fetchone()[0],
        "default_branch_commit_raw": conn.execute("SELECT COUNT(*) FROM default_branch_commit_raw WHERE committed_at <= ?", (cutoff,)).fetchone()[0],
    }
    conn.close()

    raw_file_families: dict[str, dict[str, int]] = {}
    for path in raw_files:
        relative = path.relative_to(output)
        parts = relative.parts
        family = "/".join(parts[:3]) if len(parts) >= 3 else "/".join(parts[:-1])
        stats = raw_file_families.setdefault(family, {"files": 0, "compressed_bytes": 0})
        stats["files"] += 1
        stats["compressed_bytes"] += path.stat().st_size

    base_snapshot = output / "base/vllm_2026-05-18.sqlite"
    collector_path = Path(__file__).resolve()
    gh_lines = (
        subprocess.run(["gh", "--version"], capture_output=True, text=True, check=False).stdout.splitlines()
        if shutil.which("gh") else []
    )
    gh_version = gh_lines[0] if gh_lines else None

    manifest = {
        "repository": repo,
        "base_cutoff": base_cutoff,
        "analysis_cutoff": cutoff,
        "retrieved_at": utc_now(),
        "api_version": API_VERSION,
        "software": {
            "python": sys.version.split()[0],
            "requests": requests.__version__,
            "gh": gh_version,
            "collector_sha256": sha256(collector_path),
        },
        "raw_files": len(raw_files),
        "raw_bytes": sum(path.stat().st_size for path in raw_files),
        "sqlite": str(database.relative_to(output)),
        "sqlite_sha256": sha256(database),
        "counts": counts,
        "cutoff_consistent_counts": cutoff_counts,
        "raw_file_families": raw_file_families,
        "validation": validation,
        "base_snapshot": {
            "path": str(base_snapshot.relative_to(output)) if base_snapshot.exists() else None,
            "bytes": base_snapshot.stat().st_size if base_snapshot.exists() else None,
            "sha256": sha256(base_snapshot) if base_snapshot.exists() else None,
            "source_gist": "https://gist.github.com/simon-mo/2b0f4e9f872d479a08ae53edac51ecb1",
        },
        "collection_status": "complete",
        "canonical_scope": {
            "artifacts": "All issues and pull requests created by the cutoff and updated after the base snapshot; raw current representations are retained.",
            "comments": "Repository issue/PR conversation comments and inline review comments updated after the base snapshot.",
            "timeline_events": "State, label, assignment, duplicate, dependency, reference, merge, readiness, review-request/dismissal, ref-change, auto-merge, and merge-queue events for indexed artifacts.",
            "pull_requests": "Reviews, commits, and current aggregate changed-file lists for indexed pull requests.",
            "delivery": "Default-branch commits between the two cutoffs, with detailed file records for commits not associated with a pull request.",
        },
        "identity_and_text_warning": "Raw data contains public bodies, commit emails, and actor identifiers; do not commit it.",
        "known_limitations": [
            "The public API cannot list the complete collaborator roster without repository write-level membership.",
            "Content deleted before retrieval cannot be recovered.",
            "Bodies edited after the analysis cutoff are current API representations, not historical body snapshots.",
            "Current open-PR files can include changes after the cutoff; merged-by-cutoff PRs are stable for task-source analysis.",
            "Commit timestamps do not reveal when a commit was first pushed to an open pull request; cutoff filtering uses committedAt.",
            "Ten pathological pull requests required per-PR GraphQL recovery with reduced fields, including one REST commit-list fallback; validation records the exact field profiles.",
            "Pull requests that hit GitHub's 3000-file REST cap are replaced in the materialized table by an exact blobless git diff between the captured base and head commits.",
            "Project-board-only timeline fields require read:project and are outside the benchmark-maintenance event scope.",
            "Subscription, mention, and other notification/bookkeeping events are not collected as maintainer engineering actions.",
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    return database


def main() -> None:
    opt = arguments()
    opt.output.mkdir(parents=True, exist_ok=True)
    if not opt.materialize_only:
        token = subprocess.run(["gh", "auth", "token"], check=True, capture_output=True, text=True).stdout.strip()
        if not token:
            raise RuntimeError("No GitHub token returned by `gh auth token`")
        client = Client(token=token, output=opt.output, pause=opt.pause)
        collect_rest(client, opt.repo, opt.base_cutoff)
        artifacts = artifact_index(opt.output, opt.cutoff)
        with ThreadPoolExecutor(max_workers=2) as executor:
            graphql_future = executor.submit(
                collect_graphql, client, artifacts, opt.batch_size, opt.workers,
            )
            files_future = executor.submit(
                collect_pull_request_files_rest, client, opt.repo, artifacts, opt.workers,
            )
            graphql_future.result()
            files_future.result()
        collect_capped_file_fallbacks(client, opt.repo)
        collect_default_branch_history(
            client, opt.repo, opt.base_cutoff, opt.cutoff, opt.workers,
        )
        collect_repo_state(client, opt.repo, opt.cutoff)
    database = materialize(opt.output, opt.base_cutoff, opt.cutoff, opt.repo)
    print(json.dumps({"output": str(opt.output), "database": str(database)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
