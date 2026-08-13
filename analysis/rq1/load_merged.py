"""Load cutoff-consistent analysis frames from the merged vLLM database."""

from __future__ import annotations

import sqlite3

import pandas as pd


def query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def load_merged_inputs(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """Return the frames expected by ``analyze.py`` plus PR-level file data.

    For event histories, delta-refreshed artifacts use the canonical timeline
    and all other artifacts use the original Fivetran history. This prevents
    double counting pre-base events returned again by the GitHub timeline API.
    """

    users = query(
        conn,
        """
        WITH inferred(id, login) AS (
          SELECT author_id, author_login FROM canonical_artifact
          UNION SELECT author_id, author_login FROM canonical_issue_comment
          UNION SELECT author_id, author_login FROM canonical_pull_request_review
          UNION SELECT author_id, author_login FROM canonical_review_comment
          UNION SELECT actor_id, actor_login FROM canonical_maintenance_event
          UNION SELECT subject_id, subject_login FROM canonical_maintenance_event
        )
        SELECT id, type FROM user
        UNION
        SELECT i.id,
               CASE
                 WHEN lower(COALESCE(i.login,'')) LIKE '%[bot]'
                   OR lower(COALESCE(i.login,'')) LIKE '%-bot'
                   OR lower(COALESCE(i.login,'')) IN
                     ('github-actions','dependabot','mergify','vllm-bot','codecov')
                 THEN 'Bot' ELSE 'User'
               END AS type
        FROM inferred i
        LEFT JOIN user u ON u.id=i.id
        WHERE i.id IS NOT NULL AND u.id IS NULL
        """,
    ).drop_duplicates("id", keep="first")

    frames: dict[str, pd.DataFrame] = {
        "users": users,
        "collaborators": query(
            conn,
            "SELECT user_id,role_name,pull,triage,push,maintain,admin,_fivetran_deleted "
            "FROM repo_collaborator",
        ),
        "all_issues": query(
            conn,
            """
            SELECT database_id AS id, created_at, updated_at_observed AS updated_at,
                   number, lower(CASE WHEN state_at_cutoff='MERGED' THEN 'closed'
                                     ELSE state_at_cutoff END) AS state,
                   lower(state_reason_at_cutoff) AS state_reason, title,
                   closed_at_cutoff AS closed_at,
                   CASE WHEN artifact_type='PullRequest' THEN 1 ELSE 0 END AS pull_request,
                   author_id AS user_id, representation_may_postdate_cutoff
            FROM canonical_artifact
            """,
        ),
        "prs_raw": query(
            conn,
            """
            SELECT database_id AS id, artifact_id AS issue_id, created_at,
                   closed_at_cutoff AS closed_at, is_draft_at_cutoff AS draft,
                   merge_commit_sha, base_sha, head_sha, files_cutoff_stable,
                   source_layer
            FROM canonical_pull_request
            """,
        ),
        "comments": query(
            conn,
            """
            SELECT database_id AS id, artifact_id AS issue_id, created_at,
                   author_id AS user_id
            FROM canonical_issue_comment
            WHERE artifact_id IS NOT NULL
            """,
        ),
        "reviews": query(
            conn,
            """
            SELECT database_id AS id, pull_request_id, submitted_at,
                   state, author_id AS user_id, commit_sha
            FROM canonical_pull_request_review
            """,
        ),
        "inline": query(
            conn,
            """
            SELECT database_id AS id, pull_request_id, review_id AS pull_request_review_id,
                   created_at, author_id AS user_id, path
            FROM canonical_review_comment
            WHERE pull_request_id IS NOT NULL
            """,
        ),
        "closed": query(
            conn,
            """
            SELECT closed, issue_id, updated_at, actor_id FROM issue_closed_history
            WHERE issue_id NOT IN (SELECT database_id FROM delta_artifact_raw)
            UNION ALL
            SELECT CASE WHEN event_type='ClosedEvent' THEN 1 ELSE 0 END AS closed,
                   artifact_id AS issue_id, created_at AS updated_at, actor_id
            FROM canonical_maintenance_event
            WHERE event_type IN ('ClosedEvent','ReopenedEvent')
            """,
        ),
        "merged": query(
            conn,
            """
            WITH observed AS (
              SELECT issue_id, merged_at, actor_id, commit_sha FROM issue_merged
              WHERE issue_id NOT IN (SELECT database_id FROM delta_artifact_raw)
              UNION ALL
              SELECT artifact_id AS issue_id, created_at AS merged_at, actor_id, commit_sha
              FROM canonical_maintenance_event WHERE event_type='MergedEvent'
            )
            SELECT issue_id,merged_at,actor_id,commit_sha FROM observed
            UNION ALL
            SELECT p.artifact_id AS issue_id,p.merged_at_cutoff AS merged_at,
                   NULL AS actor_id,p.merge_commit_sha AS commit_sha
            FROM canonical_pull_request p
            WHERE p.merged_at_cutoff IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM observed o WHERE o.issue_id=p.artifact_id)
            """,
        ),
        "ready": query(
            conn,
            """
            SELECT created_at,pull_request_id,ready_for_review,actor_id
            FROM pull_request_ready_for_review_history
            WHERE pull_request_id NOT IN (SELECT database_id FROM delta_pull_request_raw)
            UNION ALL
            SELECT e.created_at,p.database_id AS pull_request_id,
                   CASE WHEN e.event_type='ReadyForReviewEvent' THEN 1 ELSE 0 END,
                   e.actor_id
            FROM canonical_maintenance_event e
            JOIN canonical_pull_request p ON p.artifact_id=e.artifact_id
            WHERE e.event_type IN ('ReadyForReviewEvent','ConvertToDraftEvent')
            """,
        ),
        "issue_labels_named": query(
            conn,
            "SELECT artifact_id AS issue_id,label_name AS name FROM canonical_artifact_label",
        ),
        "commit_pr": query(
            conn,
            "SELECT commit_sha,pull_request_id FROM canonical_pull_request_commit",
        ),
        "commit_files": query(
            conn,
            "SELECT commit_sha,filename,additions,deletions,changes FROM commit_file",
        ),
        "pr_files": query(
            conn,
            """
            SELECT pull_request_id,path AS filename,additions,deletions,changes,
                   cutoff_stable
            FROM canonical_pull_request_file
            """,
        ),
        "issue_refs": query(
            conn,
            """
            SELECT issue_id,referenced_at,commit_sha,actor_id FROM issue_referenced
            WHERE issue_id NOT IN (SELECT database_id FROM delta_artifact_raw)
            UNION ALL
            SELECT artifact_id AS issue_id,created_at AS referenced_at,commit_sha,actor_id
            FROM canonical_maintenance_event
            WHERE event_type='ReferencedEvent' AND commit_sha IS NOT NULL
            """,
        ),
        "label_history": query(
            conn,
            """
            SELECT issue_id,updated_at,actor_id,labeled FROM issue_label_history
            WHERE issue_id NOT IN (SELECT database_id FROM delta_artifact_raw)
            UNION ALL
            SELECT artifact_id AS issue_id,created_at AS updated_at,actor_id,
                   CASE WHEN event_type='LabeledEvent' THEN 1 ELSE 0 END AS labeled
            FROM canonical_maintenance_event
            WHERE event_type IN ('LabeledEvent','UnlabeledEvent')
            """,
        ),
        "issue_assignees": query(
            conn,
            "SELECT artifact_id AS issue_id,user_id FROM canonical_artifact_assignee "
            "WHERE user_id IS NOT NULL",
        ),
        "reviewer_requests": query(
            conn,
            """
            SELECT created_at,pull_request_id,requested_id,actor_id,removed,
                   requested_reviewer_type
            FROM requested_reviewer_history
            WHERE pull_request_id NOT IN (SELECT database_id FROM delta_pull_request_raw)
            UNION ALL
            SELECT e.created_at,p.database_id AS pull_request_id,e.subject_id AS requested_id,
                   e.actor_id,
                   CASE WHEN e.event_type='ReviewRequestRemovedEvent' THEN 1 ELSE 0 END,
                   'User'
            FROM canonical_maintenance_event e
            JOIN canonical_pull_request p ON p.artifact_id=e.artifact_id
            WHERE e.event_type IN ('ReviewRequestedEvent','ReviewRequestRemovedEvent')
              AND e.subject_id IS NOT NULL
            """,
        ),
        "direct_main_commits": query(
            conn,
            """
            SELECT substr(committed_at,1,4) AS year,COUNT(*) AS commits
            FROM canonical_default_branch_commit WHERE is_direct=1
            GROUP BY 1 ORDER BY 1
            """,
        ),
        "git_commit_identity_audit": query(
            conn,
            """
            WITH commits AS (
              SELECT commit_sha,MAX(author_email) AS author_email,
                     MAX(committer_email) AS committer_email
              FROM canonical_pull_request_commit GROUP BY commit_sha
            )
            SELECT COUNT(*) AS commits,
                   COUNT(DISTINCT author_email) AS distinct_author_emails,
                   COUNT(DISTINCT committer_email) AS distinct_committer_emails,
                   SUM(CASE WHEN author_email=committer_email THEN 1 ELSE 0 END)
                     AS same_author_committer_email
            FROM commits
            """,
        ),
        "input_audit": query(
            conn,
            """
            SELECT 'canonical_artifacts' AS check_name,COUNT(*) AS value
              FROM canonical_artifact
            UNION ALL SELECT 'canonical_pull_requests',COUNT(*) FROM canonical_pull_request
            UNION ALL SELECT 'canonical_comments',COUNT(*) FROM canonical_issue_comment
            UNION ALL SELECT 'canonical_reviews',COUNT(*) FROM canonical_pull_request_review
            UNION ALL SELECT 'canonical_inline_comments',COUNT(*) FROM canonical_review_comment
              WHERE pull_request_id IS NOT NULL
            UNION ALL SELECT 'orphan_inline_comments',COUNT(*) FROM canonical_review_comment
              WHERE pull_request_id IS NULL
            UNION ALL SELECT 'canonical_pr_commit_associations',COUNT(*) FROM canonical_pull_request_commit
            UNION ALL SELECT 'canonical_pr_files',COUNT(*) FROM canonical_pull_request_file
            UNION ALL SELECT 'canonical_merged_prs',COUNT(*) FROM canonical_pull_request
              WHERE merged_at_cutoff IS NOT NULL
            UNION ALL SELECT 'generic_closed_state_with_merge_time',COUNT(*)
              FROM canonical_pull_request p
              JOIN canonical_artifact a ON a.database_id=p.artifact_id
              WHERE p.merged_at_cutoff IS NOT NULL AND a.state_at_cutoff='CLOSED'
            UNION ALL SELECT 'release_validation_failures',COUNT(*) FROM dataset_validation
              WHERE passed=0
            UNION ALL SELECT 'delta_refreshed_artifacts',COUNT(*) FROM delta_artifact_raw
            UNION ALL SELECT 'post_cutoff_representations',COUNT(*) FROM canonical_artifact
              WHERE representation_may_postdate_cutoff=1
            UNION ALL SELECT 'unstable_open_pr_file_snapshots',COUNT(*)
              FROM canonical_pull_request WHERE files_cutoff_stable=0
            UNION ALL SELECT 'inferred_users_absent_from_base',COUNT(*) FROM (
              SELECT author_id AS id FROM canonical_artifact
              UNION SELECT author_id FROM canonical_issue_comment
              UNION SELECT author_id FROM canonical_pull_request_review
              UNION SELECT author_id FROM canonical_review_comment
              UNION SELECT actor_id FROM canonical_maintenance_event
            ) x LEFT JOIN user u ON u.id=x.id WHERE x.id IS NOT NULL AND u.id IS NULL
            UNION ALL SELECT 'materialized_merge_fallbacks',COUNT(*)
              FROM canonical_pull_request p
              WHERE p.merged_at_cutoff IS NOT NULL
                AND NOT EXISTS (
                  SELECT 1 FROM issue_merged m
                  WHERE m.issue_id=p.artifact_id
                    AND p.artifact_id NOT IN (SELECT database_id FROM delta_artifact_raw)
                )
                AND NOT EXISTS (
                  SELECT 1 FROM canonical_maintenance_event e
                  WHERE e.artifact_id=p.artifact_id AND e.event_type='MergedEvent'
                )
            """,
        ),
    }
    return frames
