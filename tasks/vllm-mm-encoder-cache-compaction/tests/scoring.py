"""Deterministic binary grading with continuous diagnostic scoring.

This module intentionally has no pytest, vLLM, CUDA, or Harbor dependency so
recorded test evidence can be replayed offline.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


PASS = "passed"
NON_PASS = {"failed", "error", "timeout", "skipped", "not_run"}
VALID_STATUSES = {PASS, *NON_PASS}


@dataclass(frozen=True)
class TestResult:
    node_id: str
    status: str
    duration_seconds: float = 0.0
    stdout_sha256: str = ""
    stderr_sha256: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PASS


def _mean(values: Iterable[float]) -> float:
    values = tuple(values)
    return sum(values) / len(values) if values else 0.0


def calculate_scores(
    manifest: Mapping[str, Any],
    results: Iterable[TestResult],
    *,
    validity_gate: int = 1,
) -> dict[str, Any]:
    """Calculate binary reward plus diagnostic requirement/group completion."""
    if validity_gate not in (0, 1):
        raise ValueError("validity_gate must be 0 or 1")

    score_groups = manifest.get("score_groups")
    requirements = manifest.get("requirements")
    core_cap = manifest.get("core_cap")
    if not isinstance(score_groups, Mapping) or not score_groups:
        raise ValueError("score_groups must be a non-empty mapping")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("requirements must be a non-empty list")
    if not isinstance(core_cap, Mapping):
        raise ValueError("core_cap must be a mapping")

    weights = {name: float(value) for name, value in score_groups.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("score-group weights must be finite and non-negative")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("score-group weights must sum to one")

    cap = float(core_cap.get("value", -1.0))
    if not math.isfinite(cap) or not 0.0 <= cap <= 1.0:
        raise ValueError("core cap must be finite and within [0, 1]")

    results = tuple(results)
    if any(item.status not in VALID_STATUSES for item in results):
        raise ValueError("unknown test status")
    by_node = {item.node_id: item for item in results}
    if len(by_node) != len(results):
        raise ValueError("duplicate test node IDs")

    requirement_ids = [str(item.get("id", "")) for item in requirements]
    if not all(requirement_ids) or len(set(requirement_ids)) != len(requirement_ids):
        raise ValueError("requirement IDs must be non-empty and unique")

    declared_nodes: list[str] = []
    for requirement in requirements:
        group = requirement.get("score_group")
        if group not in weights:
            raise ValueError(f"requirement uses unknown score group: {group}")
        nodes = requirement.get("test_node_ids")
        if not isinstance(nodes, list) or not nodes or not all(
            isinstance(node, str) and node for node in nodes
        ):
            raise ValueError(f"requirement {requirement.get('id')} has no valid tests")
        declared_nodes.extend(nodes)
    if len(set(declared_nodes)) != len(declared_nodes):
        raise ValueError("each scored pytest node must map to exactly one requirement")
    unknown_results = sorted(set(by_node) - set(declared_nodes))
    if unknown_results:
        raise ValueError(f"results contain undeclared test nodes: {unknown_results}")

    requirement_scores: dict[str, float] = {}
    groups: dict[str, list[float]] = {
        name: [] for name in weights
    }
    for requirement in requirements:
        node_ids = requirement["test_node_ids"]
        if not node_ids:
            raise ValueError(f"requirement {requirement['id']} has no tests")
        score = _mean(
            1.0 if by_node.get(node_id, TestResult(node_id, "not_run")).passed else 0.0
            for node_id in node_ids
        )
        requirement_scores[requirement["id"]] = score
        groups[requirement["score_group"]].append(score)

    group_scores = {name: _mean(scores) for name, scores in groups.items()}
    if any(not values for values in groups.values()):
        raise ValueError("every score group must have at least one requirement")
    weighted = sum(weights[name] * group_scores[name] for name in weights)

    sentinels = core_cap.get("apply_when_all_fail")
    if not isinstance(sentinels, list) or not sentinels:
        raise ValueError("core_cap.apply_when_all_fail must be non-empty")
    if any(item not in requirement_scores for item in sentinels):
        raise ValueError("core cap references an unknown requirement")
    core_cap_applied = all(requirement_scores.get(item, 0.0) == 0.0 for item in sentinels)
    if core_cap_applied:
        weighted = min(weighted, cap)

    raw_correctness = float(validity_gate) * weighted
    # Formal correctness is all-or-nothing. Partial capability information is
    # intentionally retained as raw_correctness and group scores, but cannot
    # make an incorrect implementation count as a successful task solve.
    reward = float(
        validity_gate == 1
        and math.isclose(raw_correctness, 1.0, rel_tol=0.0, abs_tol=1e-9)
    )
    return {
        "reward": reward,
        "raw_correctness": round(raw_correctness, 6),
        "validity_gate": validity_gate,
        **{name: round(value, 6) for name, value in group_scores.items()},
        "core_cap_applied": int(core_cap_applied),
        "requirement_scores": {
            name: round(value, 6) for name, value in requirement_scores.items()
        },
    }


def semantic_evidence(
    score: Mapping[str, Any],
    results: Iterable[TestResult],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Return repeatability evidence without timings or raw output hashes."""
    ordered = sorted(results, key=lambda item: item.node_id)
    return {
        "reward": score["reward"],
        "raw_correctness": score["raw_correctness"],
        "validity_gate": score["validity_gate"],
        "requirement_scores": score["requirement_scores"],
        "collected_test_ids": [item.node_id for item in ordered],
        "failed_test_ids": [item.node_id for item in ordered if not item.passed],
        "image_digest": environment.get("image_digest"),
        "gpu_model": environment.get("gpu_model"),
        "cuda": environment.get("cuda"),
        "structural_assertions": environment.get("structural_assertions", {}),
    }


def semantic_hash(evidence: Mapping[str, Any]) -> str:
    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def diagnostic_evidence(results: Iterable[TestResult]) -> dict[str, Any]:
    return {
        "tests": [
            {
                "node_id": item.node_id,
                "duration_seconds": item.duration_seconds,
                "stdout_sha256": item.stdout_sha256,
                "stderr_sha256": item.stderr_sha256,
            }
            for item in sorted(results, key=lambda item: item.node_id)
        ]
    }
